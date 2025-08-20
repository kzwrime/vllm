# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch

from vllm import envs


class IPEXFusedMOE:

    def __init__(self, layer: torch.nn.Module) -> None:
        import intel_extension_for_pytorch as ipex
        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,
            layer.w2_weight,
            use_prepack=envs.VLLM_CPU_MOE_PREPACK,
        )

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
        )


class SGLFusedMOE:

    def __init__(self, layer: torch.nn.Module) -> None:
        pass

    @staticmethod
    def _grouped_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[0] == gating_output.shape[0], (
            "Number of tokens mismatch")

        gating_output = gating_output.float()
        if scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

        num_token = scores.shape[0]
        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use
            # biased scores for expert selection but original scores for
            # routing weights
            original_scores = scores
            scores = scores + e_score_correction_bias.unsqueeze(0)
            group_scores = (scores.view(num_token, num_expert_group,
                                        -1).topk(2, dim=-1)[0].sum(dim=-1))
        else:
            group_scores = scores.view(num_token, num_expert_group,
                                       -1).max(dim=-1).values  # [n, n_group]
        group_idx = torch.topk(group_scores,
                               k=topk_group,
                               dim=-1,
                               sorted=False)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = group_mask.unsqueeze(-1).expand(
            num_token, num_expert_group,
            scores.shape[-1] // num_expert_group).reshape(num_token,
                                                          -1)  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(),
                                        float("-inf"))  # [n, e]

        if e_score_correction_bias is not None:
            topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
            # Use original unbiased scores for the routing weights
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(tmp_scores,
                                                k=topk,
                                                dim=-1,
                                                sorted=False)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                           keepdim=True)

        return topk_weights, topk_ids.to(torch.int32)

    @staticmethod
    def _select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = SGLFusedMOE._grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        elif custom_routing_function is None:
            assert scoring_func == "softmax"
            topk_weights = torch.nn.functional.softmax(router_logits,
                                                       dim=1,
                                                       dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            if renormalize:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_ids = topk_ids.to(torch.int32)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        topk_weights, topk_ids = SGLFusedMOE._select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.ops._C.fused_experts_cpu(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            True,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            True,
        )
        return x

class SWFusedMOE:

    def __init__(self, layer: torch.nn.Module) -> None:
        pass

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        # SW impl currently supports only plain top-k softmax routing
        assert not use_grouped_topk
        assert custom_routing_function is None
        assert scoring_func == "softmax"
        assert e_score_correction_bias is None

        topk_weights, topk_ids = self._select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            expert_map=expert_map if expert_map is not None else None,
            global_num_experts=global_num_experts,
        )

        y = self._forward(
            x=x,
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        return y

    def _select_experts(
        self,
        hidden_states : torch.Tensor,
        router_logits : torch.Tensor,
        top_k : int,
        use_grouped_topk : bool,
        renormalize: bool,
        expert_map : torch.Tensor,
        global_num_experts : int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[0] == router_logits.shape[0]
        if expert_map is not None:
            assert expert_map.shape[0] == global_num_experts

        # Only plain top-k is supported here
        assert not use_grouped_topk

        # Compute softmax scores and take per-token top-k experts
        scores = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # If an expert map is provided (global -> local, -1 for non-local), map ids
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]
        # topk_weights : [num_tokens, top_k]
        # topk_ids : [num_tokens, top_k]
        return topk_weights, topk_ids.to(torch.int32)

    def _forward(
        self,
        x : torch.Tensor,
        w13_weight : torch.Tensor,
        w2_weight : torch.Tensor,
        topk_weights : torch.Tensor,
        topk_ids : torch.Tensor,
    ) -> torch.Tensor:
        # x: [num_tokens, hidden_size]
        # w13_weight: [num_experts, 2 * intermediate_size, hidden_size]
        # w2_weight: [num_experts, hidden_size, intermediate_size]
        # topk_ids: [num_tokens, top_k] (expert ids, may contain -1 for non-local)
        # topk_weights: [num_tokens, top_k]

        num_tokens, hidden_size = x.shape
        assert w13_weight.shape[-1] == hidden_size
        num_experts, two_intermediate, _ = w13_weight.shape
        intermediate_size = two_intermediate // 2

        # Prepare output buffer
        out = torch.zeros_like(x)

        # Compute in parameter dtype for correctness/perf, then cast back
        compute_dtype = w13_weight.dtype
        x_compute = x.to(compute_dtype)
        topk_weights_compute = topk_weights.to(compute_dtype)

        # Loop over k selections and accumulate contributions
        top_k = topk_ids.shape[1]
        for k in range(top_k):
            expert_ids_k = topk_ids[:, k].long()  # [N]
            weights_k = topk_weights_compute[:, k]  # [N]

            # Mask invalid experts (-1 means non-local)
            valid_mask = expert_ids_k >= 0
            if not torch.any(valid_mask):
                continue

            expert_ids_valid = expert_ids_k[valid_mask]
            token_indices_valid = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            weights_valid = weights_k[valid_mask]
           
            # Group tokens by expert id for batched matmul
            unique_experts = torch.unique(expert_ids_valid)
            for e in unique_experts.tolist():
                token_mask_e = (expert_ids_valid == e)
                if not torch.any(token_mask_e):
                    continue
                token_idx_e = token_indices_valid[token_mask_e]

                # Gather inputs and weights for these tokens
                x_e = x_compute.index_select(0, token_idx_e)  # [N_e, H]
                w13_e = w13_weight[e]  # [2I, H]
                # Gate+Up projection
                gate_up = x_e @ w13_e.t()  # [N_e, 2I]
                gate, up = gate_up.split(intermediate_size, dim=-1)
                inter = torch.nn.functional.silu(gate) * up  # [N_e, I]

                # Down projection
                w2_e = w2_weight[e]  # [H, I]
                y_e = inter @ w2_e.t()  # [N_e, H]

                # Apply routing weights and accumulate
                w_e = weights_valid[token_mask_e].unsqueeze(-1)  # [N_e, 1]
                y_weighted = y_e * w_e  # [N_e, H]
                # Accumulate into output (in compute dtype)
                out.index_add_(0, token_idx_e, y_weighted.to(out.dtype))

        return out
