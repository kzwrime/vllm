# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import AttentionConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ReqMeta,
)
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

MODEL_NAME = "RedHatAI/Qwen2.5-VL-3B-Instruct-quantized.w8a8"

SAMPLING_PARAMS = SamplingParams(temperature=0.0, top_k=1, max_tokens=128)

TEXT_PROMPTS = [
    "What's in the image(s)? Around 30 words. What's special in 2nd image?",
    "The future of AI is",
]


def test_req_meta_caps_tokens_to_allocated_blocks():
    token_ids = list(range(704))
    block_ids = list(range(16))
    block_size = 16

    meta = ReqMeta.make_meta(
        token_ids=token_ids,
        block_ids=block_ids,
        block_size=block_size,
        is_store=True,
        mm_hashes=[],
    )

    assert len(meta.token_ids) == 256
    assert len(meta.slot_mapping) == 256
    assert meta.token_ids.tolist() == token_ids[:256]
    assert meta.slot_mapping.tolist() == list(range(256))


def test_example_connector_publishes_only_complete_prefill(tmp_path):
    block_size = 16
    prompt_token_ids = list(range(704))
    connector = ExampleConnector(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=block_size),
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
            ),
        ),
        KVConnectorRole.SCHEDULER,
    )

    first_chunk = SchedulerOutput.make_empty()
    first_chunk.scheduled_new_reqs = [
        NewRequestData(
            req_id="req-0",
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(list(range(16)),),
            num_computed_tokens=0,
            lora_request=None,
        )
    ]
    first_chunk.num_scheduled_tokens = {"req-0": 256}
    assert connector.build_connector_meta(first_chunk).requests == []

    second_chunk = SchedulerOutput.make_empty()
    second_chunk.scheduled_cached_reqs = CachedRequestData(
        req_ids=["req-0"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[(list(range(16, 32)),)],
        num_computed_tokens=[256],
        num_output_tokens=[0],
    )
    second_chunk.num_scheduled_tokens = {"req-0": 256}
    assert connector.build_connector_meta(second_chunk).requests == []

    final_chunk = SchedulerOutput.make_empty()
    final_chunk.scheduled_cached_reqs = CachedRequestData(
        req_ids=["req-0"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[(list(range(32, 44)),)],
        num_computed_tokens=[512],
        num_output_tokens=[0],
    )
    final_chunk.num_scheduled_tokens = {"req-0": 192}

    meta = connector.build_connector_meta(final_chunk)
    assert len(meta.requests) == 1
    request_meta = meta.requests[0]
    assert request_meta.is_store
    assert len(request_meta.token_ids) == 704
    assert len(request_meta.slot_mapping) == 704
    assert request_meta.token_ids.tolist() == prompt_token_ids


def test_example_connector_replaces_blocks_on_resume(tmp_path):
    block_size = 16
    prompt_token_ids = list(range(704))
    connector = ExampleConnector(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=block_size),
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
            ),
        ),
        KVConnectorRole.SCHEDULER,
    )

    first_chunk = SchedulerOutput.make_empty()
    first_chunk.scheduled_new_reqs = [
        NewRequestData(
            req_id="req-0",
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(list(range(16)),),
            num_computed_tokens=0,
            lora_request=None,
        )
    ]
    first_chunk.num_scheduled_tokens = {"req-0": 256}
    assert connector.build_connector_meta(first_chunk).requests == []

    resumed_chunk = SchedulerOutput.make_empty()
    resumed_chunk.scheduled_cached_reqs = CachedRequestData(
        req_ids=["req-0"],
        resumed_req_ids={"req-0"},
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[(list(range(100, 144)),)],
        num_computed_tokens=[256],
        num_output_tokens=[0],
    )
    resumed_chunk.num_scheduled_tokens = {"req-0": 448}

    meta = connector.build_connector_meta(resumed_chunk)
    assert len(meta.requests) == 1
    request_meta = meta.requests[0]
    assert request_meta.is_store
    assert len(request_meta.token_ids) == 704
    assert request_meta.slot_mapping[:16].tolist() == list(range(1600, 1616))
    assert request_meta.slot_mapping[-16:].tolist() == list(range(2288, 2304))


class InputCase(NamedTuple):
    text: str
    img: list[Image]
    expected_len: int
    info: str


def _check_path_len(path):
    """Return the latest length in path"""
    return len(list(path.iterdir()))


def _list_path(path):
    """Return the list of foldername (hashes generated) under the path"""
    return list(path.iterdir())


def run_test(
    tmp_path,
    processor,
    llm: LLM,
    question: str,
    image_urls: list[Image],
    expected_len: int,
    info: str,
):
    """
    One individual test to process the prompt and output base on 1 set of input
    Then check if the length in the storage path matches the expected length
    `info` introduces details or purpose of the individual test
    """
    print(f"***info: {info}***")
    print(f"**Expected storage path length after llm generate: {expected_len}**")
    process_prompt(processor, llm, question, image_urls)

    print(f"Path matched expected length: {_check_path_len(tmp_path)}")
    print(f"Hashes under the storage path: {_list_path(tmp_path)}")

    assert _check_path_len(tmp_path) == expected_len, (
        f"Expect storage path length {expected_len} ;",
        f"but end up {_check_path_len(tmp_path)} instead. ",
        f"Info: {info}",
    )


def process_prompt(processor, llm: LLM, question: str, image_urls: list[Image]):
    """
    Form the prompt based on the text and image input, then llm generate output
    """
    placeholders = [
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(image_pil)},
        }
        for image_pil in image_urls
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            **({"multi_modal_data": {"image": [*image_urls]}} if image_urls else {}),
        },
        sampling_params=SAMPLING_PARAMS,
    )

    print("-" * 50)
    print("Output:")
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


@pytest.mark.parametrize(
    "attn_backend",
    (
        ["FLASH_ATTN", "TRITON_ATTN"]
        if current_platform.is_cuda()
        else ["TRITON_ATTN"]
        if current_platform.is_rocm()
        else []
    ),
)
def test_shared_storage_connector_hashes(tmp_path, attn_backend):
    """
    Tests that ExampleConnector saves KV to the storage locations
    with proper hashes; that are unique for inputs with identical text but
    different images (same size), or same multiple images but different orders.
    """
    # Using tmp_path as the storage path to store KV
    print(f"KV storage path at: {str(tmp_path)}")

    # don't put this import at the top level
    # it will call torch.accelerator.device_count()
    from transformers import AutoProcessor

    # Create processor to handle the chat prompt
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Prepare images for the tests
    # Resize to the same size to check hashes correctness
    image_1 = ImageAsset("stop_sign").pil_image.resize((1280, 720))
    image_2 = ImageAsset("cherry_blossom").pil_image.resize((1280, 720))

    # Make sure that they are not the same picture
    assert image_1 != image_2, "The images should not be identical"

    # Create the LLM instance
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        attention_config=AttentionConfig(backend=attn_backend),
        enforce_eager=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
        limit_mm_per_prompt={"image": 2},
    )

    # Prepare the input cases
    input_cases = [
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=1,
            info="image_1 single input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the first time. "
                "It is in same pixel size with image_1, yet it "
                "should be able to form a new unique hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=2,
            info=(
                "image_1 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=3,
            info="image_1 with image_2 input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info="The image order is swapped. Should form new hash.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=4,
            info=(
                "[image_1, image_2] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info=(
                "[image_2, image_1] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Pure text input test as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Identical pure text input as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[1],
            img=[],
            expected_len=6,
            info="Another pure text input as a case-control",
        ),
    ]

    # Run tests
    for case_id, (text, img, expected_len, info) in enumerate(input_cases):
        print("\n", "=" * 25, f"Below running input case: {case_id}", "=" * 25)
        run_test(tmp_path, processor, llm, text, img, expected_len, info)

    print("All tests passed successfully!")
