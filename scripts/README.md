# Test "[Config] Isolate DP master ip and head ip configurations in headless mode."

Copy and edit if needed.

```bash
cp scripts/env_template.sh scripts/env.sh
cp scripts/user_env_template.sh scripts/user_env.sh
```

Node 0: (API Server), IP: 172.33.0.10

```bash
bash ./scripts/serve_head_only_template.sh
```

Node 1: (DP 0), IP: 172.33.0.11

```bash
RANK=0 bash ./scripts/serve_headless_dp_only_template.sh
```

Node 2: (DP 2), IP: 172.33.0.12

```bash
RANK=1 bash ./scripts/serve_headless_dp_only_template.sh
```