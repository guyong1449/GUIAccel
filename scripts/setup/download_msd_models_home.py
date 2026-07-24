#!/usr/bin/env python3
"""Download MSD models via HF mirror (no proxy needed on DKUCC)."""

import os
import sys

# Use HF mirror so no proxy needed
endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = endpoint

# No proxy — mirror is directly accessible from DKUCC
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(var, None)

# Longer timeouts for large files
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download

hf_home = os.environ.get("MSD_HF_HOME", "/dkucc/home/rw335/.cache/huggingface")
os.makedirs(hf_home, exist_ok=True)

models = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "lucylyn/MSD-Qwen2VL-7B-Instruct",
]

for repo_id in models:
    print(f"\n=== Downloading {repo_id} ===")
    sys.stdout.flush()
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=os.path.join(hf_home, "hub"),
            local_dir_use_symlinks=False,
        )
        print(f"=== {repo_id} complete ===")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"=== {repo_id} FAILED: {e} ===", file=sys.stderr)
        sys.exit(1)

print(f"\nAll models cached under {hf_home}/hub")
