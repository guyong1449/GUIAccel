#!/usr/bin/env bash
# Download MSD models from hf-mirror.com using wget
# Usage: bash download_msd_models_direct.sh
set -Eeuo pipefail

MIRROR="https://hf-mirror.com"
BASE="/dkucc/home/rw335/GUIAccel/models"
mkdir -p "${BASE}"

download_model() {
    local repo_id="$1"
    local dest="$2"
    mkdir -p "${dest}"

    echo "================================================"
    echo "Downloading ${repo_id}"
    echo "  -> ${dest}"
    echo "================================================"

    files=$(curl -s --connect-timeout 10 --max-time 30 \
        "${MIRROR}/api/models/${repo_id}" \
        | python3 -c "import json,sys; [print(s['rfilename']) for s in json.load(sys.stdin).get('siblings',[])]")

    echo "$files" | while IFS= read -r f; do
        [ -z "$f" ] && continue
        url="${MIRROR}/${repo_id}/resolve/main/${f}"
        out="${dest}/${f}"
        mkdir -p "$(dirname "${out}")"

        if [ -f "${out}" ] && [ "$(stat -c%s "${out}" 2>/dev/null || echo 0)" -gt 0 ]; then
            echo "  ✓ ${f}"
            continue
        fi

        echo "  ↓ ${f}"
        wget --timeout=30 --tries=3 --retry-connrefused --continue \
            -q --show-progress -O "${out}" "${url}" 2>&1 \
            || echo "  ✗ ${f} FAILED"
    done

    echo "=== Done ==="
    du -sh "${dest}"
    echo ""
}

download_model "Qwen/Qwen2-VL-7B-Instruct" "${BASE}/Qwen2-VL-7B-Instruct"
download_model "lucylyn/MSD-Qwen2VL-7B-Instruct" "${BASE}/MSD-Qwen2VL-7B-Instruct"

echo "================================================"
echo "All done! Models saved to:"
du -sh "${BASE}"
