#!/usr/bin/env bash
# Apply this fork's ROCm/gfx1151 source patches (patches/) to upstream checkouts.
# Idempotent: a patch that is already applied is skipped. See docs/amd_rocm.md.
#
# Usage:
#   scripts/apply_rocm_patches.sh pytorch    <pytorch-src>      # PyTorch v2.7.0
#   scripts/apply_rocm_patches.sh flash-attn <flash-attn-src>   # ROCm/flash-attention (+ third_party/aiter)
set -euo pipefail

PATCHES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../patches" && pwd)"

apply() {  # apply <repo-dir> <patch>
    local dir="$1" patch="$2"
    if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied: $(basename "$patch") in $dir"
    else
        git -C "$dir" apply "$patch"
        echo "applied: $(basename "$patch") in $dir"
    fi
}

case "${1:-}" in
    pytorch)
        src="${2:?usage: $0 pytorch <pytorch-src>}"
        apply "$src" "$PATCHES/pytorch-v2.7.0-rocm-gfx1151.patch"
        # CK's cmake normally generates ck/config.h; the CK sources are excluded from this
        # build, so it never does, but other CK headers still include it.
        install -D -m 644 "$PATCHES/composable_kernel-ck-config.h" \
            "$src/third_party/composable_kernel/include/ck/config.h"
        echo "installed: third_party/composable_kernel/include/ck/config.h"
        ;;
    flash-attn)
        src="${2:?usage: $0 flash-attn <flash-attn-src>}"
        if [ ! -d "$src/third_party/aiter/.git" ] && [ ! -f "$src/third_party/aiter/.git" ]; then
            echo "error: $src/third_party/aiter is missing; clone ROCm/aiter there first" >&2
            exit 1
        fi
        apply "$src" "$PATCHES/flash-attention-rocm.patch"
        apply "$src/third_party/aiter" "$PATCHES/aiter-rocm.patch"
        ;;
    *)
        sed -n '2,8p' "$0" >&2
        exit 2
        ;;
esac
