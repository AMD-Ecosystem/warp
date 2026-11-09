#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Verify the running environment matches the known-good gfx1151 baseline
# documented in README-AMD.md. Exit 0 = all checks pass.
#
# Usage:
#   bash tools/check_gfx1151_baseline.sh
#   bash tools/check_gfx1151_baseline.sh --strict   # also fail on missing kernel pin

set -u

STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

PASS=0
FAIL=0
WARN=0

ok()   { echo "  OK    $*"; PASS=$((PASS+1)); }
fail() { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  WARN  $*"; WARN=$((WARN+1)); }

section() { echo; echo "=== $* ==="; }

# Expected values
EXPECTED_KERNEL="6.17.0-1017-oem"
EXPECTED_SRCVERSION="FC7DA320ED9D733CA6A3F1E"
EXPECTED_GFX="gfx1151"
EXPECTED_MUJOCO="3.5.0"
EXPECTED_MUJOCO_WARP="3.5.0.2"
EXPECTED_NEWTON="1.0.0"
EXPECTED_TORCH_PREFIX="2.11.0+rocm7"

# ---------------------------------------------------------------- kernel
section "Kernel + amdgpu module"

KERNEL=$(uname -r)
if [ "$KERNEL" = "$EXPECTED_KERNEL" ]; then
    ok "uname -r = $KERNEL"
else
    fail "uname -r = $KERNEL (expected $EXPECTED_KERNEL)"
    echo "        See README-AMD.md > Troubleshooting > 'hsa_queue_create page fault'"
fi

SRC=$(modinfo amdgpu 2>/dev/null | awk '/^srcversion:/ {print $2}')
if [ "$SRC" = "$EXPECTED_SRCVERSION" ]; then
    ok "amdgpu srcversion = $SRC"
else
    fail "amdgpu srcversion = ${SRC:-<not loaded>} (expected $EXPECTED_SRCVERSION)"
fi

MOD_PATH=$(modinfo amdgpu 2>/dev/null | awk '/^filename:/ {print $2}')
if echo "$MOD_PATH" | grep -q '/updates/dkms/'; then
    fail "amdgpu loaded from DKMS path: $MOD_PATH (should be in-tree)"
elif [ -n "$MOD_PATH" ]; then
    ok "amdgpu loaded in-tree: $MOD_PATH"
else
    fail "amdgpu module not loaded"
fi

# ---------------------------------------------------------------- GPU detect
section "GPU detection (rocminfo)"

if command -v rocminfo >/dev/null 2>&1; then
    ARCH=$(rocminfo 2>/dev/null | awk '/Name:.*gfx/ {print $2; exit}')
    if [ "$ARCH" = "$EXPECTED_GFX" ]; then
        ok "rocminfo reports $ARCH"
    elif [ -n "$ARCH" ]; then
        warn "rocminfo reports $ARCH (expected $EXPECTED_GFX — may still work)"
    else
        fail "rocminfo did not report any gfx device"
    fi
else
    warn "rocminfo not on PATH — skipping (install rocminfo or fix PATH)"
fi

# ---------------------------------------------------------------- python stack
section "Python stack"

if ! command -v python >/dev/null 2>&1; then
    fail "no python on PATH — activate your conda env first"
else
    # Use pip metadata so we don't import warp (which can hang on broken kernels)
    declare -A want=( \
        [mujoco]="$EXPECTED_MUJOCO" \
        [mujoco-warp]="$EXPECTED_MUJOCO_WARP" \
        [newton]="$EXPECTED_NEWTON" \
    )
    while IFS=' ' read -r pkg ver _; do
        case "$pkg" in
            mujoco|mujoco-warp|newton)
                if [ "$ver" = "${want[$pkg]}" ]; then
                    ok "$pkg = $ver"
                else
                    fail "$pkg = $ver (expected ${want[$pkg]})"
                    echo "        pip install $pkg==${want[$pkg]} --force-reinstall --no-deps"
                fi
                ;;
            torch)
                if echo "$ver" | grep -q "^$EXPECTED_TORCH_PREFIX"; then
                    ok "torch = $ver"
                else
                    fail "torch = $ver (expected starting with $EXPECTED_TORCH_PREFIX)"
                fi
                ;;
            warp-lang)
                ok "warp-lang = $ver"
                ;;
        esac
    done < <(pip list 2>/dev/null | awk 'NR>2 {print tolower($1), $2, $3}')

    EDITABLE=$(pip show warp-lang 2>/dev/null | awk -F': ' '/^Editable project location:/ {print $2}')
    if [ -n "$EDITABLE" ]; then
        ok "warp-lang editable at: $EDITABLE"
        if [ -d "$EDITABLE/.git" ]; then
            TAG=$(cd "$EDITABLE" && git describe --tags --always 2>/dev/null)
            HEAD=$(cd "$EDITABLE" && git log --oneline -1 2>/dev/null)
            echo "          git: $HEAD"
            echo "          tag: $TAG"
        fi
    else
        warn "warp-lang not editable-installed (pip wheel?) — env may differ from validated baseline"
    fi
fi

# ---------------------------------------------------------------- Newton patches
section "Newton HIP patches"

# Don't import newton — its __init__ can pull in warp + HIP runtime, which
# hangs on the broken kernel. Find the installed dir via filesystem.
NEWTON_DIR=""
for site in $(python -c 'import sys; print("\n".join(sys.path))' 2>/dev/null); do
    if [ -d "$site/newton/_src/solvers/mujoco" ]; then
        NEWTON_DIR="$site/newton"
        break
    fi
done
if [ -z "$NEWTON_DIR" ]; then
    warn "could not locate installed Newton"
else
    # MuJoCo graph_conditional patch (04): needed for stock MuJoCo examples on HIP.
    if grep -q 'is_conditional_graph_supported\|graph_conditional = False' \
        "$NEWTON_DIR/_src/solvers/mujoco/solver_mujoco.py" 2>/dev/null; then
        ok "MuJoCo graph_conditional patch applied"
    else
        warn "MuJoCo graph_conditional patch not applied — stock MuJoCo examples (ANYmal, ...) will error on HIP"
        echo "        Apply patches/newton/04-solver_mujoco-hip-graph-conditional.patch — see README-AMD.md"
    fi
    # Note: the VBD wave32 patches (01/02) are no longer required — ROCm 7.2 runs
    # one workgroup per wave, so sub-warp wp.tile_reduce is already correct.
fi

# ---------------------------------------------------------------- apt holds
section "Apt holds"

HOLDS=$(apt-mark showhold 2>/dev/null)
if echo "$HOLDS" | grep -q "^amdgpu-dkms-firmware$"; then
    ok "amdgpu-dkms-firmware is held"
else
    warn "amdgpu-dkms-firmware not held — DKMS could get reintroduced"
fi
if echo "$HOLDS" | grep -q "^hsa-rocr$"; then
    ok "hsa-rocr is held"
else
    warn "hsa-rocr not held"
fi
if echo "$HOLDS" | grep -q "^linux-image-6\.17\.0-1017-oem$"; then
    ok "linux-image-6.17.0-1017-oem is held"
else
    if [ "$STRICT" = "1" ]; then
        fail "linux-image-6.17.0-1017-oem NOT held — unattended-upgrade can replace it"
    else
        warn "linux-image-6.17.0-1017-oem NOT held — unattended-upgrade can replace it"
    fi
    echo "        sudo apt-mark hold linux-image-6.17.0-1017-oem linux-modules-6.17.0-1017-oem linux-headers-6.17.0-1017-oem"
fi

# ---------------------------------------------------------------- summary
section "Summary"
echo "  pass: $PASS   fail: $FAIL   warn: $WARN"
if [ "$FAIL" -eq 0 ]; then
    echo
    echo "Environment matches the gfx1151 baseline."
    exit 0
else
    echo
    echo "One or more checks failed. See README-AMD.md for remediation."
    exit 1
fi
