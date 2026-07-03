#!/usr/bin/env bash
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

# --- Check prerequisites ---
command -v nvcc >/dev/null 2>&1 || {
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH=/usr/local/cuda/bin:$PATH
  else
    die "nvcc not found. Install the CUDA toolkit first."
  fi
}

command -v git >/dev/null 2>&1 || die "git not found. Install with: sudo apt install git"
nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed. Check your NVIDIA driver installation."

# --- Detect GPU compute capability ---
echo "Detecting GPU architecture..."
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [[ -z "$ARCH" ]]; then
  die "Could not detect GPU compute capability"
fi
echo "  Compute capability: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1) (sm_${ARCH})"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  GPUs detected: $GPU_COUNT"

if [[ "$GPU_COUNT" -lt 2 ]]; then
  echo "warning: only $GPU_COUNT GPU detected. P2P test needs at least 2."
fi

# --- Clone cuda-samples if needed ---
WORKDIR="${TMPDIR:-/tmp}/cuda-p2p-test"
SAMPLE_DIR="$WORKDIR/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest"

if [[ ! -d "$WORKDIR/cuda-samples" ]]; then
  echo ""
  echo "Cloning cuda-samples..."
  mkdir -p "$WORKDIR"
  git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git "$WORKDIR/cuda-samples"
else
  echo "cuda-samples already cloned at $WORKDIR/cuda-samples"
fi

# --- Build ---
echo ""
echo "Building p2pBandwidthLatencyTest (sm_${ARCH})..."
cd "$SAMPLE_DIR"
nvcc -arch=sm_${ARCH} -o p2pBandwidthLatencyTest p2pBandwidthLatencyTest.cu -I../../../Common
echo "Build OK."

# --- Run ---
echo ""
echo "=== P2P Bandwidth & Latency Test ==="
echo ""
./p2pBandwidthLatencyTest

# --- Show topology for context ---
echo ""
echo "=== GPU Topology ==="
nvidia-smi topo -m

# --- Show link status ---
echo ""
echo "=== PCIe Link Status ==="
for gpu in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
  bdf=$(echo "$gpu" | sed 's/00000000://' | tr '[:upper:]' '[:lower:]')
  echo -n "  $gpu: "
  lspci -vvv -s "$bdf" 2>/dev/null | grep "LnkSta:" | head -1 | sed 's/.*LnkSta://' | xargs
