#!/usr/bin/env bash
# Habitat 通用命令包装器：在已激活的 robot-nav-habitat 环境中运行任意命令。
# HABITAT_RENDERER=auto|gpu|cpu 选择渲染后端，默认 auto。
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "robot-nav-habitat" ]]; then
  echo "错误：请先激活 robot-nav-habitat 环境（CONDA_DEFAULT_ENV 必须为 robot-nav-habitat）" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_EGL_VENDOR="$CONDA_PREFIX/share/glvnd/egl_vendor.d/50_mesa.json"
WSL_EGL_VENDOR="$SCRIPT_DIR/egl-wsl-gpu.json"

RENDERER="${HABITAT_RENDERER:-auto}"

# GPU 渲染前提：WSL 直通设备 /dev/dxg 与系统 Arch Mesa（GLVND EGL、d3d12 驱动、LLVM）。
GPU_DEPS=("/dev/dxg" "/usr/lib/libEGL_mesa.so.0" "/usr/lib/dri/d3d12_dri.so" "/usr/lib/libLLVM.so")

gpu_ready() {
  local dep
  for dep in "${GPU_DEPS[@]}"; do
    [[ -e "$dep" ]] || return 1
  done
}

case "$RENDERER" in
  gpu)
    if ! gpu_ready; then
      echo "错误：HABITAT_RENDERER=gpu 但缺少 GPU 渲染前提：" >&2
      for dep in "${GPU_DEPS[@]}"; do
        [[ -e "$dep" ]] || echo "  $dep" >&2
      done
      echo "请安装系统 Mesa：sudo pacman -S --needed mesa libglvnd mesa-utils" >&2
      exit 1
    fi
    ;;
  cpu)
    ;;
  auto)
    if gpu_ready; then
      RENDERER="gpu"
    else
      echo "提示：未检测到 WSL GPU 渲染条件，回退到 conda mesa-llvmpipe（CPU）渲染" >&2
      RENDERER="cpu"
    fi
    ;;
  *)
    echo "错误：HABITAT_RENDERER 必须是 auto|gpu|cpu，当前为 $RENDERER" >&2
    exit 1
    ;;
esac

if [[ $# -eq 0 ]]; then
  echo "用法：$0 <命令> [参数...]" >&2
  exit 1
fi

export EGL_PLATFORM=surfaceless

if [[ "$RENDERER" == "gpu" ]]; then
  LLVM_LIB="/usr/lib/libLLVM.so"
  if [[ ! -f "$LLVM_LIB" ]]; then
    echo "错误：缺少 $LLVM_LIB，请安装系统 Mesa：sudo pacman -S --needed mesa libglvnd mesa-utils" >&2
    exit 1
  fi
  if [[ ! -f "$WSL_EGL_VENDOR" ]]; then
    echo "错误：缺少 $WSL_EGL_VENDOR" >&2
    exit 1
  fi
  export __EGL_VENDOR_LIBRARY_FILENAMES="$WSL_EGL_VENDOR"
  export GALLIUM_DRIVER=d3d12
  export LIBGL_DRIVERS_PATH=/usr/lib/dri
  export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
else
  LLVM_LIB="$CONDA_PREFIX/lib/libLLVM.so.22.1"
  if [[ ! -f "$LLVM_LIB" ]]; then
    echo "错误：缺少 $LLVM_LIB，请确认已激活 robot-nav-habitat 环境" >&2
    exit 1
  fi
  if [[ ! -f "$CONDA_EGL_VENDOR" ]]; then
    echo "错误：缺少 $CONDA_EGL_VENDOR" >&2
    exit 1
  fi
  export __EGL_VENDOR_LIBRARY_FILENAMES="$CONDA_EGL_VENDOR"
  export GALLIUM_DRIVER=llvmpipe
  export LIBGL_DRIVERS_PATH="$CONDA_PREFIX/lib/dri"
fi

export LD_PRELOAD="$LLVM_LIB${LD_PRELOAD:+:$LD_PRELOAD}"

exec "$@"
