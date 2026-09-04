#!/usr/bin/env bash
# Habitat 通用命令包装器：在已激活的 robot-nav-habitat 环境中运行任意命令。
# HABITAT_RENDERER=auto|gpu|cpu 选择渲染后端；auto 会先验证 GPU EGL。
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "robot-nav-habitat" ]]; then
  echo "错误：请先激活 robot-nav-habitat 环境（CONDA_DEFAULT_ENV 必须为 robot-nav-habitat）" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_EGL_VENDOR="$CONDA_PREFIX/share/glvnd/egl_vendor.d/50_mesa.json"
WSL_EGL_VENDOR="$SCRIPT_DIR/egl-wsl-gpu.json"

RENDERER="${HABITAT_RENDERER:-auto}"

# Habitat 的 conda Python 带 RPATH。GPU 模式必须预加载完整的系统
# Mesa/GLVND/DRM 链，避免 conda 内较旧的 libdrm 覆盖 Arch Mesa 的依赖。
GPU_PRELOAD="/usr/lib/libdrm.so.2:/usr/lib/libdrm_amdgpu.so.1:/usr/lib/libdrm_intel.so.1:/usr/lib/libLLVM.so:/usr/lib/libEGL.so.1:/usr/lib/libOpenGL.so.0:/usr/lib/libGLdispatch.so.0"
GPU_DEPS=(
  "/dev/dxg"
  "/usr/bin/eglinfo"
  "/usr/lib/libdrm.so.2"
  "/usr/lib/libdrm_amdgpu.so.1"
  "/usr/lib/libdrm_intel.so.1"
  "/usr/lib/libEGL.so.1"
  "/usr/lib/libOpenGL.so.0"
  "/usr/lib/libGLdispatch.so.0"
  "/usr/lib/libEGL_mesa.so.0"
  "/usr/lib/dri/d3d12_dri.so"
  "/usr/lib/libLLVM.so"
  "/usr/lib/wsl/lib/libd3d12.so"
  "/usr/lib/wsl/lib/libdxcore.so"
  "$WSL_EGL_VENDOR"
)

gpu_ready() {
  local dep
  for dep in "${GPU_DEPS[@]}"; do
    [[ -e "$dep" ]] || return 1
  done
}

gpu_egl_ready() {
  if ! env -u LIBGL_ALWAYS_SOFTWARE \
    EGL_PLATFORM=surfaceless \
    __EGL_VENDOR_LIBRARY_FILENAMES="$WSL_EGL_VENDOR" \
    GALLIUM_DRIVER=d3d12 \
    LIBGL_DRIVERS_PATH=/usr/lib/dri \
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
    LD_PRELOAD="$GPU_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}" \
    /usr/bin/eglinfo -B -p surfaceless >/dev/null 2>&1; then
    return 1
  fi

  # eglinfo uses an explicit platform API. Habitat-Sim 0.3.3 instead calls
  # eglGetDisplay(EGL_DEFAULT_DISPLAY), so validate that exact call from the
  # conda Python process as well.
  env -u LIBGL_ALWAYS_SOFTWARE \
    EGL_PLATFORM=surfaceless \
    __EGL_VENDOR_LIBRARY_FILENAMES="$WSL_EGL_VENDOR" \
    GALLIUM_DRIVER=d3d12 \
    LIBGL_DRIVERS_PATH=/usr/lib/dri \
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
    LD_PRELOAD="$GPU_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$CONDA_PREFIX/bin/python" -c '
import ctypes

egl = ctypes.CDLL("libEGL.so.1")
egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
egl.eglGetDisplay.restype = ctypes.c_void_p
egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
egl.eglInitialize.restype = ctypes.c_uint
display = egl.eglGetDisplay(None)
raise SystemExit(0 if display and egl.eglInitialize(display, None, None) else 1)
' \
    >/dev/null 2>&1
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
    if ! gpu_egl_ready; then
      echo "错误：WSL GPU 文件存在，但无法创建 Mesa D3D12 EGL 上下文。" >&2
      echo "请先使用 HABITAT_RENDERER=cpu；GPU 模式需检查 WSLg、Mesa 与显卡驱动。" >&2
      exit 1
    fi
    ;;
  cpu)
    ;;
  auto)
    if gpu_ready && gpu_egl_ready; then
      RENDERER="gpu"
    else
      echo "提示：WSL GPU EGL 不可用，自动使用 conda mesa-llvmpipe（CPU）渲染。" >&2
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
  unset LIBGL_ALWAYS_SOFTWARE
  export LD_PRELOAD="$GPU_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "Habitat 渲染后端：WSL Mesa D3D12（GPU）。" >&2
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
  export LIBGL_ALWAYS_SOFTWARE=1
  export LD_PRELOAD="$LLVM_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "Habitat 渲染后端：Mesa llvmpipe（CPU）。" >&2
fi

exec "$@"
