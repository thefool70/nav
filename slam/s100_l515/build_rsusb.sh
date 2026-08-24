#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_dir="$script_dir/.rsusb"
build_root="$(mktemp -d /tmp/robot-nav-rsusb.XXXXXX)"
source_dir="$build_root/librealsense"
build_dir="$build_root/build"

cleanup() {
    rm -rf -- "$build_root"
}
trap cleanup EXIT

for command in git cmake c++; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "缺少构建命令：$command" >&2
        exit 1
    fi
done
if ! command -v python >/dev/null 2>&1; then
    echo "请先激活 robot-nav-slam 环境。" >&2
    exit 1
fi

echo "构建 librealsense 2.54.1 RSUSB 后端……"
git clone --quiet --depth 1 --branch v2.54.1 \
    https://github.com/IntelRealSense/librealsense.git \
    "$source_dir"
git -C "$source_dir" apply "$script_dir/librealsense-2.54.1-gcc16.patch"

cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$(command -v python)" \
    -DPYTHON_INSTALL_DIR="$install_dir/python" \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_WITH_CUDA=OFF \
    -DCHECK_FOR_UPDATES=OFF \
    -DIMPORT_DEPTH_CAM_FW=OFF
cmake --build "$build_dir" --parallel "$(nproc)"
cmake --install "$build_dir"

echo "RSUSB 后端已安装到 $install_dir"
