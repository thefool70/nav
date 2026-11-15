#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_dir="$script_dir/.rsusb"
sdk_version="${ROBOT_NAV_REALSENSE_VERSION:-2.56.5}"
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
    echo "请先激活要运行 RealSense 的 micromamba 环境。" >&2
    exit 1
fi

echo "构建 librealsense $sdk_version RSUSB 后端……"
git clone --quiet --depth 1 --branch "v$sdk_version" \
    https://github.com/realsenseai/librealsense.git \
    "$source_dir"
if [[ "$sdk_version" == "2.54.1" ]]; then
    git -C "$source_dir" apply "$script_dir/librealsense-2.54.1-gcc16.patch"
fi

cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_UNIT_TESTS=OFF \
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
