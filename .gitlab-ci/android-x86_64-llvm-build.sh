#!/bin/bash

set -e
set -x
set -u

# Early check for required env variables, relies on `set -u`
: "$ANDROID_NDK_VERSION"
: "$ANDROID_SDK_VERSION"
: "$LLVM_VERSION"
: "$LLVM_ROOT_PATH":
: "$LLVM_ARTIFACT_NAME"

# Install some dependencies needed to build AOSP
DEPS=(
  ninja-build
  unzip
)

apt-get install -y --no-install-recommends --no-remove "${DEPS[@]}"

ANDROID_NDK_ROOT="$(pwd)/${ANDROID_NDK_VERSION}-linux"

if [ ! -d "$LLVM_ROOT_PATH" ];
then
  mkdir "$LLVM_ROOT_PATH"
fi

pushd "$LLVM_ROOT_PATH"

ANDROID_NDK_ROOT="${LLVM_ROOT_PATH}/${ANDROID_NDK_VERSION}"
if [ ! -d "$ANDROID_NDK_ROOT" ];
then
  curl -L --retry 4 -f --retry-all-errors --retry-delay 60 \
    -o "${ANDROID_NDK_VERSION}.zip" \
    "https://dl.google.com/android/repository/${ANDROID_NDK_VERSION}-linux.zip"
  unzip "${ANDROID_NDK_VERSION}.zip" "$ANDROID_NDK_VERSION/source.properties" "$ANDROID_NDK_VERSION/build/cmake/*" "$ANDROID_NDK_VERSION/toolchains/llvm/*"
fi

if [ ! -d "$LLVM_ROOT_PATH/llvm-project" ];
then
  mkdir "$LLVM_ROOT_PATH/llvm-project"
  pushd "$LLVM_ROOT_PATH/llvm-project"
  git init
  git remote add origin https://github.com/llvm/llvm-project.git
  git fetch --depth 1 origin "$LLVM_VERSION"
  git checkout FETCH_HEAD
  popd
fi

pushd "llvm-project"

# Checkout again the intended version, just in case of a pre-existing full clone
git checkout "$LLVM_VERSION" || true

LLVM_INSTALL_PREFIX="${LLVM_ROOT_PATH}/${LLVM_ARTIFACT_NAME}"

rm -rf build/
cmake -GNinja -S llvm -B build/ \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=x86_64 \
    -DANDROID_PLATFORM="android-${ANDROID_SDK_VERSION}" \
    -DANDROID_NDK="${ANDROID_NDK_ROOT}" \
    -DCMAKE_ANDROID_ARCH_ABI=x86_64 \
    -DCMAKE_ANDROID_NDK="${ANDROID_NDK_ROOT}" \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION="${ANDROID_SDK_VERSION}" \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX}" \
    -DCMAKE_CXX_FLAGS="-march=x86-64 --target=x86_64-linux-android${ANDROID_SDK_VERSION} -fno-rtti" \
    -DLLVM_HOST_TRIPLE="x86_64-linux-android${ANDROID_SDK_VERSION}" \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_BUILD_DOCS=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_ENABLE_RTTI=OFF \
    -DLLVM_BUILD_INSTRUMENTED_COVERAGE=OFF \
    -DLLVM_NATIVE_TOOL_DIR="${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin" \
    -DLLVM_ENABLE_PIC=False \
    -DLLVM_OPTIMIZED_TABLEGEN=ON

ninja "-j${FDO_CI_CONCURRENT:-4}" -C build/ install

pushd "$LLVM_ROOT_PATH"
tar --zstd -cf "${LLVM_ARTIFACT_NAME}.tar.zst" "$LLVM_ARTIFACT_NAME"
