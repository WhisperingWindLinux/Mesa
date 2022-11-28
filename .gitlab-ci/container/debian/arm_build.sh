#!/bin/bash
# shellcheck disable=SC2086 # we want word splitting

set -e
set -o xtrace

apt-get -y install ca-certificates
sed -i -e 's/http:\/\/deb/https:\/\/deb/g' /etc/apt/sources.list
echo 'deb https://deb.debian.org/debian buster main' >/etc/apt/sources.list.d/buster.list
apt-get update

# Ephemeral packages (installed for this script and removed again at
# the end)
STABLE_EPHEMERAL=" \
        libssl-dev \
        "

apt-get -y install \
	${EXTRA_LOCAL_PACKAGES} \
	${STABLE_EPHEMERAL} \
	autoconf \
	automake \
	bc \
	bison \
	ccache \
	clang \
	clang-13 \
	cmake \
	curl \
	debootstrap \
	fastboot \
	flex \
	g++ \
	git \
	glslang-tools \
	kmod \
	libasan6 \
	libclang-cpp13-dev \
	libclang-13-dev \
	libdrm-dev \
	libelf-dev \
	libexpat1-dev \
	libvulkan-dev \
	libx11-dev \
	libx11-xcb-dev \
	libxcb-dri2-0-dev \
	libxcb-dri3-dev \
	libxcb-glx0-dev \
	libxcb-present-dev \
	libxcb-randr0-dev \
	libxcb-shm0-dev \
	libxcb-xfixes0-dev \
	libxdamage-dev \
	libxext-dev \
	libxrandr-dev \
	libxshmfence-dev \
	libxxf86vm-dev \
	libwayland-dev \
	ninja-build \
	llvm-13-dev \
	meson \
	pkg-config \
	python3-mako \
	python3-pil \
	python3-pip \
	python3-requests \
	python3-setuptools \
	u-boot-tools \
	xz-utils \
	zlib1g-dev \
	zstd

# Not available anymore in bullseye
apt-get install -y --no-remove -t buster \
        android-sdk-ext4-utils

pip3 install git+http://gitlab.freedesktop.org/freedesktop/ci-templates@ffe4d1b10aab7534489f0c4bbc4c5899df17d3f2

# We need at least 1.0.0 for rusticl
pip3 install meson==1.0.0

arch=armhf
. .gitlab-ci/container/cross_build.sh

. .gitlab-ci/container/container_pre_build.sh

. .gitlab-ci/container/build-mold.sh

# dependencies where we want a specific version
EXTRA_MESON_ARGS=
. .gitlab-ci/container/build-libdrm.sh

. .gitlab-ci/container/build-wayland.sh

. .gitlab-ci/container/build-llvm-spirv.sh

. .gitlab-ci/container/build-libclc.sh

# We need at least 0.61.4 for proper Rust
pip3 install meson==0.61.5

. .gitlab-ci/container/build-rust.sh

# install bindgen
RUSTFLAGS='-L native=/usr/local/lib' cargo install \
  bindgen --version 0.59.2 \
  -j ${FDO_CI_CONCURRENT:-4} \
  --root /usr/local

apt-get purge -y $STABLE_EPHEMERAL

. .gitlab-ci/container/container_post_build.sh
