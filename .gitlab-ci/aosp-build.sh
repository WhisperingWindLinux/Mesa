#!/bin/bash

set -e
set -x
set -u

# Early check for required env variables, relies on `set -u`
: "$AOSP_ROOT_PATH"
: "$AOSP_MANIFEST_REPO"
: "$AOSP_MANIFEST_TAG"
: "$AOSP_BUILD_VERSION_TAGS"
: "$AOSP_BUILD_NUMBER"
: "$AOSP_TARGET"

# If the AOSP root dir is already there this may be a local run in
# a container, and the cached dir may have been created by another user, tell
# git to be OK with that and avoid the error: "fatal: detected dubious ownership"
if [ -d  "$AOSP_ROOT_PATH" ];
then
  git config --global --add safe.directory "*"
  cache_space=$(du -BG --summarize "${AOSP_ROOT_PATH}" | sed 's/G.*$//')
else
  mkdir "$AOSP_ROOT_PATH"
  cache_space=0
fi

AOSP_OUTPUT_SIZE_GB=350

# Get the available space in GB for the root directory
available_space=$(df -BG --output=avail "${AOSP_ROOT_PATH}" | tail -n 1 | sed 's/G//')

needed_space=$((AOSP_OUTPUT_SIZE_GB - cache_space))

# Check if the available space is less than the needed space
if [ "$available_space" -lt "$needed_space" ]; then
  echo "ERROR: Less than ${AOSP_OUTPUT_SIZE_GB}GB of space available for the AOSP build."
  exit 1
else
  echo "Sufficient space available: ${needed_space}GB."
fi

# Install some dependencies needed to build AOSP
DEPS=(
  bison
  build-essential
  curl
  flex
  fontconfig
  gnupg
  libncurses5
  libxml2-utils
  meson
  procps
  repo
  rsync
  unzip
  xsltproc
  zip
  zlib1g-dev
)

# Enable the debian contrib repository which has the `repo` tool
sed -e 's/^Components: main$/Components: main contrib/g' -i /etc/apt/sources.list.d/debian.sources
apt-get update

apt-get install -y --no-install-recommends --no-remove "${DEPS[@]}"

cd "$AOSP_ROOT_PATH"
repo init "$AOSP_MANIFEST_REPO" -b "$AOSP_MANIFEST_TAG"
repo sync -c "-j${FDO_CI_CONCURRENT:-4}" --optimize --no-tags

# verify that the intended tree has been checked out
if ! grep -q "BUILD_VERSION_TAGS=$AOSP_BUILD_VERSION_TAGS" build/core/build_id.mk;
then
  echo "Check BUILD_VERSION_TAGS in the AOSP tree" 1>&2
  exit 1
fi
if ! grep -q "BUILD_NUMBER=$AOSP_BUILD_NUMBER" build/core/build_id.mk;
then
  echo "Check BUILD_NUMBER in the AOSP tree" 1>&2
  exit 1
fi

set +x
set +u

source build/envsetup.sh
lunch "$AOSP_TARGET"
make "-j${FDO_CI_CONCURRENT:-4}" > make.log
make "-j${FDO_CI_CONCURRENT:-4}" dist > make_dist.log
