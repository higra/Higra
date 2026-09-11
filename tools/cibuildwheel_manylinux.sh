#!/usr/bin/env bash

set -e -x

# compile TBB
cur_dir=`pwd`

export MFLAG="-m64"
export CXXFLAGS=" ${MFLAG} -fabi-version=8 -Wno-error=changes-meaning"
export TBB_VERSION="2023.1.0"
export TBB_URL="https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${TBB_VERSION}.tar.gz"

curl -L "${TBB_URL}" -o archive.tgz
tar -xzf archive.tgz
mv oneTBB-* tbb

cmake -S tbb -B tbb/build \
  -DTBB_TEST=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_INSTALL_PREFIX=/tbb

cmake --build tbb/build --parallel
cmake --install tbb/build





