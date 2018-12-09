#!/usr/bin/env bash

set -e -x


export TOOLCHAIN_URL='https://github.com/Noctem/pogeo-toolchain/releases/download/v1.4/gcc-7.2-binutils-2.29-centos5-x86-64.tar.bz2'

export MFLAG="-m64"

curl -L "$TOOLCHAIN_URL" -o toolchain.tar.bz2
tar -C / -xf toolchain.tar.bz2

MANYLINUX=1
export PATH="/toolchain/bin:${PATH}"
export CFLAGS="-I/toolchain/include ${MFLAG}"
export CXXFLAGS="-I/toolchain/include ${MFLAG} -static-libstdc++"
export LD_LIBRARY_PATH="/toolchain/lib64:/toolchain/lib:${LD_LIBRARY_PATH}"

pip install numpy cmake
