#!/usr/bin/env bash

set -e -x

# compile TBB
cur_dir=`pwd`

export MFLAG="-m64"
export CXXFLAGS=" ${MFLAG} -fabi-version=8 "
export TBB_URL="https://github.com/intel/tbb/archive/2019_U9.zip"

curl -L "${TBB_URL}" -o archive.tgz
unzip archive.tgz
mv oneTBB* tbb
cd tbb
tbb_dir=`pwd`

make -j tbb

cd build
cd *release
tbb_link=`pwd`

mkdir -p /tbb/lib/
mv ${tbb_link}/* /tbb/lib/
mv ${tbb_dir}/include /tbb/
ls /tbb/lib/*
ls /tbb/include/*
cd ${cur_dir}

pip install numpy==1.17.3 cmake==3.14.4




