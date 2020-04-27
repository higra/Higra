#!/usr/bin/env bash

set -e -x

# compile TBB
export TBB_URL="https://github.com/intel/tbb/archive/2019_U9.zip"
pwd
curl -L "${TBB_URL}" -o archive.tgz
unzip archive.tgz
mv oneTBB* tbb
cd tbb
tbb_dir=`pwd`

make -j tbb

cd build
cd *release
tbb_link=`pwd`

mkdir -p /Users/travis//tbb/lib/
mv ${tbb_link}/* /Users/travis//tbb/lib/
mv ${tbb_dir}/include /Users/travis/tbb/

cd ${cur_dir}

pip install numpy==1.17.3

