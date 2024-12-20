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

cd
home=`pwd`

mkdir -p ${home}/tbb/lib/
mv ${tbb_link}/* ${home}//tbb/lib/
mv ${tbb_dir}/include ${home}/tbb/

cd ${cur_dir}



