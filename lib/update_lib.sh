#!/usr/bin/env bash
############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

# To update various libs: bump their version number below and run the script

xtl_target_version="0.7.7"
xtensor_target_version="0.25.0"
xsimd_target_version="13.0.0"
xtensor_python_target_version="0.27.0"
pybind11_target_version="2.13.1"
catch_target_version="2.13.9"

#exit on any failure !
set -e

install_prefix=`pwd`

printf "Cleaning previous library versions..."
rm -rf lib
rm -rf include/catch2
rm -rf include/pybind11
rm -rf include/xsimd
rm -rf include/xtensor
rm -rf include/xtensor-python
rm -rf include/xtl
rm -rf share
rm -rf license

mkdir -p license

mkdir -p tmp
cd tmp

printf "\n\nDownloading xtl..."
curl -s -o xtl.zip -L https://github.com/QuantStack/xtl/archive/${xtl_target_version}.zip
printf "\nUncompressing xtl..."
unzip -q -o xtl.zip

printf "\n\nDownloading xtensor..."
curl -s -o xtensor.zip -L https://github.com/QuantStack/xtensor/archive/${xtensor_target_version}.zip
printf "\nUncompressing xtensor..."
unzip -q -o xtensor.zip

printf "\n\nDownloading xsimd..."
curl -s -o xsimd.zip -L https://github.com/QuantStack/xsimd/archive/${xsimd_target_version}.zip
printf "\nUncompressing xsimd..."
unzip -q -o xsimd.zip

printf "\n\nDownloading xtensor-python..."
curl -s -o xtensor_python.zip -L https://github.com/QuantStack/xtensor-python/archive/${xtensor_python_target_version}.zip
printf "\nUncompressing xtensor-python..."
unzip -q -o xtensor_python.zip

printf "\n\nDownloading pybind11..."
curl -s -o pybind11.zip -L https://github.com/pybind/pybind11/archive/v${pybind11_target_version}.zip
printf "\nUncompressing pybind11..."
unzip -q -o pybind11.zip

printf "\n\nDownloading catch2..."
curl -s -o catch2.zip -L https://github.com/catchorg/Catch2/archive/v${catch_target_version}.zip
printf "\nUncompressing catch2..."
unzip -q -o catch2.zip


printf "\n\nInstalling pybind11..."
cd pybind11-${pybind11_target_version}
cp LICENSE ../../license/LICENSE_pybind11
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix} -DPYBIND11_TEST=OFF -DPYBIND11_CMAKECONFIG_INSTALL_DIR=lib/cmake/pybind11 .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..


printf "\n\nInstalling xtl..."
cd xtl-${xtl_target_version}
cp LICENSE ../../license/LICENSE_xtl
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix} .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..


printf "\n\nInstalling xsimd..."
cd xsimd-${xsimd_target_version}
cp LICENSE ../../license/LICENSE_xsimd
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix}  .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..


printf "\n\nInstalling xtensor..."
#git clone https://github.com/xtensor-stack/xtensor.git
#cd xtensor
cd xtensor-${xtensor_target_version}
cp LICENSE ../../license/LICENSE_xtensor
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix} .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..

printf "\n\nInstalling xtensor-python..."
cd xtensor-python-${xtensor_python_target_version}
cp LICENSE ../../license/LICENSE_xtensor-python
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix}  .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..

printf "\n\nInstalling catch2..."
cd Catch2-${catch_target_version}
cp LICENSE.txt ../../license/LICENSE_catch2
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix} -DBUILD_TESTING=OFF -DCATCH_INSTALL_DOCS=OFF -DCATCH_INSTALL_HELPERS=OFF .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..

printf "\n\nCleaning temporary..."
cd ..
rm -rf tmp

printf "\nAll done!\n"
