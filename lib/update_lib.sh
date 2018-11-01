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

xtl_target_version="0.4.16"
xtensor_target_version="0.18.2"
xsimd_target_version="7.1.0"
xtensor_python_target_version="0.21.0"
pybind11_target_version="2.2.4"

#exit on any failure !
set -e

install_prefix=`pwd`

printf "Cleaning previous library versions..."
rm -rf lib
rm -rf include

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


printf "\n\nInstalling pybind11..."
cd pybind11-${pybind11_target_version}
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix} -DPYBIND11_TEST=OFF -DPYBIND11_CMAKECONFIG_INSTALL_DIR=lib/cmake/pybind11 .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..


printf "\n\nInstalling xtl..."
cd xtl-${xtl_target_version}
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix} .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..


printf "\n\nInstalling xsimd..."
cd xsimd-${xsimd_target_version}
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix}  .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..


printf "\n\nInstalling xtensor..."
cd xtensor-${xtensor_target_version}
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix} -DXTENSOR_USE_XSIMD=ON .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..

printf "\n\nInstalling xtensor-python..."
cd xtensor-python-${xtensor_python_target_version}
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_prefix}  .. >/dev/null
make >/dev/null
make install >/dev/null
cd ../..

printf "\n\nCleaning temporary..."
cd ..
rm -rf tmp

printf "\nAll done!\n"
