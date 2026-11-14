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

# To update pybind11: bump its version number below and run the script from lib/.
# The script deletes and recreates lib/, include/pybind11, share/ and license/.
# include/tbb-ssort and FindTBB.cmake are maintained by hand and not touched here;
# the pybind11 source archive hash is recorded in patches/README.md.

pybind11_target_version="3.1.0"

#exit on any failure !
set -e

install_prefix=`pwd`

printf "Cleaning previous library versions..."
rm -rf lib
rm -rf include/pybind11
rm -rf share
rm -rf license

mkdir -p license

mkdir -p tmp
cd tmp

printf "\n\nDownloading pybind11..."
curl -s -o pybind11.zip -L https://github.com/pybind/pybind11/archive/v${pybind11_target_version}.zip
printf "\nUncompressing pybind11..."
unzip -q -o pybind11.zip


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


printf "\n\nCleaning temporary..."
cd ..
rm -rf tmp

printf "\nAll done!\n"
