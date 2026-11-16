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

# To update pybind11: bump its version number below and run the script (from any directory).
# The script works in its own directory, lib/. It downloads and extracts pybind11 into lib/tmp/
# first, then deletes and recreates lib/lib/, lib/include/pybind11, lib/share/ and lib/license/.
# include/tbb-ssort and FindTBB.cmake are maintained by hand and not touched here;
# the pybind11 source archive hash is recorded in patches/README.md.

pybind11_target_version="3.1.0"

#exit on any failure !
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -d include/tbb-ssort ] || [ ! -f FindTBB.cmake ]; then
    echo "update_lib.sh: $(pwd) is not Higra's lib/ directory, nothing was changed." >&2
    exit 1
fi
install_prefix=$(pwd)

mkdir -p tmp
cd tmp

printf "Downloading pybind11..."
curl -fsSL -o pybind11.zip https://github.com/pybind/pybind11/archive/v${pybind11_target_version}.zip
printf "\nUncompressing pybind11..."
unzip -q -o pybind11.zip
[ -f pybind11-${pybind11_target_version}/CMakeLists.txt ]

printf "\n\nCleaning previous library versions..."
cd "${install_prefix}"
rm -rf lib
rm -rf include/pybind11
rm -rf share
rm -rf license

mkdir -p license

printf "\n\nInstalling pybind11..."
cd tmp/pybind11-${pybind11_target_version}
cp LICENSE "${install_prefix}/license/LICENSE_pybind11"
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="${install_prefix}" -DPYBIND11_TEST=OFF -DPYBIND11_CMAKECONFIG_INSTALL_DIR=lib/cmake/pybind11 .. >/dev/null
make >/dev/null
make install >/dev/null


printf "\n\nCleaning temporary..."
cd "${install_prefix}"
rm -rf tmp

printf "\nAll done!\n"
