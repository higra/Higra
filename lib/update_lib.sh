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
# The script works in its own directory, lib/. It downloads, builds and installs pybind11 into
# lib/tmp/stage/ first. Only when this succeeded, it replaces lib/lib/, lib/include/pybind11,
# lib/share/ and lib/license/ by their staged versions; if the replacement fails, the previous
# versions are restored. FindTBB.cmake and patches/ are maintained by hand and not touched here;
# the pybind11 source archive hash is recorded in patches/README.md.

pybind11_target_version="3.1.0"

#exit on any failure !
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -f FindTBB.cmake ] || [ ! -f patches/README.md ]; then
    echo "update_lib.sh: $(pwd) is not Higra's lib/ directory, nothing was changed." >&2
    exit 1
fi
install_prefix=$(pwd)

# directories of lib/ that are produced by this script
replaced_dirs=(lib include/pybind11 share license)

if [ -d tmp/old ]; then
    echo "update_lib.sh: $(pwd)/tmp/old exists and may hold the only copy of a previous installation;" \
         "restore or delete it first, nothing was changed." >&2
    exit 1
fi
rm -rf tmp
mkdir -p tmp
cd tmp

printf "Downloading pybind11..."
curl -fsSL -o pybind11.zip https://github.com/pybind/pybind11/archive/v${pybind11_target_version}.zip
printf "\nUncompressing pybind11..."
unzip -q -o pybind11.zip
source_dir="${install_prefix}/tmp/pybind11-${pybind11_target_version}"
[ -f "${source_dir}/CMakeLists.txt" ]

printf "\n\nBuilding pybind11 into tmp/stage..."
stage="${install_prefix}/tmp/stage"
cmake -S "${source_dir}" -B "${install_prefix}/tmp/build" -DCMAKE_INSTALL_PREFIX="${stage}" \
      -DPYBIND11_TEST=OFF -DPYBIND11_CMAKECONFIG_INSTALL_DIR=lib/cmake/pybind11 >/dev/null
cmake --build "${install_prefix}/tmp/build" >/dev/null
cmake --install "${install_prefix}/tmp/build" >/dev/null
mkdir -p "${stage}/license"
cp "${source_dir}/LICENSE" "${stage}/license/LICENSE_pybind11"

for expected in include/pybind11/pybind11.h lib/cmake/pybind11/pybind11Config.cmake license/LICENSE_pybind11; do
    if [ ! -f "${stage}/${expected}" ]; then
        echo "update_lib.sh: the pybind11 installation lacks ${expected}, nothing was changed." >&2
        exit 1
    fi
done
for dir in "${replaced_dirs[@]}"; do
    if [ ! -d "${stage}/${dir}" ]; then
        echo "update_lib.sh: the pybind11 installation lacks ${dir}/, nothing was changed." >&2
        exit 1
    fi
done

printf "\n\nReplacing the previous installation..."
cd "${install_prefix}"
mkdir -p tmp/old/include
moved_dirs=()
installed_dirs=()

restore_previous() {
    echo "update_lib.sh: replacement failed, restoring the previous installation." >&2
    for dir in ${installed_dirs[@]+"${installed_dirs[@]}"}; do
        rm -rf "${install_prefix:?}/${dir}"
    done
    for dir in ${moved_dirs[@]+"${moved_dirs[@]}"}; do
        mv "${install_prefix}/tmp/old/${dir}" "${install_prefix}/${dir}"
    done
    rm -rf "${install_prefix}/tmp/old"
}
trap restore_previous ERR

for dir in "${replaced_dirs[@]}"; do
    if [ -e "${dir}" ]; then
        mv "${dir}" "tmp/old/${dir}"
        moved_dirs+=("${dir}")
    fi
done
for dir in "${replaced_dirs[@]}"; do
    mv "${stage}/${dir}" "${dir}"
    installed_dirs+=("${dir}")
done

trap - ERR

printf "\n\nCleaning temporary..."
rm -rf tmp

printf "\nAll done!\n"
