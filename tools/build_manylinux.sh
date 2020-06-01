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

PATH_SAV=${PATH}

# Compile wheels
for PYBIN in /opt/python/cp3[3456789]*/bin; do #[3456789]
	cd ${PYBIN}
	PIP=${PYBIN}/pip
	"$PIP" install -U numpy cmake scipy scikit-learn
	export PATH=${PYBIN}:${PATH_SAV}
	echo ${PATH}
	#ln -s /usr/bin/cmake ${PYBIN}/cmake
	"$PIP" wheel -v /io/ -w /tmp/wheelhouse/
done

# Repair for manylinux compatibility
for WHL in /tmp/wheelhouse/*.whl; do
	auditwheel repair "$WHL" -w /io/wheelhouse/ || auditwheel -v show "$WHL"
done

# Install packages and test
cd /Higra/test/python
for PYBIN in /opt/python/cp3[3456789]*/bin; do
	"${PYBIN}/pip" install higra --no-index -f /io/wheelhouse
	"${PYBIN}/python" -c "import unittest;result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('/Higra/test/python/'));exit(0 if result.wasSuccessful() else 1)"
done
