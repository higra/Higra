
import os
import re
import sys
import platform
import subprocess
import os.path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

force_debug = False
force_debug_arg = '--force_debug'
if force_debug_arg in sys.argv:
    index = sys.argv.index(force_debug_arg)
    sys.argv.pop(index)  # Removes the argument
    force_debug = True


def get_version():
    with open(os.path.join("include", "higra", "config.hpp")) as file:
        lines = file.readlines()

    major = 0
    minor = 0
    patch = 0

    for l in lines:
        if l.find("HIGRA_VERSION_MAJOR") >= 0:
            major = int(l.split(' ')[2])
        if l.find("HIGRA_VERSION_MINOR") >= 0:
            minor = int(l.split(' ')[2])
        if l.find("HIGRA_VERSION_PATCH") >= 0:
            patch = int(l.split(' ')[2])

    return ".".join([str(i) for i in (major, minor, patch)])

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        global force_debug
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir = os.path.join(extdir, "higra")
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]#,
                      #'-DDO_CPP_TEST=OFF']

        cfg = 'Debug' if force_debug or self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m', '/v:q']#, '/v:q'
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

try:
    # hack because setuptools wont install files which are not inside a python package
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.symlink(os.path.join(cur_dir, 'include'), 'higra/include')
    os.symlink(os.path.join(cur_dir, 'lib'), 'higra/lib')
    setup(
        name='higra',
        version=get_version(),
        author='Benjamin Perret',
        author_email='benjamin.perret@esiee.fr',
        description='Hierarchical Graph Analysis',
        url='https://github.com/higra/Higra',
        long_description=open('README.md').read(),
        packages=[
            'higra',
            'higra.accumulator',
            'higra.algo',
            'higra.assessment',
            'higra.attribute',
            'higra.hierarchy',
            'higra.image',
            'higra.interop',
            'higra.io_utils',
            'higra.plot',
            'higra.structure'],
        ext_modules=[CMakeExtension('higram')],
        cmdclass=dict(build_ext=CMakeBuild),
        include_package_data=True,
        install_requires=[
            'numpy>=1.15.4',
        ],
        zip_safe=False,
        license='CeCILL-B',
    )
finally:
    os.unlink('higra/include')
    os.unlink('higra/lib')

