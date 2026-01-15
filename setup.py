import os
import re
import sys
import platform
import subprocess
import os.path

from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


def get_option(argname, envname):
    v = False
    if argname in sys.argv:
        index = sys.argv.index(argname)
        sys.argv.pop(index)  # Removes the argument
        v = True
    v = v or (os.getenv(envname) is not None)
    return v


force_debug = get_option("--force_debug", "HG_DEBUG")
use_tbb = get_option("--use_tbb", "HG_USE_TBB")


def get_tbb_dirs():
    link_dir = os.getenv("TBB_LIBRARY")
    include_dir = os.getenv("TBB_INCLUDE_DIR")

    if bool(link_dir) != bool(include_dir):
        print('You must either provide none or BOTH environment variables "TBB_INCLUDE" and "TBB_LINK"')
        exit(1)

    # if env not set, we assume that tbb is installed with python distro...
    if link_dir is None:
        from sysconfig import get_paths
        info = get_paths()
        python_path = info['data']

        include_dir = os.path.join(python_path, "include")
        link_dir = os.path.join(python_path, "lib")

    if not (os.path.isfile(os.path.join(include_dir, "tbb", "tbb.h"))):
        print('Cannot find "tbb.h" please provide tbb location with environment variables "TBB_INCLUDE" and "TBB_LINK"')
        exit(1)

    print("TBB dirs: ", include_dir, link_dir)
    return include_dir, link_dir


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

        print("Setup.py: python executable: ", sys.executable)

        # test if numpy is installed
        try:
            import numpy
            print("Setup.py: numpy version: ", numpy.__version__)
        except ImportError as e:
            print("Setup.py: numpy is not installed. Please install numpy first.", e)
            exit(1)

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPython_EXECUTABLE=' + sys.executable,
                      '-DHG_BUILD_WHEEL=On',
                      '-DDO_CPP_TEST=Off']

        if use_tbb:
            tbb_include, tbb_link = get_tbb_dirs()
            cmake_args = cmake_args + [
                '-DHG_USE_TBB=On',
                '-DTBB_INCLUDE_DIR=' + tbb_include,
                '-DTBB_LIBRARY=' + tbb_link,
                '-DHG_UNITY_BUILD=On',
                '-DHG_UNITY_BUILD_BATCH_SIZE=4']

        cfg = 'Debug' if force_debug or self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m', '/v:q']  # , '/v:q'
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        cxx_flags = env.get('CXXFLAGS', '')
        if platform.system() == "Linux":
            cxx_flags = cxx_flags + " -fabi-version=8 "
        cxx_flags = cxx_flags + ' -DVERSION_INFO=\\"{}\\"'.format(self.distribution.get_version())

        env['CXXFLAGS'] = cxx_flags

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

def rename_dll(inputdll, outputdll):
    import subprocess
    import re
    from shutil import copyfile
    import os.path as path

    if not path.exists(inputdll):
        raise ValueError("File not found " + inputdll)

    # dump the dll exports using dumpbin
    process = subprocess.Popen(['dumpbin', '/EXPORTS', inputdll], stdout=subprocess.PIPE)
    out, err = process.communicate()

    # get all the function definitions
    lines = out.splitlines(keepends=False)
    pattern = r'^\s*(\d+)\s+[A-Z0-9]+\s+[A-Z0-9]{8}\s+([^ ]+)'

    library_output = 'EXPORTS \n'

    for line in lines:
        matches = re.search(pattern, line.decode('ascii'))

        if matches is not None:
            # ordinal = matches.group(1)
            function_name = matches.group(2)
            library_output = library_output + function_name + '\n'

    # write the def file
    deffile_name = outputdll[:-4] + '.def'
    libfile_name = outputdll[:-4] + '.lib'
    with open(deffile_name, 'w') as f:
        f.write(library_output)

    process = subprocess.Popen(['lib', '/MACHINE:X64', '/DEF:' + deffile_name], )
    out, err = process.communicate()

    # copy the dll over
    copyfile(inputdll, outputdll)

# copy tbb dlls under unique name to mimic unix wheel delocate...
def prepare_dll_windows():
    tbb_dll = os.getenv("TBB_DLL")
    if tbb_dll is None:
        print("On windows, you must set the environment variable providing the path to TBB DLL.")
        exit(1)

    from shutil import copyfile
    copyfile(tbb_dll, "higra\\tbb.dll")

    os.chdir("./higra")
    rename_dll("tbb.dll", "tbb_higra.dll")
    if os.path.exists("tbb.lib"):
        os.remove("tbb.lib")
    os.rename("tbb_higra.lib", "tbb.lib")
    os.remove("tbb.dll")
    os.chdir("..")


try:
    python_version = sys.version_info[1] # TODO: python 3 assumed...
    requires_list = {
        9:['numpy>=1.19.5'],
        10:['numpy>=1.21.4'],
        11:['numpy>=1.23.5'],
        12:['numpy>=1.26.0'],
        13:['numpy>=2.1.3'],
        14:['numpy>=2.3.2'],
    }
    if use_tbb and platform.system() == "Windows":
        prepare_dll_windows()

    # hack because setuptools wont install files which are not inside a python package
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.symlink(os.path.join(cur_dir, 'include'), 'higra/include')
    os.symlink(os.path.join(cur_dir, 'lib'), 'higra/lib')
    packages = find_namespace_packages(include=["higra", "higra.*"])
    print("Setup.py: packages: ", packages)
    setup(
        name='higra',
        version=get_version(),
        author='Benjamin Perret',
        author_email='benjamin.perret@esiee.fr',
        description='Hierarchical Graph Analysis',
        url='https://github.com/higra/Higra',
        long_description=open('README.md').read(),
        long_description_content_type="text/markdown",
        packages=packages,
        ext_modules=[CMakeExtension('higram')],
        cmdclass=dict(build_ext=CMakeBuild),
        include_package_data=True,
        install_requires=requires_list[python_version],
        zip_safe=False,
        license='CeCILL-B',
    )
finally:
    os.unlink('higra/include')
    os.unlink('higra/lib')
