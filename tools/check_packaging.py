"""Check R03 staging and sdist contracts in an explicitly isolated source tree.

Usage: python tools/check_packaging.py /tmp/extracted/higra-VERSION [wheel.whl]
This runs metadata, repeated build_py staging, and legacy sdist generation; it
never compiles the extension. Wheel compilation/installation are separate checks.
"""

import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from unittest import mock


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def check_windows_staging(source, output):
    """Exercise Windows output paths without requiring MSVC on the test host."""
    from setuptools import Distribution
    import numpy  # Load before simulating platform.system().

    old_cwd = Path.cwd()
    try:
        os.chdir(source)
        with mock.patch('setuptools.setup'), mock.patch.object(sys, 'argv', ['setup.py']), \
                mock.patch.dict(os.environ, {'HG_USE_TBB': '1'}), \
                mock.patch('platform.system', return_value='Windows'):
            setup = runpy.run_path(str(source / 'setup.py'))
        extension = setup['CMakeExtension']('higram')
        builder = setup['CMakeBuild'](Distribution({'ext_modules': [extension]}))
        builder.build_temp = str(output / 'temp')
        builder.debug = False
        builder.get_ext_fullpath = lambda name: str(output / 'wheel' / (name + '.pyd'))
        original_dll = output / 'original.dll'
        original_dll.write_bytes(b'test runtime')

        def rename(input_dll, output_dll):
            Path(output_dll).write_bytes(Path(input_dll).read_bytes())
            Path(output_dll).with_suffix('.lib').write_bytes(b'test import library')

        namespace = builder.build_extension.__func__.__globals__
        with mock.patch.dict(namespace, {'use_tbb': True, 'get_tbb_dir': lambda: str(output / 'TBB'),
                                         'rename_dll': rename}), \
                mock.patch.dict(os.environ, {'TBB_DLL': str(original_dll)}), \
                mock.patch('platform.system', return_value='Windows'), \
                mock.patch('subprocess.check_call') as calls:
            builder.build_extension(extension)
        runtime = output / 'wheel' / 'higra' / 'tbb_higra.dll'
        library = output / 'temp' / 'tbb' / 'tbb_higra.lib'
        assert runtime.read_bytes() == original_dll.read_bytes()
        assert library.exists()
        assert '-DTBB_RENAMED_LIBRARY=' + str(library) in calls.call_args_list[0].args[0]
        print('PASS: simulated Windows DLL staging and CMake import-library path')
    finally:
        os.chdir(old_cwd)


def check(source, wheel=None):
    source = source.resolve()
    repository = Path(__file__).resolve().parents[1]
    if source == repository or repository in source.parents:
        raise ValueError('Packaging checks require a source tree outside the checkout')

    def run(*args):
        subprocess.run([sys.executable, 'setup.py', *args], cwd=source, check=True)

    # A pre-existing source resource must survive and must not leak into output.
    sentinels = [source / 'higra' / 'include' / 'r03-sentinel.hpp',
                 source / 'higra' / 'lib' / 'r03-sentinel.cmake']
    if any(p.exists() for p in sentinels):
        raise ValueError('Sentinel paths already exist')
    original = snapshot(source / 'higra')
    try:
        for sentinel in sentinels:
            sentinel.parent.mkdir(exist_ok=True)
            sentinel.write_text('pre-existing source content\n')
        before = snapshot(source / 'higra')
        with tempfile.TemporaryDirectory(prefix='higra-packaging-check-') as tmp:
            output = Path(tmp)
            run('--name')
            run('--version')
            for _ in range(2):
                run('build_py', '--build-lib', str(output / 'build'))
                assert snapshot(source / 'higra') == before
            resources = {}
            for directory in ('include', 'lib'):
                for relative, digest in snapshot(source / directory).items():
                    path = Path(directory) / relative
                    if path.suffix in ('.hpp', '.h', '.cmake') or path.parts[:2] == ('lib', 'license'):
                        resources['higra/' + path.as_posix()] = digest
            built = snapshot(output / 'build')
            assert all(built.get(name) == digest for name, digest in resources.items())
            assert not any('r03-sentinel' in name for name in built)
            run('sdist', '--dist-dir', str(output / 'dist'))
            archive = next((output / 'dist').glob('*.tar.gz'))
            with tarfile.open(archive) as tar:
                entries = {name.split('/', 1)[1] for name in tar.getnames() if '/' in name}
                expected = ['CMakeLists.txt', 'tools/higraTools.cmake', 'include/higra/config.hpp',
                            'higra/pymodule.cpp', 'test/CMakeLists.txt',
                            'test/python/resources/test.graph', 'lib/lib/cmake/pybind11/pybind11Config.cmake']
                assert all(name in entries for name in expected)
                assert not any(name.startswith(('higra/include/', 'higra/lib/')) for name in entries)
                tar.extractall(output / 'extracted', filter='data')
            extracted = next((output / 'extracted').iterdir())
            subprocess.run([sys.executable, 'setup.py', '--name'], cwd=extracted, check=True)
            assert not (extracted / 'higra' / 'include').exists()
            assert not (extracted / 'higra' / 'lib').exists()
            if wheel:
                with zipfile.ZipFile(wheel) as package:
                    entries = set(package.namelist())
                    assert all(name in entries and hashlib.sha256(package.read(name)).hexdigest() == digest
                               for name, digest in resources.items())
                    assert not any(name.endswith(('.cpp', '.cmake', '.hpp', '.h'))
                                   for name in entries if not name.startswith(('higra/include/', 'higra/lib/')))
            check_windows_staging(source, output)
            assert snapshot(source / 'higra') == before
            print('PASS: metadata, repeated staging, extracted sdist, resource hashes, source preservation')
    finally:
        for sentinel in sentinels:
            sentinel.unlink(missing_ok=True)
            if not any(sentinel.parent.iterdir()):
                sentinel.parent.rmdir()
    assert snapshot(source / 'higra') == original


if __name__ == '__main__':
    check(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
