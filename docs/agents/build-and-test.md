# Build, test, and debug

Build behavior is defined by the [root CMake file](../../CMakeLists.txt),
[binding targets](../../higra/CMakeLists.txt), and
[test targets](../../test/CMakeLists.txt).

Commands below use a POSIX shell and an active Python environment. Run root-level
commands from the repository root unless a different working directory is shown.
Use the same interpreter for configuration and manual Python test runs.

## Prerequisites and dependency sources

Provide a C++17 compiler, CMake, Git, and Python with NumPy and development
headers. The full Python suite additionally uses SciPy and scikit-learn:

```bash
python -m pip install numpy scipy scikit-learn
```

This installs test dependencies, not Higra. Installing released Higra is not a
substitute for building and importing the checkout.

| Dependency | Source in this repository |
| --- | --- |
| pybind11 2.13.6 | Vendored headers and CMake config under `lib/` |
| xtensor 0.26.0, xtl 0.8.0 | Vendored headers under `lib/include/` |
| xtensor-python 0.28.0, xsimd 13.2.0 | Vendored headers under `lib/include/` |
| Catch2 3.16.0 | Root `FetchContent`, downloaded at configure time |
| Parallel stable sort | Vendored `lib/include/tbb-ssort/` |
| oneTBB | External CMake package; CI builds version 2023.1.0 |
| Google Benchmark | External package, only with `DO_BENCHMARK=ON` |

The root declares CMake 3.10, but fetched Catch2 declares 3.16, and current
packaging pins CMake 3.30.5. Use a modern CMake; do not interpret the root minimum
as a verified 3.10-compatible build. `HG_UNITY_BUILD` also relies on newer CMake
unity support. README/CI target Python 3.10 through 3.14.

The root always configures Python and fetches Catch2, even with `DO_CPP_TEST=OFF`.
There is no root switch for a standalone C++-only configuration. `find_package`
uses the vendored prefix for pybind11. TBB uses `find_package(TBB CONFIG REQUIRED)`;
the older `lib/FindTBB.cmake` is not the active lookup path.

## Baseline development build

```bash
cmake -S . -B build/agents-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DHG_USE_TBB=OFF
cmake --build build/agents-debug --target higram test_exe --parallel 2
ctest --test-dir build/agents-debug --output-on-failure
```

Choose a fresh build directory when changing interpreter, compiler, or generator.
Existing `cmake-build-*` directories can contain old caches and multiple Python
ABI extension files. They are local artifacts, not authoritative configuration.

The `higram` target produces the extension under `build/agents-debug/higra/`.
Registered Python files are copied into that package during configuration via
`tools/higraTools.cmake::REGISTER_PYTHON_MODULE_FILES`. Tests/resources are copied
into `build/agents-debug/test/`. Re-run configuration after registration changes;
building normally triggers regeneration when configured inputs change.

Building explicit targets avoids unrelated build work and guarantees both the
extension and C++ tests exist. The default build also includes `all_tests`; do
not assume that building only `all_tests` builds `higram`, since the test
directory resets its dependency list.

## CMake controls

| Option | Default | Effect / constraint |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`, or custom `Coverage`; single-config generators |
| `Python_EXECUTABLE` | Discovered | Select interpreter and matching NumPy; inspect configure output |
| `DO_CPP_TEST` | `ON` | Build `test_exe`; does not disable Catch2 fetching or Python tests |
| `DO_AUTO_TEST` | `OFF` | Runs `ctest -V` after building `all_tests` |
| `DO_EMBEDDED_PYTHON_CPP_TEST` | `OFF` | Builds `test_python_exe`, registers `Test_python_cpp` |
| `USE_SIMD` | `ON` | Defines `XTENSOR_USE_XSIMD` |
| `HG_USE_TBB` | `OFF` | Enables TBB on extension/C++ test targets |
| `TBB_DIR` | Discovered | Directory containing `TBBConfig.cmake`, not headers or a runtime-library directory |
| `HG_UNITY_BUILD` | `OFF` | Unity builds for extension and C++ tests; module initializer is excluded |
| `HG_UNITY_BUILD_BATCH_SIZE` | `8` | Integer; CI and TBB packaging use `4` |
| `HG_BUILD_WHEEL` | `OFF` | Packaging-specific behavior, including Windows TBB linkage |
| `DO_BENCHMARK` | `OFF` | Google Benchmark; rejects Debug and additionally requires TBB |

Non-Debug GNU/Clang builds attempt LTO. Avoid high job counts for large template
translation units; setup.py uses two jobs on Unix and CI commonly uses two.
`ccache` is used automatically if found. Multi-config generators additionally
need `--config Debug` or `--config Release` when building and `ctest -C Debug` or
`ctest -C Release` when testing; root build-type/trace logic still reads
`CMAKE_BUILD_TYPE`, so inspect the configured flags rather than assuming parity.

For unity builds, add `-DHG_UNITY_BUILD=ON -DHG_UNITY_BUILD_BATCH_SIZE=4` to the
baseline configure command. Check for collisions among file-local names/macros
if adding binding or test translation units.

For parallel work, configure another build directory with `-DHG_USE_TBB=ON` and
`-DTBB_DIR=/absolute/path/to/lib/cmake/TBB`. Repeat the affected tests in both
modes. CI's `tools/cibuildwheel_*.sh` and `build_tbb_windows.ps1` build oneTBB;
they write/install outside the normal build tree and are not required if a
suitable TBB installation already exists.

## Offline configuration

Catch2 is fetched before the test options are applied. If a Catch2 3.16.0 source
checkout is already available, supply it explicitly to avoid that download:

```bash
cmake -S . -B build/agents-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DHG_USE_TBB=OFF \
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/absolute/path/to/Catch2
```

The final path is a placeholder for source containing Catch2's `CMakeLists.txt`,
not an installed library. A prior build's `_deps/catch2-src` can supply it after
checking its version. Do not assume `DO_CPP_TEST=OFF` makes configuration offline.

## Focused tests

CTest's project entries are `Test_cpp`, `Test_python`, and optional
`Test_python_cpp`. `ctest -R` selects these entries, not individual Catch2 cases:

```bash
ctest --test-dir build/agents-debug -R '^Test_cpp$' --output-on-failure
ctest --test-dir build/agents-debug -R '^Test_python$' --output-on-failure
```

C++ filters operate on test names/tags. Run from the build root to match the
normal test working directory:

```bash
(cd build/agents-debug && ./test/cpp/test_exe --list-tests '[component_tree]')
(cd build/agents-debug && ./test/cpp/test_exe '[component_tree]')
(cd build/agents-debug && ./test/cpp/test_exe '[sorting]')
```

Listing does not run tests. Catch2 name wildcards work at the beginning/end of
the pattern, not in the middle; quote filters so the shell does not expand them.
Inspect the matched-test count and execution summary: a stale executable can
lack newly registered cases, and zero tests is not validation. Older binaries
can also have different listing exit behavior. Available tags live in test
sources. C++ tests use Catch2 v3 and `test/cpp/test_utils.hpp`, not the older Boost
framework described in the contributor page.

Python tests use standard `unittest`, with no configured pytest/tox workflow:

```bash
(cd build/agents-debug && python -m unittest discover -s test/python -v)
(cd build/agents-debug && python -m unittest discover \
  -s test/python -p 'test_component_tree.py' -v)
(cd build/agents-debug && python -m unittest discover \
  -s test/python/test_structure -p 'test_tree.py' -v)
(cd build/agents-debug && python -m unittest discover \
  -s test/python/test_structure -p 'test_tree.py' -k lowest_common_ancestor -v)
```

`-k` filters method names. Verify discovery reports tests, especially after adding
files: an unregistered file is absent from the copied tree. Check dtype, shape,
concept metadata, and exception behavior through public `hg` wrappers rather
than testing only underscored bindings.

Some I/O tests create temporary files in the working directory and remove them.
Use the build directory, not a directory containing unrelated files with those
names. The embedded Python test is optional, has substantial commented-out
coverage, and is not a replacement for the Python suite.

## Import diagnosis and debugging

From the build directory, inspect the actual imports:

```bash
(cd build/agents-debug && python -c \
  'import higra as hg; print(hg.__file__); print(hg.cpp.__file__); print(hg.version())')
```

Running from the source root normally puts the source `higra/` first. A build path
in `PYTHONPATH` alone may not overcome that. CTest inserts the build path at index
zero. If manually testing installed wheels, work outside the repository. Use an
interpreter matching the extension ABI, and check all import paths before
investigating a supposedly unchanged result.

For a C++ failure:

```bash
(cd build/agents-debug && gdb --args ./test/cpp/test_exe '[component_tree]')
```

For a native crash during Python tests:

```bash
(cd build/agents-debug && gdb --args python -X faulthandler -m unittest discover \
  -s test/python -p 'test_component_tree.py' -v)
```

Use the platform debugger equivalent where GDB is unavailable. In a Debug build,
`hg.set_trace(True)` enables compiled function traces; `hg.get_trace()` reports
the runtime switch. Setting it in a Release build does not add compiled traces.
The compile-time log threshold is in `include/higra/detail/log.hpp`.

GCC coverage uses `-DCMAKE_BUILD_TYPE=Coverage`, builds `test_exe`, and runs
`ctest -R '^Test_cpp$'`. Root flags enable gcov instrumentation and exclude Catch2
from it. `tools/azure-pipelines-linux-gcc.yml` shows matching gcov/lcov tooling and
exclusion of vendor/system/`_deps` files. Coverage is not a portable MSVC mode.

## Packaging and platforms

`python -m pip wheel . --no-deps --wheel-dir dist` invokes the setuptools/CMake
backend with build isolation. `pyproject.toml` pins build NumPy by Python version
and CMake 3.30.5; runtime NumPy minimums are separate in `setup.py`. The Python 3.9
runtime entry has no matching current README/CI support or isolated NumPy pin;
do not promise 3.9 support from that entry alone.

setup.py sets `HG_BUILD_WHEEL=ON`, disables C++ tests, and normally builds Release.
`HG_DEBUG` and `HG_USE_TBB` are activated by environment-variable **presence**:
setting either to `0` or `OFF` still enables it. Omit/unset the variable to disable
it. CMake's boolean `-D` options have different semantics. TBB packaging also
enables unity builds with batch size four.

Windows builds need MSVC tooling. TBB wheels additionally require `TBB_DLL` and
`TBB_DIR`; setup.py renames the runtime library to `tbb_higra.dll`, invokes
`dumpbin`/`lib`, and supplies `TBB_RENAMED_LIBRARY` to CMake. Current CI enters the
Visual Studio developer environment first. Do not use the legacy
`tools/build_wheel_msvc.bat`, whose own comments say not to use it, or infer current
support from the old `tools/build_manylinux.sh` script.

The benchmark target is `benchmark_higra`, with a runner target `benchmark_exe`.
Configure Release, `DO_BENCHMARK=ON`, `HG_USE_TBB=ON`, and an installed Google
Benchmark package if working on benchmarks. Most cases in its source list are
commented out. It also contains legacy C++14/optimization flag logic; inspect
actual compile commands before treating it as a representative measurement.

## CI evidence and limits

`.azure-pipelines.yml` includes the active templates under `tools/`:

| Job group | Configured coverage |
| --- | --- |
| Linux core | Ubuntu 24.04, GCC 14, Debug with TBB on/off; GCC Coverage with TBB |
| macOS core | macOS 15, Clang, Debug with TBB on/off |
| Wheels | CPython 3.10–3.14; Linux x86_64 manylinux_2_28, macOS x86_64/ARM64, Windows x86_64 |

Core jobs use unity batch size four. Coverage runs C++ tests only. Wheel jobs use
cibuildwheel 3.3.1, enable TBB, and run the Python suite against installed wheels.
Unix excludes 32-bit manylinux and musllinux; Windows excludes win32. Most wheel
jobs are tag-triggered, with the current Python 3.14 entries forced on ordinary
builds too. Tag jobs can upload to PyPI; they are release operations, not local
validation commands.

These are checked-in configurations, not confirmation of successful live CI.
Report what was actually configured, built, and tested; existing artifacts alone
do not verify a clean build or every supported platform.
