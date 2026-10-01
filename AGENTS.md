# Working on Higra

Higra is a C++17/Python library for hierarchical graph analysis and image
processing. Its C++ algorithms are largely header-based templates; Python calls
them through a single pybind11 extension, `higra.higram`, and adds shape handling,
metadata, caching, and composed algorithms.

## Find the right layer

| Task | Source | Tests |
| --- | --- | --- |
| Core data structure or algorithm | `include/higra/<module>/*.hpp` | `test/cpp/<module>/test_*.cpp` |
| Bind or change a C++ API | `higra/<module>/py_*.cpp`, matching `.hpp`, `all.hpp`, `higra/pymodule.cpp` | `test/python/test_<module>/test_*.py` |
| Python wrapper or composed algorithm | `higra/<module>/*.py` and package `__init__.py` | `test/python/test_<module>/test_*.py` |
| Shapes, concepts, caching, dtype conversion | `higra/hg_utils.py`, `concept.py`, `data_cache.py` | Top-level `test/python/test_*.py` |
| Mutable component trees and CASF | `include/higra/detail/hierarchy/`, `include/higra/hierarchy/component_tree_casf.hpp` | `test/cpp/hierarchy/`, `test/python/test_hierarchy/` |
| Build, packaging, CI | `CMakeLists.txt`, module CMake files, `setup.py`, `pyproject.toml`, `tools/`, `.azure-pipelines.yml` | See [build guide](docs/agents/build-and-test.md) |
| Published documentation | `doc/source/`, public docstrings, `doc/Doxyfile` | See [maintenance guide](docs/agents/documentation-and-maintenance.md) |

The Python I/O module is `io_utils`; its C++ counterpart is `io`. Not every C++
API has a Python binding. Follow an existing neighboring API through all layers
before changing its behavior.

## Build and test

Run from the repository root with the desired Python environment active. Install
NumPy, SciPy, and scikit-learn in that environment for the full Python suite;
also provide CMake, Git, and a C++17 toolchain. Configuration fetches Catch2 even
when C++ tests are disabled. See the [build guide](docs/agents/build-and-test.md)
for offline configuration, platform details, and dependency versions.

```bash
cmake -S . -B build/agents-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DHG_USE_TBB=OFF
cmake --build build/agents-debug --target higram test_exe --parallel 2
ctest --test-dir build/agents-debug --output-on-failure
```

Targeted tests, using the same Python selected above:

```bash
(cd build/agents-debug && ./test/cpp/test_exe '[component_tree]')
(cd build/agents-debug && python -m unittest discover \
  -s test/python -p 'test_component_tree.py' -v)
```

CTest registers `Test_cpp` and `Test_python` as whole-suite entries; use Catch2
filters or unittest discovery to select individual cases. Python sources and
tests are copied at CMake configuration time. Reconfigure/build after edits and
run Python from the build directory. Merely setting `PYTHONPATH` while staying
at the repository root can still import the source package first.

There is no repository-configured formatter, linter, or corresponding command.
Match adjacent code; avoid broad formatting changes. Check whitespace with
`git diff --check`. Documentation generation is `make -C doc html` with the
development package importable; see the maintenance guide before installing its
requirements, which include a released Higra package.

## Constraints agents must respect

- Preserve static tree indexing: leaves first, parents after children, root last
  and its own parent. Preserve tree category and maps between old and new nodes.
- Keep topology separate from weights; weights' first axis follows the relevant
  vertex/edge/node IDs. Use `hg::index_t` and `hg::invalid_index` consistently.
- Preserve public signatures, defaults, return ordering, dtype/shape behavior,
  concept links, and Python class extensions. The wrapper is part of the API.
- Register new Python files and tests in their module CMake files. Register new
  binding translation units, declarations, aggregate includes, and initializer
  calls. An unregistered source test can pass locally yet never run in CTest.
- Do not return lazy xtensor expressions or borrowed views of destroyed locals.
  Review NumPy return policies, backing storage, mutation, and iterator lifetimes.
  A `keep_alive` policy does not prevent vector reallocation.
- Account for identity-based caching of mutable inputs. Use existing cache
  controls when recomputation is needed; do not assume mutations invalidate it.
- Preserve validation and exception behavior. `hg_assert` throws in Release too;
  it is not the standard debug-only `assert`.
- Preserve stable-sort semantics and dependencies in tree traversals. A function
  named "parallel" need not use threads. Do not assume const tree operations are
  thread-safe: lazy child computation changes shared state.
- Edit sources, not build-tree copies, generated XML/HTML, or packaging copies
  under `higra/include` and `higra/lib`. Treat vendored `lib/` changes as explicit
  dependency work; its update script deletes and recreates directories.
- Keep existing license notices and follow the neighboring file structure.
  Leave unrelated worktree changes intact. The existing contributor page has
  stale instructions; current code and CMake registration are authoritative.

## Before declaring a change complete

1. Build the affected targets and run focused tests. For C++ changes, include
   C++ tests; for wrappers/bindings, include public Python API tests. Check that
   tests imported the intended extension. Run the broader relevant suite after
   focused checks; explain any unavailable checks.
2. Exercise changed contracts: supported graph/dtype overloads, scalar versus
   vector data, image shapes, maps/metadata, invalid inputs, and lifetimes as
   applicable. For parallel changes, check TBB enabled and disabled; for
   incremental trees, use the existing structural and rebuild-baseline tests.
3. Update public docstrings/API pages for changed public behavior. Inspect the
   diff and whitespace. Documentation-only changes need link/command review;
   do not rebuild the entire library solely for prose edits.

## Detailed guidance

- [Architecture and API paths](docs/agents/architecture.md)
- [Build, test, debugging, and CI](docs/agents/build-and-test.md)
- [C++ development and performance](docs/agents/cpp-development.md)
- [Python bindings and frontend conventions](docs/agents/python-bindings.md)
- [Incremental component trees](docs/agents/incremental-component-trees.md)
- [Documentation, generated files, and maintenance](docs/agents/documentation-and-maintenance.md)
