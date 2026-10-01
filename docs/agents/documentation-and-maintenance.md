# Documentation and maintenance

## Published documentation

The existing documentation source is `doc/source/`, not `docs/`. Configuration is
[doc/source/conf.py](../../doc/source/conf.py); navigation is
[doc/source/index.rst](../../doc/source/index.rst). It accepts `.rst` and uses
Sphinx autodoc/autosummary, Breathe, Napoleon, and sphinx-tabs. The Markdown agent
guides are outside that build and do not require a new Sphinx extension.

Python API pages under `doc/source/python/` use `.. currentmodule:: higra`,
autosummary entries, and `.. autofunction::`/class directives. Add a new public
function to the appropriate module page and include new pages in the relevant
toctree when needed. Public Python docstrings use reStructuredText parameters,
returns, examples, cross-references, and mathematical markup. Preserve valid
Python string escaping when adding LaTeX expressions.

C++ documentation comes from header comments under `include/`.
`doc/Doxyfile` recursively generates XML in `doc/xml`, omits symbols matching
`*internal*`, and disables Doxygen HTML. Breathe consumes the XML through the
`higra` project and `doc/source/cpp/cpp_all.rst` renders the `hg` namespace. A new
C++ header in that include tree does not require its own Doxygen input entry.
The current C++ API page is a raw namespace listing, not curated per-module docs.

## Build documentation against the checkout

First build the development extension as described in
[build-and-test.md](build-and-test.md). Install the Sphinx tools and provide the
external `doxygen` executable:

```bash
python -m pip install 'sphinx==8.2.3' sphinx_rtd_theme sphinx-tabs breathe
```

This command mirrors the documentation tool requirements while omitting their
released Higra pin. `doc/requirements.txt` also installs `higra==0.6.13`; simply
installing that file can make autodoc describe a released API instead of your
changes.

From the repository root, with the configured Python environment active:

```bash
HG_DOC_PACKAGE="$(pwd)/build/agents-debug"
(cd doc && PYTHONPATH="$HG_DOC_PACKAGE${PYTHONPATH:+:$PYTHONPATH}" \
  python -c 'import higra as hg; print(hg.__file__); print(hg.cpp.__file__)')
PYTHONPATH="$HG_DOC_PACKAGE${PYTHONPATH:+:$PYTHONPATH}" \
  make -C doc html SPHINXBUILD="$(command -v sphinx-build)"
```

The working directory during generation is `doc/`, so the source root does not
shadow the build package as it does in ordinary root-level Python runs. Confirm
the printed paths and that `sphinx-build` belongs to the same environment. The
Makefile runs Doxygen before Sphinx; output is `doc/build/html/`. For changed API
pages, inspect the relevant generated page and warnings. Do not claim a clean
documentation build if Doxygen or Sphinx was unavailable.

Read the Docs is configured in `readthedocs.yml` for Ubuntu 24.04, Python 3.13,
the Sphinx config above, and `doc/requirements.txt`. The config invokes Doxygen
when `READTHEDOCS=True`. Its requirements import a released wheel rather than
building this checkout; updating docs for an unpublished API may therefore need
a corresponding release before hosted autodoc can find it.

## Generated and copied artifacts

| Artifact | Source / generator | Editing rule |
| --- | --- | --- |
| Build-tree `higra/**/*.py` and `test/python/**/*.py` | Module CMake lists and `REGISTER_PYTHON_MODULE_FILES` | Edit the source and reconfigure/build |
| Extension `.so`/`.pyd`, executables, CMake caches, unity sources, `_deps` | CMake/compiler | Rebuild; do not patch generated output |
| Build-tree `setup.py`, `README.md`, test resources | Root/resource `configure_file` calls | Edit original files |
| `higra/include/`, `higra/lib/` | setup.py temporary copies of `include/`, `lib/` | Do not edit, commit, or create them as permanent source directories |
| `dist/`, `build/`, egg-info, repaired wheels and DLL/link artifacts | setuptools and packaging scripts | Treat as packaging output |
| `doc/xml/`, `doc/build/`, autosummary output | Doxygen/Sphinx | Edit comments, docstrings, or source pages |
| Installed vendor headers/CMake files in `lib/` | `lib/update_lib.sh` and upstream releases | Modify only as explicit dependency work |

setup.py creates the temporary header directories even for metadata processing
and cleans up directories it successfully registered in its `finally` block.
Pre-existing copies can make `copytree` fail. Diagnose the origin of such paths
before cleanup; do not remove unrelated content by assumption. `MANIFEST.in`
describes packaged header/vendor files and the renamed Windows TBB runtime.

Installed packages expose `get_include`, `get_lib_include`, and `get_lib_cmake`
for external C++ extensions. An ordinary CMake build tree lacks the packaging
copies, so those helpers can warn or return missing paths there. Use checkout
`include/` and `lib/include/` paths for repository C++ builds.

## Dependency and version maintenance

[lib/update_lib.sh](../../lib/update_lib.sh) pins vendor versions and deletes
existing vendor include/config/license trees before downloading and reinstalling
them. Its install prefix is the **current working directory**, intended to be
`lib/`. Running it from the repository root would delete important root paths.
Do not run it as a setup step. For an explicitly requested update, inspect its
paths, intended versions, header-layout changes, licenses, and resulting diff.
The bundled `tbb-ssort` implementation is separate from its main download list.

The principal version source is `include/higra/config.hpp`. Root CMake and
setup.py derive versions from those macros; `hg.version()` returns them too.
`higram.__version__` is `dev` unless `VERSION_INFO` is defined, as setup.py does.
Use `hg.version()` rather than the extension attribute when diagnosing a direct
CMake build. The released Higra pin in `doc/requirements.txt` is a second value
to review during version work.

NumPy's isolated-build pins in `pyproject.toml`, runtime minima in `setup.py`,
and CI wheel inputs serve different purposes. Change them together only when the
requested compatibility work requires it. Preserve vendor license files and
existing source attribution notices.

The contributor page's release section is marked admin-only and describes
version/tag/publication operations. Current Azure templates can publish wheels
on tags. Creating agent documentation or validating a change does not authorize
a release, tag push, or package upload.

## Existing guidance and evidence limits

`doc/source/contributing.rst` remains useful as a registration checklist, but its
Boost test initialization, `py_module.cpp` filename, Python import example, and
fixed setuptools package-list instruction do not match current implementation.
Use `pymodule.cpp`, Catch2 v3, ordinary module imports without a `.py` suffix, and
current namespace-package discovery.

The active CI entry point is `.azure-pipelines.yml`, with templates under
`tools/`; old standalone wheel scripts are not evidence for the current support
matrix. See the build guide for the configured environments and their limits.

For these agent guides, validate relative links and referenced paths, inspect
shell examples against current targets/options, and check whitespace. They are
not part of Sphinx, so generating the published site does not validate them.
Distinguish commands verified on an existing local build from clean-build,
platform, and hosted-CI claims. Do not turn discovered inconsistencies into
unrequested code changes while maintaining documentation.
