# Python bindings and frontend development

The extension is `higra.higram`; `higra/__init__.py` aliases it as `hg.cpp` and
exports its public names. Most users call Python wrappers in the top-level
`higra` namespace. Preserve both layers' contracts when changing an API.

## Registering functionality

For a new operation following the established wrapper pattern:

1. Implement the C++ algorithm in the relevant `include/higra/<module>/` header
   and add C++ tests if that layer changes.
2. Add `higra/<module>/py_<name>.hpp` declaring the initializer in its binding
   namespace, and `.cpp` implementing it. Use an adjacent binding as the template.
3. Add the `.cpp` to the module's `PYMODULE_COMPONENTS` list, include its
   declaration in the module `all.hpp`, and call its initializer in
   `higra/pymodule.cpp`. Module source lists propagate with `PARENT_SCOPE`.
4. Add a wrapper to a registered Python file, or register a new file in `PY_FILES`
   and export it from the module `__init__.py`. Preserve import ordering for any
   dependencies used at module definition time.
5. Add tests to `test/python/test_<module>/` and its `PY_FILES` CMake list. Add
   public documentation as described in the maintenance guide.

For an entirely new subpackage, also add its `add_subdirectory` in
`higra/CMakeLists.txt`, its aggregate include in `higra/all.hpp`, and its package
export in `higra/__init__.py`. Current setuptools uses namespace-package discovery;
the contributor page's instruction to edit a fixed package list is obsolete.

`pymodule.cpp` owns `FORCE_IMPORT_ARRAY` and `xt::import_numpy()`. Do not duplicate
the module definition or NumPy import ownership in each binding file. Its
exclusion from unity batches is explicit in `higra/CMakeLists.txt`.

## Binding patterns and overloads

`higra/py_common.hpp::add_type_overloads` calls a registration functor's
`template def<T>` for each supplied type. `include/higra/utils.hpp` defines the
`HG_TEMPLATE_*` lists for integral, signed, floating, and numeric values. Many
binding files then instantiate that functor for each supported graph type.

Check a function's actual overload list. Availability of a `RegularGraph5d` class
does not imply every algorithm is bound for dimension five; component-tree
construction, for example, registers dimensions one through four. CASF currently
registers `ugraph` and the 2D regular grid graph. Preserve deliberate restrictions
and avoid multiplying overloads without a use case: template instantiation adds
compile time and binary size.

Use `py::arg` names/defaults consistent with the public wrapper and neighboring
bindings. Some APIs use `.noconvert()` to require an existing matching NumPy
dtype; others allow conversion. This is an observable difference, especially
for in-place operations and overload selection.

`xt::pyarray<T>` and `xt::pytensor<T,N>` are Python-owned NumPy-backed containers,
not the owning C++ `hg::array_*` aliases. Their vendored casters can wrap matching
arrays or create converted arrays. Dynamic layout does not universally require
contiguity. Review dtype, rank, strides, and mutation behavior for the actual
container and algorithm; do not add a blanket contiguous-copy rule.

Hierarchy bindings often move owned `tree` and `altitudes` members into a Python
tuple. Accumulator bindings dispatch a runtime `hg::accumulators` enum to a C++
accumulator type through `higra/accumulator/common.hpp`.

## Array ownership and lifetime

The authoritative conversion implementation is
[xtensor_type_caster_base.hpp](../../lib/include/xtensor-python/xtensor_type_caster_base.hpp), together with the
`pyarray`/`pytensor` casters. Owning xtensor values returned by value are moved to
a heap object retained by a Python capsule. Default lvalue array returns are
copied; `reference_internal` instead associates a view with its parent Python
object. Const references produce non-writeable arrays in this path.

| Existing API | Ownership / mutation behavior |
| --- | --- |
| `hg.Tree(parent_array)` | C++ owns an evaluated parent array; does not share the constructor's NumPy storage |
| `Tree.parents()` | Read-only array referencing tree-owned storage, with `reference_internal` |
| `Tree.children(node)` | Allocates and returns a copy of the child IDs |
| `Tree.sources()` | Python creates a new `np.arange` array |
| `Tree.targets()` | Python slices `parents()`, retaining its backing reference |
| `UndirectedGraph.sources()/targets()` | Zero-copy, read-only strided ndarrays; private base owners retain the storage generation and graph |
| `SimplifiedTree.tree()/node_map()` | Internal references tied to the result object |

Explicit-graph endpoint arrays retain their creation-time length and storage
allocation, including through derived views. In-place edits can change their
values; growth detaches shared storage only before relocation, so older arrays
may become stale. They are not immutable creation-time snapshots. Obtain a new
array for current topology or copy before mutation for stable values. NumPy
cannot enable writeability through the private capsule base. Both explicit graph
variants use this binding contract; the optimized-delete variant retains its
underscored accessor API. Native endpoint adapters remain borrowed and invalid
after graph mutation, assignment/move, or destruction.

Returning an adapter by value does not make its pointed-to storage owned. For a
new borrowed result, explicitly review the backing owner and invalidation rules.
Likewise, several existing `py::make_iterator` bindings have no explicit
`keep_alive` policy. Keep the backing graph/tree alive and unchanged during
iteration; do not copy a neighboring iterator binding blindly as proof of safe
lifetime handling.

For changed views/references, validate values, whether storage is shared,
writeability, deletion of external references, and structural mutation where
applicable. For in-place operations, confirm the caller's array was changed,
rather than a temporary made by conversion.

## Wrapper and public API conventions

Wrappers commonly call `linearize_vertex_weights` for grid images, preserve
trailing feature dimensions, then use `delinearize_vertex_weights` for image-shaped
outputs. Shape is associated with graphs through `CptGridGraph`. The helpers
infer intent from shapes and can be ambiguous; preserve existing behavior and
test the concrete scalar/vector inputs affected by a change.

Hierarchy constructors link outputs with `CptHierarchy.link(tree, graph)`.
RAGs and binary hierarchies also carry maps/MST metadata through their concepts.
Without those links, the low-level result may look correct while higher-level
functions cannot resolve optional arguments.

`@hg.argument_helper(...)` fills missing or `None` parameters using concepts.
It inspects the underlying function through its `original` attribute; preserve
that chain and introspection when composing wrappers.

Attribute computations recompute on every call. Do not introduce `auto_cache`
uses: its public decorator and state/clear APIs are deprecated warning-emitting
no-ops. For one release, formerly cached functions use `_legacy_cache_keywords`
as the outer decorator to discard implicit `no_cache`, `force_recompute`,
`attribute_name` and `data_cache` controls with `FutureWarning`. Ordinary calls
stay quiet. Declared parameters with those names retain their meaning. Keep
the deprecated explicit `Concept.construct(data_cache=...)` custom-store path
separate from this retired result-cache compatibility. It is the only Concept
custom-store boundary. Plain `argument_helper` rejects undeclared `data_cache`;
do not intercept a real declared parameter of that name.

Metadata lives directly in dynamic instance fields; use private helpers or direct
assignments internally and `Concept.tag` in custom Concept links. No global store,
weak-key fallback, attribute mirror or legacy-pickle migration is permitted.
Public metadata/tag helpers warn and become private next release. Both attribute
clearing functions are warning-emitting no-ops scheduled for removal. Keep maps,
shapes, Concept links and explicit LCA/lazy MST state intact. Dynamic owners need
no weak-reference support; plain ndarrays cannot own metadata. Current-version
pickle reducers must preserve instance state and shared associations.
Callers can retain computed results explicitly and recompute after input changes.

Class behavior is augmented through `@hg.extend_class` in `higra/structure/`.
Inspect both binding and Python methods, including `__new__`, `__init__`, and
`__reduce__` when relevant. Bound classes frequently enable `py::dynamic_attr()`
to support metadata and pickle state. Do not remove it without reviewing those
uses. Top-level star exports also mean a new non-private helper may unintentionally
become a public name.

## Errors, callbacks, and concurrency

No Higra-specific exception translator is registered. The bundled pybind11
translator maps `std::runtime_error` to `RuntimeError`, `std::invalid_argument`
to `ValueError`, `std::out_of_range` to `IndexError`, and allocation failures to
`MemoryError`. Python wrappers have their own checks and exception types. Match
the existing boundary's behavior, not just the C++ result.

Bindings include `pybind11/functional.h` for callback APIs, including custom
agglomeration rules. Current bindings do not explicitly release the GIL. Adding
GIL release requires auditing Python callbacks, NumPy object creation/destruction,
logging callbacks, borrowed mutable data, and shared lazy state. Do not add a
global release policy as a routine performance change.

`hg.set_num_threads` controls a oneTBB global limit when compiled with TBB; zero
resets it to the default concurrency. Without TBB it logs a warning. This does
not imply Python calls run concurrently or every algorithm uses multiple threads.

Use public API tests for signatures, return order, concepts, image shapes,
dtype/graph overloads, exceptions, and ownership where changed. Exact focused
commands and import-path checks are in [build-and-test.md](build-and-test.md).
