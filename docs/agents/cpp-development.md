# C++ development

## Core conventions

Use the surrounding header as the style reference: `#pragma once`, the existing
license notice, namespace `hg`, snake_case in most of the established API, and
Doxygen comments for public algorithms. Recent dynamic component-tree classes
use CamelCase classes and camelCase methods; match that subsystem rather than
renaming it to match older code. There is no configured formatter.

Algorithms commonly take a generic `graph_t` and array inputs as
`array_view<const T>` by value (`T` is the element type), validate them, and
return `array<T>` by value; in-place parameters are `array_view<T>`.
Internal helpers often live in `<module>_internal` namespaces. `detail/hierarchy`
instead uses `hg::detail::hierarchy`. Inspect the existing location before adding
a new helper or exposing an internal type.

Primary contracts are defined in [utils.hpp](../../include/higra/utils.hpp),
[graph.hpp](../../include/higra/graph.hpp), and
[tree_graph.hpp](../../include/higra/structure/tree_graph.hpp).

There is no central C++ umbrella containing all algorithms. `graph.hpp` collects
graph structures and free-function adapters; include an algorithm's own header.
The unrelated `higra/all.hpp` is an aggregate of Python binding declarations.

## Graph and tree contracts

- `utils.hpp` defines `index_t = int64_t`, `size_t = std::size_t`, and
  `invalid_index = -1`. Do not convert an invalid index to an unsigned size before
  checking it. Descending loops require signed indices.
- Graph vertices and edges index separate data arrays. Prefer free functions
  such as `num_vertices`, `source`, `target`, `parent`, and the range helpers in
  `graph.hpp` when extending generic algorithms.
- Explicit `ugraph` stores indexed edges and adjacency containers. Endpoints
  are normalized on insertion. Removal sets endpoints to `invalid_index` and
  removes adjacency entries while retaining the edge slot and total slot count.
  Algorithms traversing edge storage must consider whether deleted slots are
  acceptable; do not assume `num_edges()` counts only live edges.
- Regular grid graphs are implicit, with embedding/neighborhood behavior.
  Materializing them into explicit graphs can change memory use substantially.
- A nonempty static `tree` has contiguous leaves before internal nodes,
  `parent(v) > v` for every non-root node, and root `num_vertices()-1` with
  `parent(root) == root`. Leaves map to the represented graph's vertices.
- Preserve `tree_category::component_tree` versus `partition_tree`. Algorithms
  validate or rely on the distinction.
- Remapping results map **new node IDs to original node IDs**. Check map direction
  and how node attributes/leaf metadata are transferred after transformations.

`tree_graph.hpp::_init` checks several structural contracts, but does not make
every arbitrary parent array safe: for example, the parent loop indexes a count
array without an explicit upper-bound check on each parent. Add validation at the
relevant boundary when changing accepted input behavior; do not advertise more
input validation than the implementation provides.

## Lazy state and traversals

`tree::compute_children()` fills mutable child lists even on a const object. It
has no synchronization. Algorithms requiring `children_iterator` explicitly
compute the relation first; `clear_children()` discards it.

Do not start concurrent first-time child computation on the same tree. Prepare
required shared state before parallel readers, and do not clear it during their
work. References and iterators into child lists are invalid after clearing them.

Node order makes leaves-to-root and root-to-leaves traversals inexpensive.
Respect data dependencies: parent accumulations cannot be arbitrarily moved into
`parfor`. In accumulator names, "parallel" often describes reduction semantics
over direct children, in contrast to recursive sequential aggregation; it does
not guarantee execution by multiple threads.

## Arrays, views, and ownership

`structure/array.hpp` includes the array library (`structure/array/`) and defines
`array_1d` … `array_nd` as aliases of `hg::array<T>`: owning, C-contiguous, value
semantics, rank known only at run time (0–8). The public reference is
[doc/source/arrays.rst](../../doc/source/arrays.rst). Conventions:

- Read-only inputs: `template<class T> R f(array_view<const T> a)`. It accepts
  arrays, views, const arrays and temporaries with one instantiation per element
  type. Mutable inputs: `array_view<T>` (rejects const arrays and temporaries at
  compile time). Results: `array<T>` by value. Members that keep data are
  `array<T>`; never store a view past the call that received it.
- The rank is not part of the type: keep `hg_assert_1d_array` and similar checks
  where a rank is required. Overloads that differ only by rank collide.
- There are no arithmetic operators, broadcasting, strides or lazy expressions.
  Write loops on `data()`/`operator()`, or use the eager free functions of
  `structure/array/functions.hpp` (`map`, `assign`, `cast`, `take`, `put`, reductions).
  Sub-views are first-axis only: `row(i)`, `slice(b, e)`, `reshaped`, `flat`;
  anything else (columns, strided sub-volumes) is an explicit loop.
- Brace initialization means data, never a shape: `array<int>{5}` is `[5]`. Shapes
  go through `array_shape`, `empty`/`zeros`/`full`, or `from_shape`. Beware the
  vexing parse `array<T> x(array_shape(n));`.
- Sizes and indices are signed `index_t`; graph sizes are `size_t`, so cast once at
  comparisons. `operator()`/`operator[]` are checked only with `HG_ARRAY_CHECKS`
  (non-Release CMake builds), whereas shape checks in functions use `hg_assert`.
- `point<T, dim>` is a plain struct with inline storage; `to_array` converts it.

The first axis normally indexes graph entities; remaining axes describe vector
data. Check existing scalar/vector dispatch before assuming every weight array
is one-dimensional. `structure/details/light_axis_view.hpp` deliberately offers
a small, efficient slice interface, without general broadcasting.

Views are non-owning. Do not return a view (`row`, `slice`, `reshaped`, a
`strided_span`) whose storage dies on return, and do not keep one across a
`resize`, assignment or move of its array; copy into an `array<T>` when the data
must outlive the source.

Explicit-graph endpoint arrays (`undirected_graph::sources()/targets()`) are
`strided_span<const index_t>`: a field of the edge structs, strided, const, and
borrowed; consumers are templated on "1-d indexable" (`strided_span` or
`array_view`), the only exception to the single-input-type rule.
Mutation, assignment, moving from the graph,
or destruction invalidates them; reacquire after edits. Explicit graphs own a
shared edge-storage generation. Python endpoint arrays retain a const storage
handle and graph reference. Insertion with insufficient capacity copies a shared
generation into a larger vector before switching; an unshared vector can grow
normally. In-place set/remove operations do not detach. Native spans do not
retain storage, and an exported allocation does not extend their validity.
Graph copies deep-copy storage and adjacency to preserve independent values.

`make_node_weighted_tree` and `make_remapped_tree` use forwarding references and
deduce member types. Passing lvalues can produce reference members. Review
ownership at construction; use appropriate moves for owned results and do not
return aggregates referring to locals.

Cross-language array policies are described in
[python-bindings.md](python-bindings.md). Dynamic-tree borrowed buffers have
additional rules in [incremental-component-trees.md](incremental-component-trees.md).

## Validation and errors

Use nearby `hg_assert_*` helpers for shape, entity count, category, and index
validation. `HG_MAIN_ASSERT` is defined in `utils.hpp`, so `hg_assert` throws
`std::runtime_error` in optimized builds too. Its diagnostic includes function,
file, and line information. Do not replace public validation with ordinary
`assert`, which can disappear under `NDEBUG`.

Integral-type helpers use `static_assert`; other helpers perform runtime checks.
Index-array checks use minimum/maximum reductions, so empty-input behavior needs
separate consideration. These checks are not a universal validation layer.

pybind11's existing translation maps runtime errors to Python `RuntimeError`;
changing exception type can change a public API. Tests include `REQUIRE_THROWS`
and Python `assertRaises`. Preserve expected failure behavior as well as values.

`HG_TRACE` is compiled in for configurations not matching Release, and runtime
tracing is disabled by default. Logging/trace controls are bound in
`higra/detail/py_log.cpp`; see the build guide for debugging commands.

## Performance and parallelism

`utils.hpp::parfor` selects TBB or a serial loop. `sorting.hpp` selects standard
sorting or oneTBB plus the bundled `tbb-ssort` stable sort. `HG_USE_TBB` is
enabled on the bindings/test targets when requested. There is no explicit SIMD
code (`USE_SIMD` is a deprecated no-op).

Under `HG_USE_TBB`, array-library loops over at least `HG_ARRAY_PARALLEL_THRESHOLD`
elements (default 2^16) run through `hg::detail::parallel_blocks` (static
partitioner): copies, `assign`, `cast`, `take`, reductions and comparisons. Fills,
`arange` and `put` stay sequential (scatter races; parallel fills hurt the
sequential consumer's cache locality). Results must be bit-identical with TBB on and
off and for any thread count: floating-point sums use fixed `HG_ARRAY_REDUCE_CHUNK`
chunks combined in chunk order, and `argmin`/`argmax` return the first index.
`map`/`map_into` run the caller's function sequentially unless `hg::exec::par` is
passed, which asserts that the function is thread-safe (lazy tree children, for
example, are not).

Performance-sensitive paths include component-tree vertex sorting and union-find,
agglomeration, accumulator loops, LCA/RMQ preprocessing and queries, image
hierarchies, and incremental updates. Preserve existing stable sorting for equal
weights: component-tree construction uses `stable_arg_sort` specifically.

Use independent output ranges in parallel workers. Do not share mutable view
positions or accumulator storage across workers. Avoid repeated allocations,
materialization of implicit graphs, and repeated evaluation of expressions in
inner loops unless a change requires them. Check both TBB modes for changed
parallel behavior; the Python bindings currently retain the GIL.

Most older sources in `benchmark/CMakeLists.txt` are disabled. BPT benchmarks
support TBB on/off; see the [benchmark guide](../../benchmark/README-bpt.md)
for the five anchor cases and measurement boundaries.

## Tests and integration

Add C++ tests under the corresponding `test/cpp/<module>/` directory and list
them in its `CMakeLists.txt`, propagating `TEST_CPP_COMPONENTS` to the parent.
The single `test_exe` links `Catch2::Catch2WithMain`; do not add another main or
follow the stale Boost initialization instruction in the contributor page.

Use `test/cpp/test_utils.hpp` for Catch2 macros and existing comparison helpers.
Choose an existing tag or a descriptive module tag so focused runs are possible.
Tests use exact structural comparisons where IDs are contractual, numeric
tolerances where appropriate, and bijection comparisons for arbitrary labels.
Match the operation's semantics rather than equating equivalent labelings with
identical integer arrays. Exact commands are in the build guide.
