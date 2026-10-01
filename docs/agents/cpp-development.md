# C++ development

## Core conventions

Use the surrounding header as the style reference: `#pragma once`, the existing
license notice, namespace `hg`, snake_case in most of the established API, and
Doxygen comments for public algorithms. Recent dynamic component-tree classes
use CamelCase classes and camelCase methods; match that subsystem rather than
renaming it to match older code. There is no configured formatter.

Algorithms commonly take a generic `graph_t` and `const xt::xexpression<T>&`,
then obtain `x.derived_cast()`, validate it, and work with its concrete type.
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

## Arrays, expressions, and ownership

`structure/array.hpp` supplies owning `array_1d` through `array_4d` and `array_nd`
aliases. Fixed-rank arrays and fixed-size points allow compile-time optimization.
Use the most specific representation that fits the operation, while preserving
support for generic expressions at API boundaries.

The first axis normally indexes graph entities; remaining axes describe vector
data. Check existing scalar/vector dispatch before assuming every weight array
is one-dimensional. `structure/details/light_axis_view.hpp` deliberately offers
a small, efficient slice interface, without general broadcasting.

xtensor operations can be lazy. Do not return `a + b`, a view, or an adaptor when
its operands/storage die on return. Store into an owning array or use `xt::eval`
when materialization is needed. `xt::adapt` can either wrap storage or produce a
representation subsequently copied into an owning result; inspect the exact
construction and lifetime rather than inferring ownership from `auto`.

`HG_ADAPT_STRUCT_ARRAY` adapts fields of an array of structs using
`xt::no_ownership()`. Explicit-graph endpoint arrays use this machinery and are
strided. Their storage belongs to the graph's vector and can move on insertion.

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
sorting or oneTBB plus the bundled `tbb-ssort` stable sort. `XTENSOR_USE_TBB` and
`HG_USE_TBB` are enabled on the bindings/test targets when requested; SIMD is
separately controlled by `USE_SIMD`.

Performance-sensitive paths include component-tree vertex sorting and union-find,
agglomeration, accumulator loops, LCA/RMQ preprocessing and queries, image
hierarchies, and incremental updates. Preserve existing stable sorting for equal
weights: component-tree construction uses `stable_arg_sort` specifically.

Use independent output ranges in parallel workers. Do not share mutable view
positions or accumulator storage across workers. Avoid repeated allocations,
materialization of implicit graphs, and repeated evaluation of expressions in
inner loops unless a change requires them. Check both TBB modes for changed
parallel behavior; the Python bindings currently retain the GIL.

`benchmark/CMakeLists.txt` is not a turnkey performance suite: most benchmark
sources are disabled and it requires TBB. See the build guide before enabling it.

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
