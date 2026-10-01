# Incremental component trees and CASF

This subsystem needs separate guidance because mutations break assumptions that
are valid for ordinary static `hg::tree` objects. Its primary sources are
[dynamic_component_tree.hpp](../../include/higra/detail/hierarchy/dynamic_component_tree.hpp),
[dynamic_component_tree_attribute_computers.hpp](../../include/higra/detail/hierarchy/dynamic_component_tree_attribute_computers.hpp),
[dual_min_max_tree_incremental_filter.hpp](../../include/higra/detail/hierarchy/dual_min_max_tree_incremental_filter.hpp),
and [component_tree_casf.hpp](../../include/higra/hierarchy/component_tree_casf.hpp).

## Representation and ID spaces

`hg::detail::hierarchy::DynamicComponentTree` separates input vertices (proper
parts) from mutable internal nodes. For `L` proper parts and `I` internal slots:

| Value | Meaning |
| --- | --- |
| `[0, L)` | Proper-part global IDs, matching input graph vertices |
| `[L, L+I)` | Internal-node global IDs; some slots can be released |
| `[0, I)` | Local internal-slot IDs |
| `global = local + L` | Conversion for internal nodes only |
| `getGlobalIdSpaceSize()` | Full space `L+I`, including released slots |
| `getNumNodes()` | Alive internal nodes, not full buffer capacity |

The implementation's storage rules are:

- `nodeParent` is indexed by local slot and stores a **global** parent ID or
  `invalid_index`. A root stores its own global ID.
- Internal child/sibling links are indexed by local slot and store **local**
  slots or `invalid_index`.
- `properPartOwner` is indexed by proper-part ID and stores the owning internal
  node's **global** ID.
- Proper-part lists connect global proper-part IDs; list heads/tails/counts are
  indexed by internal local slot.
- The free list stores local slots. Allocation returns a global node ID.

The class-level prose says parent storage uses local IDs, while `buildFromParent`,
`getNodeParent`, and mutation methods use global values. Follow the implementation
and tests; do not apply the class-level statement indiscriminately.

Surviving IDs remain stable through topology edits; released slots can be reused.
Do not infer current ancestor order from numeric ID order, assume root is the
largest live ID, or size altitude/attribute buffers by the live node count.

## Mutation and iteration

Use methods that maintain both sides of relationships: moving children must
update parent links and sibling lists; moving proper parts must update ownership
and linked lists. `pruneNode` removes a subtree and transfers proper parts to the
parent. `mergeNodeIntoParent` promotes children/proper parts. Root, detached,
released, and empty-node cases have distinct preconditions; consult the method
and its tests before composing low-level edits.

Ranges are lazy and versioned. Alive-node iteration is invalidated by node-set
changes, topology traversals by structural changes, and proper-part iteration by
ownership changes. They fail fast when invalidated. Materialize the intended
candidate IDs before mutating through them, or use an explicitly mutation-aware
algorithm. Materializing IDs does not guarantee the nodes remain alive: validate
them after earlier edits. Do not suppress version checks to allow mutation inside
a normal traversal.

## Dual-tree updater ownership and attributes

`DualMinMaxTreeIncrementalFilter<altitude_t, graph_t>` keeps borrowed pointers to
the min/max trees, shared graph, attribute computers, and attribute buffers.
`setAltitudeBuffers` stores raw data pointers and sizes. All of these must outlive
the updater, and altitude storage must remain at the registered addresses; renew
registration after any reallocation. The graph is a fixed domain during updates.

Buffers cover the full global ID space. Keep altitude, proper-part ownership,
topology, and incremental attribute summaries synchronized after local edits.
Attribute computers in `dynamic_component_tree_attribute_computers.hpp` support
area and bounding-box measures; they manage auxiliary summaries as well as
outputs, so editing only the visible attribute value is insufficient.

Level merging selects dense buckets for small integral domains and sparse
ordered maps for larger integral/floating domains. The default
`HG_COMPONENT_TREE_ADJUSTMENT_DENSE_MAX_BITS` is eight. Signed values use an
order-preserving mapping. Preserve comparisons and numeric limits; do not cast
float levels to integer buckets or use raw signed values as dense indices.

## CASF contracts and exports

`hg::ComponentTreeCasf` owns the mutable trees, altitude/attribute buffers,
attribute computers, updater, and reusable candidate buffers. It borrows the
graph. Its helper references member objects, so audit those references before
moving the enclosing state or changing buffer allocation.

Each threshold applies max-tree pruning/attribute opening first, then min-tree
pruning/attribute closing. Candidate selection uses `attribute <= threshold`,
selects maximal non-root subtrees, and does not prune the root. Threshold order
is preserved; the wrapper does not sort the sequence.

C++ `filter` is stateful: later calls continue from the filtered state. An empty
sequence reconstructs the current state without further filtering. The Python
`connected_alternating_sequential_filter` binding constructs a new CASF object
per call, so separate Python calls do not share that state.

Area works on supported graph types; bounding boxes require a 2D embedding.
Current Python overloads cover explicit `ugraph` and the implicit 2D regular
graph; unsupported bounding-box inputs are rejected. The wrapper resolves string
or enum attributes and restores input image shape.

`exportMaxTree`/`exportMinTree` reindex alive internal nodes after leaves and
reconstruct leaf altitudes from their current owner. Exported static IDs are not
the dynamic IDs. The current exporter constructs `hg::tree(parents)` with the
default partition-tree category; do not assume exported category is component
tree merely from the method name/comment. Treat category changes as a separately
reviewed behavior change.

## Existing validation workflows

Run the affected tags from a configured build directory:

```bash
./test/cpp/test_exe '[dynamic_component_tree]'
./test/cpp/test_exe '*computer*'
./test/cpp/test_exe '[dual_min_max_tree_incremental_filter]'
./test/cpp/test_exe '[component_tree_casf]'
python -m unittest discover -s test/python \
  -p 'test_component_tree_dual_filter.py' -v
```

The attribute-computer cases currently have no tag, so their command uses a
test-name wildcard. Check that the two cases in
`test_dynamic_component_tree_attribute_computers.cpp` actually ran. The structural
tests cover both mapping directions, stable surviving IDs, slot reuse, root
cases, and iterator invalidation. Dual-tree tests compare updates to
rebuilding and verify structure/areas. CASF tests cover export, determinism, empty
sequences, dense/sparse agreement, and naive reference filters. Its stress case
includes large structured images up to 1014 by 1014; select by name to isolate a
failure before rerunning the full tag.

When changing topology or incremental attributes, retain rebuild-based oracles
and structural checks. Comparing only a final image can miss inconsistent
internal state that fails on the next threshold. See
[build-and-test.md](build-and-test.md) for configuration and targeted filters.
