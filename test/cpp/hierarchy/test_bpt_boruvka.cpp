/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/


#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/structure/unionfind.hpp"
#ifdef HG_USE_TBB
#include <tbb/global_control.h>
#endif
#include "../test_utils.hpp"
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>

namespace bpt_boruvka_tests {
    using namespace hg;
    namespace internal = hg::hierarchy_core_internal;

    template<typename E1, typename E2, typename edge_less_t>
    void check_traversal_history(const internal::bpt_boruvka_history &full,
                                 const E1 &src, const E2 &dst, index_t n, const edge_less_t &less) {
        using traversal = internal::bpt_boruvka_traversal;
        auto active = internal::bpt_boruvka_waterfall(src, dst, n, less, traversal::active);
        REQUIRE(active.raw_parent.size() == size_t(n - 1));
        std::vector<index_t> edge_to_slot(src.size(), invalid_index);
        for (index_t slot = 0; slot < n - 1; ++slot) edge_to_slot[active.selected_edges(slot)] = slot;
        auto original = [](const auto &history, index_t slot) {
            return slot == invalid_index ? invalid_index : history.selected_edges(slot);
        };
        for (index_t v = 0; v < n; ++v)
            REQUIRE(original(active, active.leaf_parent(v)) == original(full, full.leaf_parent(v)));
        auto actual_expand = internal::bpt_boruvka_expand(active, less);
        auto expected_expand = internal::bpt_boruvka_expand(full, less);
        for (index_t slot = 0; slot < n - 1; ++slot) {
            const auto a = edge_to_slot[full.selected_edges(slot)];
            REQUIRE(a != invalid_index);
            REQUIRE(original(active, active.raw_parent(a)) == original(full, full.raw_parent(slot)));
            REQUIRE(active.raw_basin(a) == full.raw_basin(slot));
            REQUIRE(original(active, actual_expand.first(a)) == original(full, expected_expand.first(slot)));
            REQUIRE(actual_expand.second(a) == expected_expand.second(slot));
        }
        for (auto numbering: {internal::bpt_boruvka_numbering::exact, internal::bpt_boruvka_numbering::basin}) {
            auto expected = internal::bpt_boruvka_export(full, less, numbering);
            auto actual = internal::bpt_boruvka_from_edge_order(src, dst, n, less, numbering, traversal::active);
            REQUIRE(actual.first == expected.first);
            REQUIRE(actual.second == expected.second);
        }
    }

    template<typename E1, typename E2, typename edge_less_t>
    auto check_numberings(const E1 &src, const E2 &dst, index_t n, const edge_less_t &less) {
        using numbering = internal::bpt_boruvka_numbering;
        auto history = internal::bpt_boruvka_waterfall(src, dst, n, less);
        check_traversal_history(history, src, dst, n, less);
        const auto raw_parent = history.raw_parent, raw_basin = history.raw_basin;
        auto expanded = internal::bpt_boruvka_expand(history, less);
        const index_t count = history.selected_edges.size(), total = n + count;
        for (index_t s = 0; s < count; ++s) {
            auto p = history.raw_parent(s);
            if (p != invalid_index) {
                REQUIRE(history.raw_basin(p) > history.raw_basin(s));
            }
            REQUIRE(expanded.second(s) >= history.raw_basin(s));
            p = expanded.first(s);
            if (p != invalid_index) {
                REQUIRE(expanded.second(p) > expanded.second(s));
            } else {
                REQUIRE(expanded.second(s) == *std::max_element(raw_basin.begin(), raw_basin.end()));
            }
        }
        auto exact = internal::bpt_boruvka_export_exact(history, less);
        auto relaxed = internal::bpt_boruvka_export(history, less, numbering::basin);
        auto checked = internal::bpt_boruvka_from_edge_order(src, dst, n, less, numbering::basin);
        REQUIRE(checked.first == relaxed.first);
        REQUIRE(checked.second == relaxed.second);
        checked = internal::bpt_boruvka_from_edge_order(src, dst, n, less);
        REQUIRE(checked.first == exact.first);
        REQUIRE(checked.second == exact.second);
        REQUIRE(history.raw_parent == raw_parent);
        REQUIRE(history.raw_basin == raw_basin);

        array_1d<index_t> order = xt::arange<index_t>(src.size());
        std::sort(order.begin(), order.end(), less);
        auto reference = internal::bpt_canonical_from_sorted_edges(src, dst, order, n);
        REQUIRE(exact.first == reference.first);
        REQUIRE(exact.second == reference.second);

        // Check that the returned order really reuses the expansion's basin
        // permutation, rather than silently falling back to global numbering.
        order = xt::arange<index_t>(count);
        std::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
            return expanded.second(a) < expanded.second(b) ||
                   (expanded.second(a) == expanded.second(b) &&
                    less(history.selected_edges(a), history.selected_edges(b)));
        });
        auto edge_to_exact = internal::bpt_boruvka_invalid_indices(src.size());
        for (index_t i = 0; i < count; ++i) {
            edge_to_exact(exact.second(i)) = n + i;
            REQUIRE(relaxed.second(i) == history.selected_edges(order(i)));
        }
        array_1d<index_t> to_exact = xt::arange<index_t>(total);
        for (index_t i = 0; i < count; ++i) {
            REQUIRE(edge_to_exact(relaxed.second(i)) != invalid_index);
            to_exact(n + i) = edge_to_exact(relaxed.second(i));
        }
        // Leaves map to themselves; internal nodes map by original edge ID.
        // Equality of every mapped parent checks the entire edge-labeled tree.
        REQUIRE(relaxed.first(total - 1) == total - 1);
        std::vector<std::vector<index_t>> children(total);
        for (index_t v = 0; v < total; ++v) {
            REQUIRE(to_exact(relaxed.first(v)) == exact.first(to_exact(v)));
            if (v + 1 < total) {
                REQUIRE(relaxed.first(v) > v);
                REQUIRE(relaxed.first(v) < total);
                children[relaxed.first(v)].push_back(v);
                if (v >= n) {
                    REQUIRE(less(relaxed.second(v - n), relaxed.second(relaxed.first(v) - n)));
                }
            }
        }
        hg::tree tree(relaxed.first);
        REQUIRE(num_leaves(tree) == static_cast<size_t>(n));
        REQUIRE(category(tree) == tree_category::partition_tree);
        std::vector<std::vector<bool>> region(total, std::vector<bool>(n));
        std::vector<bool> seen(src.size());
        for (index_t v = 0; v < n; ++v) {
            REQUIRE(children[v].empty());
            region[v][v] = true;
        }
        for (index_t v = n; v < total; ++v) {
            REQUIRE(children[v].size() == 2);
            auto a = children[v][0], b = children[v][1], edge = relaxed.second(v - n);
            REQUIRE_FALSE(seen[edge]);
            seen[edge] = true;
            REQUIRE(((region[a][src(edge)] && region[b][dst(edge)]) ||
                     (region[b][src(edge)] && region[a][dst(edge)])));
            for (index_t leaf = 0; leaf < n; ++leaf) {
                REQUIRE_FALSE((region[a][leaf] && region[b][leaf]));
                region[v][leaf] = region[a][leaf] || region[b][leaf];
            }
        }
        for (bool contains: region.back()) {
            REQUIRE(contains);
        }
        return relaxed;
    }

    // Independent small-graph oracle: transitive closure of the original graph
    // at each distinct input threshold, both before and after the whole tie.
    // It uses neither an MST nor the construction's component/expansion helpers.
    template<typename weights_t, typename output_t>
    void check_thresholds(const ugraph &graph, const weights_t &weights, const output_t &output) {
        const index_t n = num_vertices(graph);
        auto src = sources(graph), dst = targets(graph);
        std::vector<typename weights_t::value_type> levels(weights.begin(), weights.end());
        std::sort(levels.begin(), levels.end());
        levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
        for (auto threshold: levels) {
            for (bool inclusive: {false, true}) {
                auto enabled = [&](const auto weight) {
                    return inclusive ? !(threshold < weight) : weight < threshold;
                };
                std::vector<std::vector<bool>> reached(n, std::vector<bool>(n));
                for (index_t v = 0; v < n; ++v) {
                    reached[v][v] = true;
                }
                for (size_t e = 0; e < weights.size(); ++e) {
                    if (enabled(weights(e))) {
                        reached[src(e)][dst(e)] = reached[dst(e)][src(e)] = true;
                    }
                }
                for (index_t k = 0; k < n; ++k) {
                    for (index_t a = 0; a < n; ++a) {
                        for (index_t b = 0; b < n; ++b) {
                            reached[a][b] = reached[a][b] || (reached[a][k] && reached[k][b]);
                        }
                    }
                }
                std::vector<index_t> region(n);
                for (index_t v = 0; v < n; ++v) {
                    auto node = v, parent = output.first(node);
                    while (parent != node && enabled(weights(output.second(parent - n)))) {
                        node = parent;
                        parent = output.first(node);
                    }
                    region[v] = node;
                }
                for (index_t a = 0; a < n; ++a) {
                    for (index_t b = 0; b < n; ++b) {
                        REQUIRE((region[a] == region[b]) == reached[a][b]);
                    }
                }
            }
        }
    }

    template<typename weights_t>
    void check_tie_refinements(const ugraph &graph, const weights_t &weights) {
        for (bool descending: {false, true}) {
            auto less = [&](index_t a, index_t b) {
                return weights(a) < weights(b) || (!(weights(b) < weights(a)) && (descending ? a > b : a < b));
            };
            auto relaxed = check_numberings(sources(graph), targets(graph), num_vertices(graph), less);
            check_thresholds(graph, weights, relaxed);
        }
    }

    template<typename graph_t, typename weights_t>
    void check_exact(const graph_t &graph, const weights_t &weights) {
        auto result = bpt_canonical_boruvka(graph, weights);
        const auto reference = bpt_canonical(graph, weights);
        static_assert(std::is_same<typename decltype(result.altitudes)::value_type,
                                   typename weights_t::value_type>::value, "Preserve weight type.");
        REQUIRE(result.tree.parents() == reference.tree.parents());
        REQUIRE(result.mst_edge_map == reference.mst_edge_map);
        REQUIRE(result.altitudes == reference.altitudes);
        auto less = [&](index_t a, index_t b) {
            return internal::bpt_boruvka_scalar_less(weights, a, b);
        };
        check_numberings(sources(graph), targets(graph), num_vertices(graph), less);
        const index_t n = num_vertices(graph), count = n + n - 1;
        REQUIRE(num_vertices(result.tree) == static_cast<size_t>(count));
        REQUIRE(num_leaves(result.tree) == static_cast<size_t>(n));
        REQUIRE(category(result.tree) == tree_category::partition_tree);
        REQUIRE(result.mst_edge_map.size() == static_cast<size_t>(n - 1));
        REQUIRE(result.tree.parents()(count - 1) == count - 1);

        // Independent binary/tree and building-edge region checks, using only
        // the output parents and original endpoints (no Kruskal state).
        std::vector<std::vector<index_t>> children(count);
        std::vector<std::vector<bool>> region(count, std::vector<bool>(n));
        for (index_t v = 0; v + 1 < count; ++v) {
            auto p = result.tree.parents()(v);
            REQUIRE(p > v);
            REQUIRE(p < count);
            children[p].push_back(v);
        }
        std::vector<bool> seen(num_edges(graph));
        auto src = sources(graph), dst = targets(graph);
        for (index_t v = 0; v < n; ++v) {
            REQUIRE(children[v].empty());
            REQUIRE(result.altitudes(v) == 0);
            region[v][v] = true;
        }
        for (index_t v = n; v < count; ++v) {
            REQUIRE(children[v].size() == 2);
            const auto a = children[v][0], b = children[v][1];
            auto edge = result.mst_edge_map(v - n);
            REQUIRE(edge >= 0);
            REQUIRE(edge < (index_t) num_edges(graph));
            REQUIRE_FALSE(seen[edge]);
            seen[edge] = true;
            REQUIRE(((region[a][src(edge)] && region[b][dst(edge)]) ||
                     (region[b][src(edge)] && region[a][dst(edge)])));
            for (index_t leaf = 0; leaf < n; ++leaf) {
                REQUIRE_FALSE((region[a][leaf] && region[b][leaf]));
                region[v][leaf] = region[a][leaf] || region[b][leaf];
            }
        }
        for (bool contains: region.back()) {
            REQUIRE(contains);
        }
    }

    auto path(index_t n) {
        ugraph graph(n);
        for (index_t i = 1; i < n; ++i) {
            add_edge(i - 1, i, graph);
        }
        return graph;
    }

    TEST_CASE("Boruvka basin numbering can have unsorted internal altitudes",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_policies]") {
        auto graph = path(5);
        array_1d<int> weights{4, 3, 10, 1};
        auto less = [&](index_t a, index_t b) {
            return internal::bpt_boruvka_scalar_less(weights, a, b);
        };
        auto relaxed = check_numberings(sources(graph), targets(graph), 5, less);
        REQUIRE(relaxed.second == array_1d<index_t>{1, 0, 3, 2});
        array_1d<int> internal_altitudes = xt::index_view(weights, relaxed.second);
        REQUIRE(internal_altitudes == array_1d<int>{3, 4, 1, 10});
        REQUIRE_FALSE(std::is_sorted(internal_altitudes.begin(), internal_altitudes.end()));
        auto exact = bpt_canonical_boruvka(graph, weights);
        REQUIRE(exact.mst_edge_map == array_1d<index_t>{3, 1, 0, 2});
        // Distinct weights make both tie refinements identical, but still
        // permit a numbering change between independent branches.
        auto reverse = [&](index_t a, index_t b) {
            return weights(a) < weights(b) || (!(weights(b) < weights(a)) && a > b);
        };
        auto other = check_numberings(sources(graph), targets(graph), 5, reverse);
        REQUIRE(other.first == relaxed.first);
        REQUIRE(other.second == relaxed.second);
        check_tie_refinements(graph, weights);
    }

    TEST_CASE("Boruvka controlled ties preserve threshold partitions",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_policies]") {
        ugraph star(9), cycle(4);
        for (index_t v = 1; v < 9; ++v) {
            add_edge(0, v, star);
        }
        check_tie_refinements(star, array_1d<int>{0, 0, 0, 0, 0, 0, 0, 0});
        for (index_t v = 0; v < 4; ++v) {
            add_edge(v, (v + 1) % 4, cycle);
        }
        array_1d<int> equal{0, 0, 0, 0};
        check_tie_refinements(cycle, equal);
        auto forward = [&](index_t a, index_t b) { return internal::bpt_boruvka_scalar_less(equal, a, b); };
        auto reverse = [&](index_t a, index_t b) {
            return a > b; // An explicitly supplied total order remains authoritative.
        };
        auto a = check_numberings(sources(cycle), targets(cycle), 4, forward);
        auto b = check_numberings(sources(cycle), targets(cycle), 4, reverse);
        REQUIRE(a.second == array_1d<index_t>{0, 1, 2});
        REQUIRE(b.second == array_1d<index_t>{3, 2, 1});
        REQUIRE_FALSE(a.second == b.second);

        auto deep = path(17);
        check_tie_refinements(deep, array_1d<int>{100, 1, 20, 2, 90, 3, 10, 4, 80, 5, 30, 6, 70, 7, 40, 8});
        check_tie_refinements(deep, array_1d<int>{9, 1, 3, 1, 8, 1, 2, 1, 7, 1, 4, 1, 6, 1, 5, 1});
        add_edge(0, 0, cycle);
        add_edge(0, 1, cycle);
        const auto inf = std::numeric_limits<double>::infinity();
        check_tie_refinements(cycle, array_1d<double>{inf, -inf, -0., 0., -inf, inf});

        std::mt19937 random(0x02B0A017u);
        for (index_t trial = 0; trial < 60; ++trial) {
            const index_t n = 2 + random() % 7;
            auto graph = path(n);
            for (index_t e = 0; e < 2 * n; ++e) {
                add_edge(random() % n, random() % n, graph);
            }
            array_1d<int> weights = xt::empty<int>({num_edges(graph)});
            for (auto &weight: weights) {
                weight = (int)(random() % 5) - 2;
            }
            INFO("threshold oracle trial=" << trial);
            check_tie_refinements(graph, weights);
        }
    }

    TEST_CASE("Boruvka output policies honor explicit ranks and vector comparators",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_policies]") {
        array_1d<int> src{0, 1, 2, 3}, dst{1, 2, 3, 0};
        array_1d<index_t> order{0, 1, 2, 3}, rank = xt::empty<index_t>({4});
        // All 24 supplied orders are authoritative, even if scalar weights or
        // another tie rule would prefer a different MST. No scalar comparator
        // is consulted by the endpoint/rank entry point in either numbering.
        do {
            for (index_t i = 0; i < 4; ++i) {
                rank(order(i)) = i;
            }
            auto less = [&](index_t a, index_t b) { return rank(a) < rank(b); };
            check_numberings(src, dst, 4, less);
        } while (std::next_permutation(order.begin(), order.end()));
        array_2d<int> vectors{{1, 0}, {0, 4}, {1, -1}, {0, 4}};
        for (bool reverse: {false, true}) {
            auto less = [&](index_t a, index_t b) {
                for (index_t k = 0; k < 2; ++k) {
                    if (vectors(a, k) != vectors(b, k)) {
                        return vectors(a, k) < vectors(b, k);
                    }
                }
                return reverse ? a > b : a < b;
            };
            check_numberings(src, dst, 4, less);
        }
    }

    TEST_CASE("Boruvka output policy validation and owning arrays",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_policies]") {
        auto less = [](index_t a, index_t b) { return a < b; };
        for (auto numbering: {internal::bpt_boruvka_numbering::exact,
                              internal::bpt_boruvka_numbering::basin}) {
            auto run = [&](const auto &src, const auto &dst, index_t n) {
                return internal::bpt_boruvka_from_edge_order(src, dst, n, less, numbering);
            };
            REQUIRE_THROWS(run(array_1d<int>{-1}, array_1d<int>{0}, 2));
            REQUIRE_THROWS(run(array_1d<int>{0}, array_1d<int>{2}, 2));
            REQUIRE_THROWS(run(array_1d<int>{0}, array_1d<int>{}, 2));
            REQUIRE_THROWS(run(array_2d<int>{{0}}, array_2d<int>{{1}}, 2));
            REQUIRE_THROWS(run(array_1d<int>{}, array_1d<int>{}, 0));
            REQUIRE_THROWS(run(array_1d<int>{0}, array_1d<int>{1}, 3));
            auto singleton = run(array_1d<int>{0}, array_1d<int>{0}, 1);
            REQUIRE(singleton.first == array_1d<index_t>{0});
            REQUIRE(singleton.second.size() == 0);
        }
        auto owned = [] {
            array_1d<int> src{0, 1}, dst{1, 2};
            return internal::bpt_boruvka_from_edge_order(src, dst, 3,
                                                        [](index_t a, index_t b) { return a > b; },
                                                        internal::bpt_boruvka_numbering::basin);
        }();
        REQUIRE(owned.first == array_1d<index_t>{4, 3, 3, 4, 4});
        REQUIRE(owned.second == array_1d<index_t>{1, 0});
    }

    TEST_CASE("Boruvka fixed Kruskal references", "[hierarchy_core][bpt_boruvka]") {
        auto graph = get_4_adjacency_graph({1, 2});
        check_exact(graph, array_1d<double>{2});
        auto trivial = bpt_canonical_boruvka(graph, array_1d<double>{2});
        REQUIRE(trivial.tree.parents() == array_1d<index_t>{2, 2, 2});
        graph = get_4_adjacency_graph({2, 3});
        array_1d<double> weights{1, 0, 2, 1, 1, 1, 2};
        check_exact(graph, weights);
        auto grid = bpt_canonical_boruvka(graph, weights);
        REQUIRE(grid.tree.parents() == array_1d<index_t>{6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10});
        REQUIRE(grid.mst_edge_map == array_1d<index_t>{1, 0, 3, 4, 2});
        check_exact(get_8_adjacency_graph({3, 4}), xt::eval(xt::zeros<int>({29})));
    }

    TEST_CASE("Boruvka paper Figures 3 and 4", "[hierarchy_core][bpt_boruvka]") {
        // Figure 3: row-major a..l on a 3x4 grid, edges e0..e16.
        ugraph graph(12);
        for (auto edge: std::vector<std::pair<index_t, index_t>>{
                {0, 1}, {0, 4}, {1, 2}, {1, 5}, {2, 3}, {2, 6}, {3, 7},
                {4, 5}, {4, 8}, {5, 6}, {5, 9}, {6, 7}, {6, 10}, {7, 11},
                {8, 9}, {9, 10}, {10, 11}}) {
            add_edge(edge.first, edge.second, graph);
        }
        array_1d<int> weights{6, 10, 18, 1, 4, 13, 14, 16, 12, 15, 9, 2, 7, 8, 3, 17, 11};
        check_exact(graph, weights);
        auto result = bpt_canonical_boruvka(graph, weights);
        REQUIRE(result.mst_edge_map == array_1d<index_t>{3, 11, 14, 4, 0, 12, 13, 10, 1, 5, 9});
        // Figure 4's final edge parents, with leaves on their initial saddles.
        REQUIRE(result.tree.parents() == array_1d<index_t>{
                16, 12, 15, 15, 20, 12, 13, 13, 14, 14, 17, 18,
                16, 17, 19, 21, 19, 18, 21, 20, 22, 22, 22});
    }

    TEST_CASE("Boruvka nonmonotone and terminal basins", "[hierarchy_core][bpt_boruvka]") {
        auto graph = path(5);
        array_1d<int> weights{10, 1, 9, 2};
        check_exact(graph, weights);
        auto result = bpt_canonical_boruvka(graph, weights);
        REQUIRE(result.tree.parents() == array_1d<index_t>{8, 5, 5, 6, 6, 7, 7, 8, 8});
        REQUIRE(result.mst_edge_map == array_1d<index_t>{1, 3, 2, 0});
        REQUIRE(result.altitudes == array_1d<int>{0, 0, 0, 0, 0, 1, 2, 9, 10});

        // Edge 100 crosses multiple raw ancestors; this checks the fixed-start
        // comparison and simultaneous basin relocation, not in-place compression.
        auto deep = path(17);
        array_1d<int> keys{100, 1, 20, 2, 90, 3, 10, 4, 80, 5, 30, 6, 70, 7, 40, 8};
        check_exact(deep, keys);
        auto less = [&](index_t a, index_t b) { return keys(a) < keys(b); };
        auto history = internal::bpt_boruvka_waterfall(sources(deep), targets(deep), 17, less);
        auto snapshot_parent = history.raw_parent, snapshot_basin = history.raw_basin;
        auto expanded = internal::bpt_boruvka_expand(history, less);
        index_t max_climb = 0;
        for (index_t s = 0; s < 16; ++s) {
            index_t node = s, climb = 0;
            while (history.raw_parent(node) != invalid_index &&
                   less(history.selected_edges(history.raw_parent(node)), history.selected_edges(s))) {
                node = history.raw_parent(node);
                ++climb;
            }
            max_climb = (std::max)(max_climb, climb);
            REQUIRE(expanded.first(s) == history.raw_parent(node));
            REQUIRE(expanded.second(s) == history.raw_basin(node));
        }
        REQUIRE(max_climb >= 2);
        REQUIRE(history.raw_parent == snapshot_parent);
        REQUIRE(history.raw_basin == snapshot_basin);

        ugraph star(30);
        for (index_t v = 1; v < 30; ++v) {
            add_edge(0, v, star);
        }
        check_exact(star, xt::eval(xt::arange<int>(29)));
        auto balanced = path(64);
        array_1d<int> levels = xt::empty<int>({63});
        for (index_t i = 1; i < 64; ++i) {
            levels(i - 1) = 0;
            for (auto j = i; j % 2 == 0; j /= 2) {
                ++levels(i - 1);
            }
        }
        check_exact(balanced, levels);
    }

    TEST_CASE("Boruvka newly internal undecided cycle edge", "[hierarchy_core][bpt_boruvka]") {
        ugraph graph(5);
        for (auto edge: std::vector<std::pair<index_t, index_t>>{{0, 1}, {1, 2}, {0, 2}, {2, 3}, {3, 4}}) {
            add_edge(edge.first, edge.second, graph);
        }
        // The triangle's unselected edge of weight 3 must be dropped before
        // computing the next saddle (bridge 10).
        check_exact(graph, array_1d<int>{1, 2, 3, 10, 1});
        add_edge(0, 1, graph);
        add_edge(2, 2, graph);
        check_exact(graph, array_1d<int>{1, 2, 3, 10, 1, 1, -100});
        check_exact(graph, array_1d<int>{0, 0, 0, 0, 0, 0, 0});
    }

    TEMPLATE_TEST_CASE("Boruvka numeric scalar types and extrema", "[hierarchy_core][bpt_boruvka]",
                       int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double) {
        using value_t = TestType;
        auto graph = path(7);
        const auto low = (std::numeric_limits<value_t>::lowest)();
        const auto high = (std::numeric_limits<value_t>::max)();
        array_1d<value_t> weights{high, low, high, value_t(0), low, value_t(1)};
        check_exact(graph, weights);
        // Adjacent uint64/int64 keys above double's exact integer range.
        weights = {high, value_t(high - 1), value_t(high - 2), low, value_t(1), value_t(1)};
        check_exact(graph, weights);
        if constexpr (std::is_floating_point<value_t>::value) {
            const auto inf = std::numeric_limits<value_t>::infinity();
            weights = {inf, -inf, inf, -value_t(0), value_t(0), -inf};
            check_exact(graph, weights);
            auto result = bpt_canonical_boruvka(graph, weights);
            REQUIRE(std::signbit(result.altitudes(9)));
            REQUIRE_FALSE(std::signbit(result.altitudes(10)));
            weights(0) = std::numeric_limits<value_t>::quiet_NaN();
            REQUIRE_THROWS(bpt_canonical_boruvka(graph, weights));
        }
    }

    TEST_CASE("Boruvka singleton validation and owning results", "[hierarchy_core][bpt_boruvka]") {
        ugraph singleton(1);
        check_exact(singleton, array_1d<int>{});
        add_edge(0, 0, singleton);
        check_exact(singleton, array_1d<double>{std::numeric_limits<double>::infinity()});
        REQUIRE_THROWS(bpt_canonical_boruvka(singleton, array_1d<double>{std::numeric_limits<double>::quiet_NaN()}));
        auto result = [] {
            return bpt_canonical_boruvka(path(3), array_1d<int>{2, 1});
        }();
        REQUIRE(result.tree.parents() == array_1d<index_t>{4, 3, 3, 4, 4});
        REQUIRE(result.altitudes == array_1d<int>{0, 0, 0, 1, 2});
        REQUIRE(result.mst_edge_map == array_1d<index_t>{1, 0});
        REQUIRE_THROWS(bpt_canonical_boruvka(ugraph(0), array_1d<int>{}));
        REQUIRE_THROWS(bpt_canonical_boruvka(ugraph(2), array_1d<int>{}));
        ugraph disconnected(3);
        add_edge(0, 1, disconnected);
        REQUIRE_THROWS(bpt_canonical_boruvka(disconnected, array_1d<int>{1}));
        REQUIRE_THROWS(bpt_canonical_boruvka(path(2), array_1d<int>{}));
        REQUIRE_THROWS(bpt_canonical_boruvka(path(2), array_2d<int>{{1}}));
        REQUIRE_THROWS(bpt_canonical_boruvka(path(2), array_nd<int>(1)));
        auto deleted = path(3);
        remove_edge(0, deleted);
        REQUIRE_THROWS(bpt_canonical_boruvka(deleted, array_1d<int>{1, 2}));
    }

    TEST_CASE("Boruvka endpoint core validation and total comparator boundary", "[hierarchy_core][bpt_boruvka]") {
        auto less = [](index_t a, index_t b) { return a < b; };
        auto run = [&](const auto &src, const auto &dst, index_t n) {
            return internal::bpt_boruvka_from_edge_order(src, dst, n, less);
        };
        REQUIRE_THROWS(run(array_1d<int>{-1}, array_1d<int>{0}, 2));
        REQUIRE_THROWS(run(array_1d<int>{0}, array_1d<int>{2}, 2));
        REQUIRE_THROWS(run(array_1d<uint64_t>{UINT64_MAX}, array_1d<int>{0}, 2));
        REQUIRE_THROWS(run(array_1d<int>{0}, array_1d<int>{}, 2));
        REQUIRE_THROWS(run(array_2d<int>{{0}}, array_2d<int>{{1}}, 2));
        REQUIRE_THROWS(run(array_1d<int>{}, array_1d<int>{}, -1));
        REQUIRE_THROWS(run(array_1d<int>{}, array_1d<int>{}, 0));
        REQUIRE_THROWS(run(array_1d<int>{}, array_1d<int>{}, INT64_MAX));
        REQUIRE_THROWS(internal::bpt_boruvka_check_sizes(1, (std::numeric_limits<size_t>::max)()));
        REQUIRE_THROWS(internal::bpt_boruvka_check_sizes(INT64_MAX / 2 + 1, 0));

        // Reverse endpoints, strided inputs, and a non-weight permutation.
        array_1d<uint64_t> src_storage{1, 99, 2, 99, 3, 99, 3, 99};
        array_1d<int> dst{0, 1, 2, 0};
        auto src = xt::view(src_storage, xt::range(0, 8, 2));
        array_1d<index_t> order{2, 0, 3, 1}, rank{1, 3, 0, 2};
        auto rank_less = [&](index_t a, index_t b) { return rank(a) < rank(b); };
        auto result = internal::bpt_boruvka_from_edge_order(src, dst, 4, rank_less);
        auto ref = internal::bpt_canonical_from_sorted_edges(src, dst, order, 4);
        REQUIRE(result.first == ref.first);
        REQUIRE(result.second == ref.second);
        check_numberings(src, dst, 4, rank_less);
        REQUIRE(src_storage == array_1d<uint64_t>{1, 99, 2, 99, 3, 99, 3, 99});
        array_2d<int> vectors{{1, 0}, {0, 4}, {1, -1}, {0, 4}};
        auto vector_less = [&](index_t a, index_t b) {
            for (index_t k = 0; k < 2; ++k) {
                if (vectors(a, k) != vectors(b, k)) {
                    return vectors(a, k) < vectors(b, k);
                }
            }
            return a < b;
        };
        array_1d<index_t> vector_order{1, 3, 2, 0};
        result = internal::bpt_boruvka_from_edge_order(src, dst, 4, vector_less);
        ref = internal::bpt_canonical_from_sorted_edges(src, dst, vector_order, 4);
        REQUIRE(result.first == ref.first);
        REQUIRE(result.second == ref.second);
        check_numberings(src, dst, 4, vector_less);
    }

    bool connected(const ugraph &graph) {
        std::vector<bool> reached(num_vertices(graph));
        reached[0] = true;
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto edge: edge_iterator(graph)) {
                auto a = source(edge, graph), b = target(edge, graph);
                if (reached[a] != reached[b]) {
                    reached[a] = reached[b] = true;
                    changed = true;
                }
            }
        }
        return std::all_of(reached.begin(), reached.end(), [](bool b) { return b; });
    }

    TEST_CASE("Boruvka exhaustive simple graphs through four vertices", "[hierarchy_core][bpt_boruvka]") {
        // Each possible edge is absent, weight 0, or weight 1: 3^6 cases
        // for n=4, covering every simple topology and all binary ties.
        for (index_t n = 2; n <= 4; ++n) {
            index_t cases = 1;
            for (index_t e = 0; e < n * (n - 1) / 2; ++e) {
                cases *= 3;
            }
            for (index_t code = 0; code < cases; ++code) {
                ugraph graph(n);
                std::vector<int> values;
                auto state = code;
                for (index_t a = 0; a < n; ++a) {
                    for (index_t b = a + 1; b < n; ++b) {
                        auto digit = state % 3;
                        state /= 3;
                        if (digit) {
                            add_edge(a, b, graph);
                            values.push_back(digit - 1);
                        }
                    }
                }
                INFO("n=" << n << " code=" << code);
                array_1d<int> weights = xt::adapt(values);
                if (connected(graph)) {
                    check_exact(graph, weights);
                    check_tie_refinements(graph, weights);
                } else {
                    REQUIRE_THROWS(bpt_canonical_boruvka(graph, weights));
                }
            }
        }
    }

    TEST_CASE("Boruvka seeded multigraphs and permuted edge storage", "[hierarchy_core][bpt_boruvka]") {
        std::mt19937 random(0xB0A017u);
        for (index_t trial = 0; trial < 160; ++trial) {
            const index_t n = 2 + random() % 39;
            std::vector<std::pair<index_t, index_t>> edges;
            for (index_t v = 1; v < n; ++v) {
                edges.emplace_back(v, random() % v);
            }
            for (index_t e = 0; e < n * (trial % 6); ++e) {
                edges.emplace_back(random() % n, random() % n);
            }
            std::shuffle(edges.begin(), edges.end(), random);
            ugraph graph(n);
            array_1d<int64_t> storage = xt::empty<int64_t>({edges.size() * 2});
            for (size_t e = 0; e < edges.size(); ++e) {
                add_edge(edges[e].first, edges[e].second, graph);
                storage(2 * e) = (trial % 2 == 0) ? (int64_t)(random() % 4) - 2 : (int64_t)random();
                storage(2 * e + 1) = INT64_MAX;
            }
            auto weights = xt::view(storage, xt::range(0, storage.size(), 2));
            auto copy = storage;
            INFO("trial=" << trial << " n=" << n);
            check_exact(graph, weights);
            REQUIRE(storage == copy);
        }
    }
    // Independent serial forest oracle. No union-find instance is shared with
    // the implementation or accessed concurrently, and assertions stay here.
    template<typename weights_t>
    void check_successor_rounds(const ugraph &graph, const weights_t &weights) {
        const index_t n = num_vertices(graph), m = num_edges(graph);
        auto src = sources(graph), dst = targets(graph);
        array_1d<index_t> labels = xt::arange<index_t>(n);
        auto count = n;
        while (count > 1) {
            auto saddle = internal::bpt_boruvka_invalid_indices(count);
            for (index_t e = 0; e < m; ++e) {
                const auto a = labels(src(e)), b = labels(dst(e));
                if (a == b) {
                    continue;
                }
                for (auto c: {a, b}) {
                    if (saddle(c) == invalid_index || internal::bpt_boruvka_scalar_less(weights, e, saddle(c))) {
                        saddle(c) = e;
                    }
                }
            }
            union_find forest(count);
            for (auto e: saddle) {
                REQUIRE(e != invalid_index);
                auto a = forest.find(labels(src(e))), b = forest.find(labels(dst(e)));
                if (a != b) {
                    forest.link(a, b);
                }
            }
            auto labeling = internal::bpt_boruvka_label_components(src, dst, labels, saddle);
            REQUIRE(labeling.second <= count / 2);
            auto forest_to_new = internal::bpt_boruvka_invalid_indices(count);
            auto new_to_forest = internal::bpt_boruvka_invalid_indices(labeling.second);
            for (index_t c = 0; c < count; ++c) {
                auto root = forest.find(c), label = labeling.first(c);
                REQUIRE(label >= 0);
                REQUIRE(label < labeling.second);
                if (forest_to_new(root) == invalid_index) {
                    forest_to_new(root) = label;
                }
                if (new_to_forest(label) == invalid_index) {
                    new_to_forest(label) = root;
                }
                REQUIRE(forest_to_new(root) == label);
                REQUIRE(new_to_forest(label) == root);
            }
            for (auto root: new_to_forest) {
                REQUIRE(root != invalid_index);
            }
            for (auto &label: labels) {
                label = labeling.first(label);
            }
            count = labeling.second;
        }
    }

    TEST_CASE("Boruvka block compaction reserves disjoint stable ranges",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_parallel]") {
        for (index_t size: {0, 1, 1023, 1024, 1025, 4099, 66000}) {
            internal::bpt_boruvka_scratch scratch(2, size);
            std::vector<index_t> active(size), retained(size), selected(size), ref_keep, ref_select;
            std::iota(active.begin(), active.end(), 0);
            auto classify = [](index_t e) -> uint8_t { return (e * 7 + e / 5) % 3; };
            for (auto e: active) {
                if (classify(e) == 1) {
                    ref_keep.push_back(e);
                } else if (classify(e) == 2) {
                    ref_select.push_back(e);
                }
            }
            auto counts = internal::bpt_boruvka_compact(active, scratch, size, classify,
                [&](index_t e, uint8_t kind, index_t destination) {
                    (kind == 1 ? retained : selected)[destination] = e;
                });
            retained.resize(counts.retained);
            selected.resize(counts.selected);
            REQUIRE(retained == ref_keep);
            REQUIRE(selected == ref_select);

        }
    }

    TEST_CASE("Boruvka successor labeling matches serial forest connectivity",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_parallel]") {
        const index_t n = 4096;
        auto chain = path(n);
        check_successor_rounds(chain, xt::eval(xt::arange<index_t>(n - 1)));
        array_1d<int> levels = xt::empty<int>({n - 1});
        for (index_t v = 1; v < n; ++v) {
            levels(v - 1) = 0;
            for (auto j = v; j % 2 == 0; j /= 2) {
                ++levels(v - 1);
            }
        }
        check_successor_rounds(chain, levels);
        ugraph star(n);
        for (index_t v = 1; v < n; ++v) {
            add_edge(0, v, star);
        }
        check_successor_rounds(star, xt::eval(xt::zeros<int>({n - 1})));
        std::mt19937 random(0x03B0A017u);
        auto irregular = path(1024);
        for (index_t e = 0; e < 4096; ++e) {
            add_edge(random() % 1024, random() % 1024, irregular);
        }
        array_1d<int> weights = xt::empty<int>({num_edges(irregular)});
        for (auto &w: weights) {
            w = random() % 13;
        }
        check_successor_rounds(irregular, weights);
        // Nontrivial old labels: three original vertices per old component,
        // permuted so the mutual pair's smaller endpoint is not its first leaf.
        array_1d<index_t> labels{3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2};
        array_1d<index_t> src{2, 1, 3}, dst{1, 3, 0}, saddle{0, 1, 1, 2};
        auto result = internal::bpt_boruvka_label_components(src, dst, labels, saddle);
        REQUIRE(result.second == 1);
        REQUIRE(result.first == array_1d<index_t>{0, 0, 0, 0});
        // Each pair roots at its smaller endpoint; dense IDs scan these roots
        // in order, even when the pairs' endpoints interleave.
        labels = xt::arange<index_t>(6);
        src = {0, 1, 2};
        dst = {5, 4, 3};
        saddle = {0, 1, 2, 2, 1, 0};
        result = internal::bpt_boruvka_label_components(src, dst, labels, saddle);
        REQUIRE(result.second == 3);
        REQUIRE(result.first == array_1d<index_t>{0, 1, 2, 2, 1, 0});
    }

    // Linear-space validation for graphs large enough to exercise block scans,
    // parallel sorting and long jump chains without the small oracle's n^2 data.
    template<typename output_t, typename reference_t>
    void check_large_relaxed(const output_t &output, const reference_t &reference, index_t n, index_t m) {
        auto edge_to_exact = internal::bpt_boruvka_invalid_indices(m);
        for (index_t i = 0; i < n - 1; ++i) {
            edge_to_exact(reference.mst_edge_map(i)) = n + i;
        }
        array_1d<index_t> to_exact = xt::arange<index_t>(2 * n - 1);
        for (index_t i = 0; i < n - 1; ++i) {
            auto e = output.second(i);
            REQUIRE(e >= 0);
            REQUIRE(e < m);
            REQUIRE(edge_to_exact(e) != invalid_index);
            to_exact(n + i) = edge_to_exact(e);
        }
        std::vector<index_t> children(2 * n - 1, 0);
        for (index_t v = 0; v < 2 * n - 1; ++v) {
            auto p = output.first(v);
            REQUIRE(p >= 0);
            REQUIRE(p < 2 * n - 1);
            REQUIRE(to_exact(p) == reference.tree.parents()(to_exact(v)));
            if (v != 2 * n - 2) {
                REQUIRE(p > v);
                ++children[p];
            } else {
                REQUIRE(p == v);
            }
        }
        for (index_t v = 0; v < 2 * n - 1; ++v) {
            REQUIRE(children[v] == (v < n ? 0 : 2));
        }
    }

    TEST_CASE("Boruvka scratch survives shrinking rounds and alternating fresh calls",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_scratch]") {
        array_1d<index_t> saved_parents, saved_map, expected_parents, expected_map;
        for (index_t n: {1025, 1, 2, 65, 1025, 3}) {
            auto graph = path(n);
            for (index_t v = 0; v < n; ++v) {
                add_edge(v, v, graph);
                if (v + 1 < n) add_edge(v, v + 1, graph);
            }
            array_1d<int> weights = xt::empty<int>({num_edges(graph)});
            weights.fill(100);
            // Balanced path weights force several geometrically shrinking rounds;
            // loops and parallel edges exercise compaction tails across block boundaries.
            for (index_t e = 0; e < n - 1; ++e) {
                weights(e) = 0;
                for (auto v = e + 1; v % 2 == 0; v /= 2) ++weights(e);
            }
            auto reference = bpt_canonical(graph, weights);
            auto result = bpt_canonical_boruvka(graph, weights);
            REQUIRE(result.tree.parents() == reference.tree.parents());
            REQUIRE(result.mst_edge_map == reference.mst_edge_map);
            REQUIRE(result.altitudes == reference.altitudes);
            auto less = [&](index_t a, index_t b) { return internal::bpt_boruvka_scalar_less(weights, a, b); };
            auto relaxed = internal::bpt_boruvka_from_edge_order(sources(graph), targets(graph), n, less,
                                                               internal::bpt_boruvka_numbering::basin);
            check_large_relaxed(relaxed, reference, n, num_edges(graph));
            if (saved_parents.size()) {
                REQUIRE(saved_parents == expected_parents);
                REQUIRE(saved_map == expected_map);
            }
            saved_parents = std::move(relaxed.first);
            saved_map = std::move(relaxed.second);
            expected_parents = saved_parents;
            expected_map = saved_map;
            // An exceptional call must release partial scratch and leave subsequent
            // calls and previously returned owning outputs independent.
            ugraph disconnected(3);
            add_edge(0, 1, disconnected);
            REQUIRE_THROWS(bpt_canonical_boruvka(disconnected, array_1d<int>{0}));
        }
    }

    TEST_CASE("Boruvka caller controlled concurrency preserves exact and relaxed output",
              "[hierarchy_core][bpt_boruvka][bpt_boruvka_parallel]") {
        const index_t n = 4096;
        std::mt19937 random(0x03C0FFEEu);
        for (index_t family = 0; family < 5; ++family) {
            auto graph = family == 1 ? ugraph(n) : path(n);
            if (family == 1) {
                for (index_t v = 1; v < n; ++v) {
                    add_edge(0, v, graph);
                    add_edge(0, v, graph);
                }
            } else if (family == 4) {
                for (index_t e = 0; e < 5 * n; ++e) {
                    add_edge(random() % n, random() % n, graph);
                }
            }
            array_1d<double> weights = xt::empty<double>({num_edges(graph)});
            for (index_t e = 0; e < (index_t)num_edges(graph); ++e) {
                weights(e) = family == 0 ? double(e) : family == 1 ? 0. : double(random() % 9);
                if (family == 2) {
                    weights(e) = 0;
                    for (auto v = e + 1; v % 2 == 0; v /= 2) {
                        ++weights(e);
                    }
                } else if (family == 3) {
                    weights(e) = e % 3 == 0 ? std::numeric_limits<double>::infinity() :
                                 e % 3 == 1 ? -std::numeric_limits<double>::infinity() : 0.;
                }
            }
            // Both references are constructed under caller-owned concurrency 1.
#ifdef HG_USE_TBB
            auto serial_limit = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, 1);
#endif
            auto reference = bpt_canonical(graph, weights);
            auto baseline = bpt_canonical_boruvka(graph, weights);
#ifdef HG_USE_TBB
            serial_limit.reset();
#endif
            REQUIRE(baseline.tree.parents() == reference.tree.parents());
            REQUIRE(baseline.mst_edge_map == reference.mst_edge_map);
            for (int limit: {1, 2, 4, 8}) {
#ifdef HG_USE_TBB
                tbb::global_control control(tbb::global_control::max_allowed_parallelism, limit);
#endif
                for (int repeat = 0; repeat < 3; ++repeat) {
                    INFO("family=" << family << " limit=" << limit << " repeat=" << repeat);
                    auto result = bpt_canonical_boruvka(graph, weights);
                    REQUIRE(result.tree.parents() == reference.tree.parents());
                    REQUIRE(result.mst_edge_map == reference.mst_edge_map);
                    REQUIRE(result.altitudes == reference.altitudes);
                    auto less = [&](index_t a, index_t b) { return internal::bpt_boruvka_scalar_less(weights, a, b); };
                    auto history = internal::bpt_boruvka_waterfall(sources(graph), targets(graph), n, less);
                    check_traversal_history(history, sources(graph), targets(graph), n, less);
                    auto relaxed = internal::bpt_boruvka_export(history, less, internal::bpt_boruvka_numbering::basin);
                    check_large_relaxed(relaxed, reference, n, num_edges(graph));
                    // Alternate total tie order: compare edge-labeled trees to
                    // Kruskal under exactly that order at every tested limit.
                    auto reverse = [&](index_t a, index_t b) {
                        return weights(a) < weights(b) || (!(weights(b) < weights(a)) && a > b);
                    };
                    array_1d<index_t> order = xt::arange<index_t>(num_edges(graph));
                    std::sort(order.begin(), order.end(), reverse);
                    auto reversed_ref = internal::bpt_canonical_from_sorted_edges(sources(graph), targets(graph), order, n);
                    auto reversed = internal::bpt_boruvka_from_edge_order(sources(graph), targets(graph), n, reverse,
                                                                          internal::bpt_boruvka_numbering::basin);
                    auto active_reversed = internal::bpt_boruvka_from_edge_order(
                        sources(graph), targets(graph), n, reverse, internal::bpt_boruvka_numbering::basin,
                        internal::bpt_boruvka_traversal::active);
                    REQUIRE(active_reversed.first == reversed.first);
                    REQUIRE(active_reversed.second == reversed.second);
                    auto ref_tree = make_node_weighted_tree_and_mst(tree(std::move(reversed_ref.first)),
                                                                    array_1d<double>{}, std::move(reversed_ref.second));
                    check_large_relaxed(reversed, ref_tree, n, num_edges(graph));
                }
            }
        }
        // Independent threshold partition oracle also runs under every limit.
        auto small = path(9);
        add_edge(0, 8, small);
        for (int limit: {1, 2, 4, 8}) {
#ifdef HG_USE_TBB
            tbb::global_control control(tbb::global_control::max_allowed_parallelism, limit);
#endif
            (void)limit;
            check_tie_refinements(small, array_1d<int>{2, 0, 1, 0, 2, 0, 1, 0, 2});
        }
    }

}

TEST_CASE("Stable basin grouping preserves arbitrary global edge order", "[bpt_boruvka][hierarchy_core]") {
    using namespace hg;
    namespace core = hg::hierarchy_core_internal;
#ifdef HG_USE_TBB
    for (int cap: {1, 2, 8}) {
        tbb::global_control control(tbb::global_control::max_allowed_parallelism, cap);
#else
    {
#endif
        for (index_t count: {0, 1, 2, 255, 256, 257, 16385, 65537, 1048577}) {
            array_1d<index_t> global = xt::arange<index_t>(count);
            std::mt19937 rng(1729);
            std::shuffle(global.begin(), global.end(), rng);
            for (int pattern = 0; pattern < 5; ++pattern) {
                array_1d<index_t> basins = xt::empty<index_t>({count});
                for (index_t i = 0; i < count; ++i) {
                    basins(i) = pattern == 0 ? 0 : pattern == 1 ? i :
                        pattern == 2 ? (i % 11 == 0 ? count - 1 : 0) :
                        pattern == 3 ? (i % 3) * ((count - 1) / 3) : rng() % count;
                }
                auto expected = global;
                std::stable_sort(expected.begin(), expected.end(), [&](index_t a, index_t b) {
                    return basins(a) < basins(b);
                });
                // Independent serial counting reference includes empty buckets.
                std::vector<index_t> counts(count + 1, 0);
                for (auto slot: global) ++counts[basins(slot) + 1];
                std::partial_sum(counts.begin(), counts.end(), counts.begin());
                array_1d<index_t> serial = xt::empty<index_t>({count});
                for (auto slot: global) serial(counts[basins(slot)]++) = slot;
                REQUIRE(serial == expected);
                const auto saved = global;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    REQUIRE(core::bpt_boruvka_group_basins(global, basins) == expected);
                    REQUIRE(global == saved);
                }
            }
        }
    }
}

TEST_CASE("Saddle enumeration matches edge scans and history by original edge ID",
          "[bpt_boruvka][hierarchy_core][bpt_boruvka_selection]") {
    using namespace hg;
    namespace core = hg::hierarchy_core_internal;
    std::mt19937 rng(172903);
    bool saw_one_sided = false, saw_mutual = false, saw_internal = false, saw_terminal_leftover = false;
    for (int cap: {1, 2, 8}) {
#ifdef HG_USE_TBB
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, cap);
#endif
        for (int trial = 0; trial < 70; ++trial) {
            const index_t n = trial < 3 ? trial + 2 : 3 + rng() % 100;
            std::vector<std::pair<index_t, index_t>> edges;
            for (index_t v = 1; v < n; ++v) edges.emplace_back(trial % 3 == 0 ? 0 : v - 1, v);
            // Include bare trees (m=n-1), tied parallel edges, loops and cycles.
            if (trial % 4) for (index_t i = 0; i < 3 * n; ++i) edges.emplace_back(rng() % n, rng() % n);
            std::shuffle(edges.begin(), edges.end(), rng);
            const index_t m = edges.size();
            array_1d<index_t> src = xt::empty<index_t>({m});
            array_1d<index_t> dst = xt::empty<index_t>({m}), weights = xt::empty<index_t>({m});
            for (index_t e = 0; e < m; ++e) {
                src(e) = edges[e].first; dst(e) = edges[e].second;
                if (e % 2) std::swap(src(e), dst(e));
                weights(e) = rng() % 7;
            }
            auto less = [&](index_t a, index_t b) { return core::bpt_boruvka_scalar_less(weights, a, b); };
            auto got = core::bpt_boruvka_waterfall(src, dst, n, less);
            std::vector<index_t> slots(m, invalid_index);
            for (index_t s = 0; s < n - 1; ++s) {
                REQUIRE(slots[got.selected_edges(s)] == invalid_index);
                slots[got.selected_edges(s)] = s;
            }
            array_1d<index_t> labels = xt::arange<index_t>(n);
            index_t c = n, offset = 0, total = 0;
            while (c > 1) {
                // Serial original-edge oracle: no enumeration/compaction helper.
                auto saddle = core::bpt_boruvka_invalid_indices(c);
                for (index_t e = 0; e < m; ++e) if (labels(src(e)) != labels(dst(e))) {
                    for (auto a: {labels(src(e)), labels(dst(e))})
                        if (saddle(a) == invalid_index || less(e, saddle(a))) saddle(a) = e;
                }
                if (c == n) for (index_t v = 0; v < n; ++v)
                    REQUIRE(got.selected_edges(got.leaf_parent(v)) == saddle(v));
                std::vector<index_t> expected, emitted;
                for (index_t e = 0; e < m; ++e) {
                    auto a = labels(src(e)), b = labels(dst(e));
                    if (a != b && (saddle(a) == e || saddle(b) == e)) {
                        expected.push_back(e);
                        saw_mutual |= saddle(a) == e && saddle(b) == e;
                        saw_one_sided |= (saddle(a) == e) != (saddle(b) == e);
                    }
                }
                // Scatter slots rather than append from parallel tasks.
                emitted.resize(expected.size());
                core::bpt_boruvka_scratch scratch(n, m);
                auto count = core::bpt_boruvka_enumerate_saddles(src, dst, labels, saddle, scratch,
                    emitted.size(), [&](index_t e, index_t i) { emitted[i] = e; });
                REQUIRE(count == static_cast<index_t>(expected.size()));
                std::vector<index_t> component_order;
                for (index_t v = 0; v < c; ++v) {
                    const auto edge = saddle(v);
                    const auto a = labels(src(edge)), b = labels(dst(edge)), d = v == a ? b : a;
                    if (saddle(d) != edge || v < d) component_order.push_back(edge);
                }
                REQUIRE(emitted == component_order);
                auto repeated = emitted;
                REQUIRE(core::bpt_boruvka_enumerate_saddles(src, dst, labels, saddle, scratch,
                    emitted.size(), [&](index_t e, index_t i) { repeated[i] = e; }) == count);
                REQUIRE(repeated == emitted);
                std::sort(emitted.begin(), emitted.end());
                REQUIRE(emitted == expected);
                auto labeling = core::bpt_boruvka_label_components(src, dst, labels, saddle);
                REQUIRE(count == c - labeling.second);
                const auto old_labels = labels;
                for (auto &label: labels) label = labeling.first(label);
                c = labeling.second;
                auto next_saddle = core::bpt_boruvka_invalid_indices(c);
                for (index_t e = 0; e < m; ++e) {
                    auto a = labels(src(e)), b = labels(dst(e));
                    if (a != b) {
                        for (auto v: {a, b}) if (next_saddle(v) == invalid_index || less(e, next_saddle(v))) next_saddle(v) = e;
                    } else if (!std::binary_search(expected.begin(), expected.end(), e) &&
                               old_labels(src(e)) != old_labels(dst(e))) {
                        saw_internal |= c > 1;
                        saw_terminal_leftover |= c == 1;
                    }
                }
                for (auto edge: expected) {
                    const auto slot = slots[edge];
                    REQUIRE(slot != invalid_index);
                    REQUIRE(got.raw_basin(slot) == offset + labels(src(edge)));
                    const auto p = got.raw_parent(slot);
                    REQUIRE((p == invalid_index ? invalid_index : got.selected_edges(p)) ==
                            (c == 1 ? invalid_index : next_saddle(labels(src(edge)))));
                }
                offset += c; total += count;
            }
            REQUIRE(total == n - 1);
        }
    }
    REQUIRE(saw_one_sided); REQUIRE(saw_mutual); REQUIRE(saw_internal); REQUIRE(saw_terminal_leftover);
}

TEST_CASE("Compact history sparse IDs and error cleanup", "[bpt_boruvka][hierarchy_core]") {
    using namespace hg;
    namespace core = hg::hierarchy_core_internal;
    for (auto mode: {core::bpt_boruvka_traversal::full_scan, core::bpt_boruvka_traversal::active}) {
        array_1d<index_t> src = xt::zeros<index_t>({4099});
        array_1d<index_t> dst = src;
        dst(4096) = 1; src(4097) = 1; dst(4097) = 2; src(4098) = 2; dst(4098) = 3;
        array_1d<int> weights = xt::zeros<int>({4099});
        weights(4096) = 10; weights(4097) = 1; weights(4098) = 9;
        auto less = [&](index_t a, index_t b) { return core::bpt_boruvka_scalar_less(weights, a, b); };
        bpt_boruvka_tests::check_numberings(src, dst, 4, less);
        REQUIRE_THROWS(core::bpt_boruvka_from_edge_order(src, dst, 5, less, core::bpt_boruvka_numbering::exact, mode));
        REQUIRE_THROWS(core::bpt_boruvka_from_edge_order(src, dst, 0, less, core::bpt_boruvka_numbering::exact, mode));
        dst(0) = 4;
        REQUIRE_THROWS(core::bpt_boruvka_from_edge_order(src, dst, 4, less, core::bpt_boruvka_numbering::exact, mode));
        REQUIRE_THROWS(core::bpt_boruvka_from_edge_order(src, array_1d<index_t>{}, 4, less, core::bpt_boruvka_numbering::exact, mode));
        REQUIRE_THROWS(core::bpt_boruvka_from_edge_order(
            array_2d<index_t>{{0,1}}, array_2d<index_t>{{1,2}}, 3, less, core::bpt_boruvka_numbering::exact, mode));
        REQUIRE_THROWS(core::bpt_boruvka_from_edge_order(
            array_1d<index_t>{-1}, array_1d<index_t>{0}, 1, less, core::bpt_boruvka_numbering::exact, mode));
        auto owning = [mode] {
            array_1d<index_t> a{0,1}, b{1,2};
            return core::bpt_boruvka_from_edge_order(a,b,3,[](index_t x,index_t y){return x>y;}, core::bpt_boruvka_numbering::exact, mode);
        }();
        REQUIRE(owning.first == array_1d<index_t>{4,3,3,4,4});
    }
}

TEST_CASE("Boruvka traversal validation and native defaults", "[hierarchy_core][bpt_boruvka]") {
    using namespace hg;
    namespace c = hg::hierarchy_core_internal;
    using numbering = c::bpt_boruvka_numbering;
    using traversal = c::bpt_boruvka_traversal;
    array_1d<index_t> src = {0, 1, 0, 2}, dst = {1, 2, 2, 2};
    auto less = [](index_t a, index_t b) { return a < b; };
    auto full = c::bpt_boruvka_from_edge_order(src, dst, 3, less);
    auto active = c::bpt_boruvka_from_edge_order(src, dst, 3, less, numbering::exact, traversal::active);
    REQUIRE(full.first == active.first);
    REQUIRE(full.second == active.second);
    for (int i = 0; i < 16; ++i) {
        auto transient = c::bpt_boruvka_from_edge_order(src, dst, 3, less);
        transient.first.fill(77);
    }
    REQUIRE(full.first == active.first);
    REQUIRE_THROWS(c::bpt_boruvka_from_edge_order(src, dst, 3, less, static_cast<numbering>(9)));
    REQUIRE_THROWS(c::bpt_boruvka_from_edge_order(src, dst, 3, less, numbering::exact, static_cast<traversal>(9)));
    for (auto mode: {traversal::full_scan, traversal::active}) {
        REQUIRE_THROWS(c::bpt_boruvka_from_edge_order(src, dst, 4, less, numbering::exact, mode));
        REQUIRE_THROWS(c::bpt_boruvka_from_edge_order(src, dst, 0, less, numbering::exact, mode));
        // Disconnected components each have an initial saddle: failure must
        // still be detected in the next round, before successor indexing.
        REQUIRE_THROWS(c::bpt_boruvka_from_edge_order(array_1d<index_t>{0,2},
            array_1d<index_t>{1,3}, 4, less, numbering::exact, mode));
        auto singleton = c::bpt_boruvka_from_edge_order(array_1d<index_t>{},
            array_1d<index_t>{}, 1, less, numbering::exact, mode);
        REQUIRE(singleton.first == array_1d<index_t>{0});
        REQUIRE(singleton.second.size() == 0);
        dst(0) = 3;
        REQUIRE_THROWS(c::bpt_boruvka_from_edge_order(src, dst, 3, less, numbering::exact, mode));
        dst(0) = 1;
    }
    auto graph = bpt_boruvka_tests::path(7);
    array_1d<int> weights{8,1,9,2,5,3};
    auto default_result = bpt_canonical_boruvka(graph, weights);
    auto kruskal = bpt_canonical(graph, weights);
    REQUIRE(default_result.tree.parents() == kruskal.tree.parents());
    REQUIRE(default_result.mst_edge_map == kruskal.mst_edge_map);
    for (auto num: {numbering::exact, numbering::basin}) {
        auto a = bpt_canonical_boruvka(graph, weights, num, false);
        auto b = bpt_canonical_boruvka(graph, weights, num, true);
        REQUIRE(a.tree.parents() == b.tree.parents());
        REQUIRE(a.mst_edge_map == b.mst_edge_map);
        REQUIRE(a.altitudes == b.altitudes);
    }
}

TEST_CASE("Traversal phases freeze equal minima and monotone edge states", "[hierarchy_core][bpt_boruvka]") {
    using namespace hg;
    namespace c = hg::hierarchy_core_internal;
    std::mt19937 rng(1729);
    for (int cap: {1,2,8}) {
#ifdef HG_USE_TBB
        tbb::global_control control(tbb::global_control::max_allowed_parallelism, cap);
#endif
        for (int trial = 0; trial < 25; ++trial) {
            const index_t n = 7 + trial, m = 6 * n;
            array_1d<index_t> src = xt::empty<index_t>({m});
            array_1d<index_t> dst = xt::empty<index_t>({m}), weights = xt::empty<index_t>({m});
            for (index_t e = 0; e < m; ++e) {
                src(e) = e < n - 1 ? e : rng() % n;
                dst(e) = e < n - 1 ? e + 1 : rng() % n;
                weights(e) = rng() % 11;
            }
            auto less = [&](index_t a, index_t b) { return c::bpt_boruvka_scalar_less(weights,a,b); };
            c::bpt_boruvka_full_scan full(n,m);
            c::bpt_boruvka_active_scan active(n,m);
            REQUIRE(full.scratch.edge_class.size() == size_t(n));
            REQUIRE(active.scratch.edge_class.size() == size_t(m));
            array_1d<index_t> labels = xt::arange<index_t>(n);
            index_t count = n;
            std::vector<uint8_t> previous(m, c::bpt_boruvka_undecided);
            while (count > 1) {
                auto saddle = c::bpt_boruvka_minima(src,dst,labels,count,full,less);
                auto other = c::bpt_boruvka_minima(src,dst,labels,count,active,less);
                REQUIRE(saddle == other);
                auto oracle = c::bpt_boruvka_invalid_indices(count);
                size_t j = 0;
                for (index_t e = 0; e < m; ++e) {
                    auto a = labels(src(e)), b = labels(dst(e));
                    if (a != b) for (auto v: {a,b})
                        if (oracle(v) == invalid_index || less(e,oracle(v))) oracle(v) = e;
                    REQUIRE((previous[e] == c::bpt_boruvka_undecided || previous[e] == full.states[e]));
                    if (full.states[e] == c::bpt_boruvka_undecided) {
                        REQUIRE(a != b);
                        REQUIRE(j < active.active.size());
                        REQUIRE(active.active[j++] == e);
                    }
                }
                REQUIRE(j == active.active.size());
                REQUIRE(saddle == oracle);
                c::bpt_boruvka_enumerate_saddles(src,dst,labels,saddle,full.scratch,n-1,
                    [&](index_t e,index_t) { full.selected(e); });
                for (index_t e = 0; e < m; ++e) previous[e] = full.states[e];
                auto labeling = c::bpt_boruvka_label_components(src,dst,labels,saddle);
                for (auto &v: labels) v = labeling.first(v);
                count = labeling.second;
            }
        }
    }
}
