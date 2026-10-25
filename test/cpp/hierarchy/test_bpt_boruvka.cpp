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
#include "../test_utils.hpp"
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>

namespace bpt_boruvka_tests {
    using namespace hg;
    namespace internal = hg::hierarchy_core_internal;

    template<typename E1, typename E2, typename edge_less_t>
    auto check_numberings(const E1 &src, const E2 &dst, index_t n, const edge_less_t &less) {
        using numbering = internal::bpt_boruvka_numbering;
        auto history = internal::bpt_boruvka_waterfall(src, dst, n, less);
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
        for (auto ties: {internal::bpt_boruvka_tie_order::edge_id,
                         internal::bpt_boruvka_tie_order::reverse_edge_id}) {
            auto less = [&](index_t a, index_t b) {
                return internal::bpt_boruvka_scalar_less(weights, a, b, ties);
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
            return internal::bpt_boruvka_scalar_less(weights, a, b,
                                                    internal::bpt_boruvka_tie_order::reverse_edge_id);
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
            return internal::bpt_boruvka_scalar_less(equal, a, b,
                                                    internal::bpt_boruvka_tie_order::reverse_edge_id);
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
}
