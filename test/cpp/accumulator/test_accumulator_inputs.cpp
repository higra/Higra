/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/*
 * Accumulator entry points fed with xtensor containers, strided views and lazy expressions:
 * every kind of input must give the result obtained with the equivalent contiguous container.
 */

#include "../test_utils.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "higra/accumulator/graph_accumulator.hpp"
#include "higra/accumulator/at_accumulator.hpp"
#include "higra/accumulator/tree_contour_accumulator.hpp"
#include "higra/algo/rag.hpp"
#include "higra/structure/details/light_axis_view.hpp"
#include <functional>

namespace accumulator_inputs {

    using namespace hg;

    // node data (8, 3), its strided and lazy equivalents
    struct inputs {
        hg::tree t{xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7}};
        array_2d<double> nodes;
        array_2d<double> wide;      // (8, 6): nodes in the even columns
        array_2d<double> nodes_t;   // (3, 8): nodes transposed
        array_2d<double> leaves;
        array_1d<bool> condition{false, true, false, true, true, false, true, false};

        inputs() {
            nodes = xt::zeros<double>({8, 3});
            wide = xt::zeros<double>({8, 6});
            nodes_t = xt::zeros<double>({3, 8});
            for (index_t i = 0; i < 8; ++i) {
                for (index_t j = 0; j < 3; ++j) {
                    double v = (double) ((i * 7 + j * 3) % 11) - 2.5;
                    nodes(i, j) = v;
                    wide(i, 2 * j) = v;
                    wide(i, 2 * j + 1) = 1000;
                    nodes_t(j, i) = v;
                }
            }
            leaves = xt::view(nodes, xt::range(0, 5), xt::all());
        }

        auto strided_nodes() const {
            return xt::view(wide, xt::all(), xt::range(0, 6, 2));
        }

        auto transposed_nodes() const {
            return xt::transpose(nodes_t);
        }

        auto lazy_nodes() const {
            return nodes * 1.0;
        }

        auto strided_column() const {
            return xt::view(wide, xt::all(), 2);
        }

        array_1d<double> column() const {
            return xt::view(nodes, xt::all(), 1);
        }
    };

    template<typename F>
    void check_node_inputs(const inputs &d, F f) {
        auto ref = f(d.nodes);
        REQUIRE((f(d.strided_nodes()) == ref));
        REQUIRE((f(d.transposed_nodes()) == ref));
        REQUIRE((f(d.lazy_nodes()) == ref));

        auto ref1 = f(d.column());
        REQUIRE((f(d.strided_column()) == ref1));
        REQUIRE((f(d.column() + 0.0) == ref1));
    }

    TEST_CASE("tree accumulators with strided and lazy inputs", "[accumulator_inputs]") {
        inputs d;
        auto &t = d.t;

        for (bool children: {false, true}) {
            if (children) {
                t.compute_children();
            } else {
                t.clear_children();
            }
            check_node_inputs(d, [&t](const auto &x) {
                return accumulate_parallel(t, x, accumulator_sum());
            });
            check_node_inputs(d, [&t](const auto &x) {
                return accumulate_sequential(t, xt::view(x, xt::range(0, 5)), accumulator_max());
            });
            check_node_inputs(d, [&t](const auto &x) {
                // leaf data: the leaves part of the input itself (same kind of expression)
                return accumulate_and_combine_sequential(t, x, xt::view(x, xt::range(0, 5)), accumulator_mean(),
                                                         std::plus<double>());
            });
        }

        check_node_inputs(d, [&t](const auto &x) {
            return propagate_parallel(t, x);
        });
        check_node_inputs(d, [&t, &d](const auto &x) {
            return propagate_parallel(t, x, d.condition);
        });
        check_node_inputs(d, [&t, &d](const auto &x) {
            return propagate_sequential(t, x, d.condition);
        });
        check_node_inputs(d, [&t](const auto &x) {
            return propagate_sequential_and_accumulate(t, x, accumulator_sum());
        });
        check_node_inputs(d, [&t, &d](const auto &x) {
            return propagate_sequential_and_accumulate(t, x, accumulator_sum(), d.condition);
        });

        // lazy condition
        auto r1 = propagate_parallel(t, d.nodes, d.condition);
        auto r2 = propagate_parallel(t, d.nodes, xt::view(d.condition, xt::all()) && true);
        REQUIRE((r1 == r2));

        // leaf data given as a strided view of the leaves part of a larger array
        auto s1 = accumulate_sequential(t, d.leaves, accumulator_sum());
        auto s2 = accumulate_sequential(t, xt::view(d.strided_nodes(), xt::range(0, 5)), accumulator_sum());
        REQUIRE((s1 == s2));

        // reference values (scalar case)
        auto ap = accumulate_parallel(t, xt::ones<double>({8}), accumulator_sum());
        REQUIRE((ap == xt::xarray<double>{0, 0, 0, 0, 0, 2, 3, 2}));
        auto pp = propagate_parallel(t, xt::arange<index_t>(8));
        REQUIRE((pp == xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7}));
    }

    TEST_CASE("graph and contour accumulators with strided and lazy inputs", "[accumulator_inputs]") {
        inputs d;
        // a cycle on 8 vertices: 8 edges, 8 vertices
        ugraph g(8);
        for (index_t i = 0; i < 8; ++i) {
            add_edge(i, (i + 1) % 8, g);
        }
        check_node_inputs(d, [&g](const auto &x) {
            return accumulate_graph_edges(g, x, accumulator_sum());
        });
        check_node_inputs(d, [&g](const auto &x) {
            return accumulate_graph_vertices(g, x, accumulator_max());
        });

        // contours: path graph on the 5 leaves
        ugraph path(5);
        for (index_t i = 0; i < 4; ++i) {
            add_edge(i, i + 1, path);
        }
        array_1d<index_t> depth{2, 2, 2, 2, 2, 1, 1, 0};
        check_node_inputs(d, [&d, &path, &depth](const auto &x) {
            return accumulate_on_contours(path, d.t, x, depth, accumulator_sum());
        });
        auto c1 = accumulate_on_contours(path, d.t, d.nodes, depth, accumulator_sum());
        auto c2 = accumulate_on_contours(path, d.t, d.nodes, xt::view(depth, xt::all()) + 0, accumulator_sum());
        REQUIRE((c1 == c2));
    }

    TEST_CASE("at accumulator and rag back projection with strided and lazy inputs", "[accumulator_inputs]") {
        inputs d;
        array_1d<index_t> indices{2, 0, -1, 2, 1, 0, -1, 4};
        check_node_inputs(d, [&indices](const auto &x) {
            return accumulate_at(indices, x, accumulator_sum());
        });

        // indices: strided, lazy and of another integral type
        auto ref = accumulate_at(indices, d.nodes, accumulator_sum());
        REQUIRE(ref.shape()[0] == 5);
        REQUIRE(ref.shape()[1] == 3);
        array_1d<index_t> wide_indices = xt::zeros<index_t>({16});
        xt::view(wide_indices, xt::range(0, 16, 2)) = indices;
        REQUIRE((accumulate_at(xt::view(wide_indices, xt::range(0, 16, 2)), d.nodes, accumulator_sum()) == ref));
        REQUIRE((accumulate_at(indices + 0, d.nodes, accumulator_sum()) == ref));
        array_1d<int> int_indices = indices;
        REQUIRE((accumulate_at(int_indices, d.nodes, accumulator_sum()) == ref));

        // every index ignored, or no index at all: empty result
        array_1d<index_t> ignored{-1, -1, -1, -1, -1, -1, -1, -1};
        auto empty_vec = accumulate_at(ignored, d.nodes, accumulator_sum());
        REQUIRE(empty_vec.shape()[0] == 0);
        REQUIRE(empty_vec.shape()[1] == 3);
        auto empty_scalar = accumulate_at(ignored, d.column(), accumulator_sum());
        REQUIRE(empty_scalar.shape()[0] == 0);
        array_1d<index_t> no_index = xt::zeros<index_t>({0});
        auto empty_input = accumulate_at(no_index, xt::zeros<double>({0, 3}), accumulator_sum());
        REQUIRE(empty_input.shape()[0] == 0);
        REQUIRE(empty_input.shape()[1] == 3);
        REQUIRE_THROWS(accumulate_at(no_index, d.nodes, accumulator_sum()));

        // rag back projection: 8 rag vertices projected on 10 vertices
        array_1d<index_t> rag_map{0, 1, 2, 3, 4, 5, 6, 7, 7, -1};
        check_node_inputs(d, [&rag_map](const auto &x) {
            return rag_back_project_weights(rag_map, x);
        });
        auto bp = rag_back_project_weights(rag_map, xt::arange<double>(8));
        REQUIRE((bp == xt::xarray<double>{0, 1, 2, 3, 4, 5, 6, 7, 7, 0}));
    }

    TEST_CASE("light axis view on arrays, views and containers", "[accumulator_inputs]") {
        array_2d<int> a{{1, 2, 3},
                        {4, 5, 6}};
        auto v = make_light_axis_view<true>(a);
        REQUIRE(v.end() - v.begin() == 3);
        v.set_position(1);
        REQUIRE(*v.begin() == 4);
        *v.begin() = 40;
        REQUIRE(a(1, 0) == 40);

        const array_2d<int> &ca = a;
        auto cv = make_light_axis_view<true>(ca);
        static_assert(std::is_same<decltype(cv.begin()), const int *>::value, "read-only view of a const array");

        hg::array<int> h{{1, 2}, {3, 4}, {5, 6}};
        auto hv = make_light_axis_view<true>(h);
        hv.set_position(2);
        REQUIRE(*hv.begin() == 5);
        REQUIRE(hv.end() - hv.begin() == 2);
        auto hcv = make_light_axis_view<true>(hg::array_view<const int>(h));
        static_assert(std::is_same<decltype(hcv.begin()), const int *>::value, "read-only view of a const view");
        hcv.set_position(1);
        hv.set_position(0);
        hv = hcv;
        REQUIRE(h(0, 0) == 3);
        REQUIRE(h(0, 1) == 4);
        hv.combine(hcv, std::plus<int>());
        REQUIRE(h(0, 0) == 6);
        REQUIRE(h(0, 1) == 8);

        // scalar views
        hg::array<double> s{1, 2, 3};
        auto sv = make_light_axis_view<false>(s);
        sv.set_position(2);
        REQUIRE(sv.end() - sv.begin() == 1);
        REQUIRE(*sv.begin() == 3);

        // rows of size 0 and arrays with no row
        array_2d<double> no_rows = xt::zeros<double>({0, 3});
        auto nv = make_light_axis_view<true>(no_rows);
        REQUIRE(nv.begin() == nv.end());
        hg::array<double> zero_width(hg::array_shape(4, 0));
        auto zv = make_light_axis_view<true>(zero_width);
        zv.set_position(3);
        REQUIRE(zv.begin() == zv.end());

        // column-major containers are rejected
        xt::xarray<int, xt::layout_type::column_major> cm{{1, 2}, {3, 4}};
        REQUIRE_THROWS(make_light_axis_view<true>(cm));
    }
}
