/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include <cmath>
#include <sstream>
#include "higra/algo/tree_energy_optimization.hpp"
#include "higra/graph.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;
using namespace std;
using namespace hg::tree_energy_optimization_internal;

using lep_t = piecewise_linear_energy_function_piece<double>;
using lef_t = piecewise_linear_energy_function<double>;

std::ostream &operator<<(std::ostream &os, const lep_t &rhs) {
    os << "((" << rhs.origin_x() << ", " << rhs.origin_y() << "), " << rhs.slope() << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const lef_t &rhs) {
    os << "{";
    for (const auto &lp: rhs) {
        os << lp << ", ";
    }
    os << "}";
    return os;
}

namespace Catch {

    template<>
    struct StringMaker<lep_t> {
        static std::string convert(lep_t const &rhs) {
            std::ostringstream strs;
            strs << "((" << rhs.origin_x() << ", " << rhs.origin_y() << "), " << rhs.slope() << ")";
            return strs.str();
        }
    };

}

namespace test_linear_energy_function_optimization {

    TEST_CASE("test piecewise_linear_energy_function_piece", "[linear_energy_function_optimization]") {

        lep_t p1(1, 2, 3);
        lep_t p2(2, 1, 3);
        lep_t p3(1, 2, 3);

        REQUIRE(p1 != p2);
        REQUIRE(p1 == p3);

        REQUIRE(p1(1) == 2);
        REQUIRE(p1(0) == -1);
    }

    TEST_CASE("test piecewise_linear_energy_function sum", "[linear_energy_function_optimization]") {

        SECTION("edge case empty") {
            lep_t p1(1, 2, 3);
            lef_t f1(p1);
            lef_t empty;

            REQUIRE(empty.sum(empty) == empty);
            REQUIRE(f1.sum(empty) == f1);
            REQUIRE(empty.sum(f1) == f1);
        }
        SECTION("simple + simple") {
            lef_t f1({0, 0, 1});
            lef_t f2({0, 1, 1});

            lef_t r({0, 1, 2});
            REQUIRE(f1.sum(f2) == r);
            REQUIRE(f2.sum(f1) == r);
        }
        SECTION("compound + compound") {
            lef_t f1({{0, 0, 2},
                      {1, 2, 1}});
            lef_t f2({{0,   0,   1},
                      {0.5, 0.5, 0.5},
                      {2.5, 1.5, 0.1}});

            lef_t r({{0,   0,    3},
                     {0.5, 1.5,  2.5},
                     {1,   2.75, 1.5},
                     {2.5, 5,    1.1}});
            REQUIRE(f1.sum(f2) == r);
            REQUIRE(f2.sum(f1) == r);
        }

        SECTION("compound + compound with max pieces") {
            lef_t f1({{0, 0, 2},
                      {1, 2, 1}});
            lef_t f2({{0,   0,   1},
                      {0.5, 0.5, 0.5},
                      {2.5, 1.5, 0.1}});

            lef_t r({{0.,  0.25, 2.5},
                     {1,   2.75, 1.5},
                     {2.5, 5,    1.1}});
            REQUIRE(f1.sum(f2, 3) == r);
            REQUIRE(f2.sum(f1, 3) == r);
        }
    }

    TEST_CASE("test piecewise_linear_energy_function infimum", "[linear_energy_function_optimization]") {

        SECTION("simple no intersection") {
            lef_t f({0, 0, 1});
            lep_t p(0, 1, 1);

            lef_t r(f);
            REQUIRE(f.infimum(p) >= std::numeric_limits<double>::max());
            REQUIRE(f == r);
        }
        SECTION("simple intersection") {
            lef_t f({0, 1, 1});
            lep_t p(0, 2, 0.5);

            lef_t r({{0, 1, 1},
                     {2, 3, 0.5}});
            REQUIRE(f.infimum(p) == 2);
            REQUIRE(f == r);
        }
        SECTION("compound intersection") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 6, 1);

            lef_t r({{0, 0, 5},
                     {1, 5, 3},
                     {2, 8, 1}});
            REQUIRE(f.infimum(p) == 2);
            REQUIRE(f == r);
        }
        SECTION("parallel edge case 1") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 20, 2);

            lef_t r({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            REQUIRE(f.infimum(p) >= std::numeric_limits<double>::max());
            REQUIRE(f == r);
        }
        SECTION("parallel edge case 2") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 5, 2);

            lef_t r({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            REQUIRE(f.infimum(p) == 3);
            REQUIRE(f == r);
        }
        SECTION("parallel edge case 3") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 1, 2);

            lef_t r({{0,      0,      5},
                     {1. / 3, 5. / 3, 2}});
            REQUIRE(f.infimum(p) == Approx(1.0 / 3));
            REQUIRE(f == r);
        }
    }

    TEST_CASE("test labelisation_optimal_cut_from_energy", "[optimal_cut_tree]") {
        tree t(array_1d<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});
        array_1d<double> energy_attribute{2, 1, 3, 2, 1, 1, 1, 2, 2, 4, 10, 5, 20};

        auto res = labelisation_optimal_cut_from_energy(t, energy_attribute);

        array_1d<index_t> ref{0, 0, 1, 1, 1, 2, 3};
        REQUIRE(is_in_bijection(res, ref));
    }

    TEST_CASE("test hierarchy_to_optimal_energy_cut_hierarchy", "[optimal_cut_tree]") {
        tree t(array_1d<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});
        array_1d<double> data_fidelity_attribute{1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 15, 25, 45};
        array_1d<double> regularization_attribute{4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 10, 4, 12};

        auto res = hierarchy_to_optimal_energy_cut_hierarchy(t, data_fidelity_attribute, regularization_attribute);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> ref_parents{8, 8, 9, 7, 7, 10, 10, 9, 9, 10, 10};
        array_1d<double> ref_altitudes{0., 0., 0., 0., 0., 0., 0., 0.5, 0.75, 2.5, 14.0 / 3};
        REQUIRE(tree.parents() == ref_parents);
        REQUIRE(xt::allclose(altitudes, ref_altitudes));
    }

    TEST_CASE("test binary_partition_tree_MumfordShah_energy scalar", "[optimal_cut_tree]") {
        auto g = hg::get_4_adjacency_graph({3, 3});
        array_1d<double> edge_length = xt::ones<double>({num_edges(g)});
        array_1d<double> vertex_perimeter({9}, 4);
        array_1d<double> vertex_values{1, 1, 20, 6, 1, 20, 10, 10, 10};
        array_1d<double> squared_vertex_values = vertex_values * vertex_values;
        array_1d<double> vertex_area = xt::ones<double>({num_vertices(g)});

        auto res = binary_partition_tree_MumfordShah_energy(
                g,
                vertex_perimeter,
                vertex_area,
                vertex_values,
                squared_vertex_values,
                edge_length);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;
        array_1d<index_t> ref_parents{10, 10, 11, 14, 13, 11, 12, 9, 9, 12, 13, 16, 15, 14, 15, 16, 16};
        array_1d<double> ref_altitudes{0., 0., 0.,
                                       0., 0., 0.,
                                       0., 0., 0.,
                                       0., 0., 0.,
                                       0., 0., 4.6875, 25.741071, 53.973545};
        REQUIRE(tree.parents() == ref_parents);
        REQUIRE(xt::allclose(altitudes, ref_altitudes));
    }

    TEST_CASE("test binary_partition_tree_MumfordShah_energy vectorial", "[optimal_cut_tree]") {
        auto g = hg::get_4_adjacency_graph({3, 3});
        array_1d<double> edge_length = xt::ones<double>({num_edges(g)});
        array_1d<double> vertex_perimeter({9}, 4);
        array_2d<double> vertex_values{{1, 2}, {1, 2}, {20, 30}, {6, 7}, {1, 2}, {20, 30}, {10, 12}, {10, 12},
                                       {10, 12}};
        array_2d<double> squared_vertex_values = vertex_values * vertex_values;
        array_1d<double> vertex_area = xt::ones<double>({num_vertices(g)});

        auto res = binary_partition_tree_MumfordShah_energy(
                g,
                vertex_perimeter,
                vertex_area,
                vertex_values,
                squared_vertex_values,
                edge_length);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> ref_parents{10, 10, 11, 14, 13, 11, 12, 9, 9, 12, 13, 16, 15, 14, 15, 16, 16};
        array_1d<double> ref_altitudes{0., 0., 0.,
                                       0., 0., 0.,
                                       0., 0., 0.,
                                       0., 0., 0.,
                                       0., 0., 9.375, 58.553571, 191.121693};
        REQUIRE(tree.parents() == ref_parents);
        REQUIRE(xt::allclose(altitudes, ref_altitudes));
    }
}