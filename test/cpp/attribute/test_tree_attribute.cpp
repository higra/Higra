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
#include "higra/attribute/tree_attribute.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/io/tree_io.hpp"

namespace tree_attributes {

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(hg::array<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;

    TEST_CASE("tree attribute area", "[tree_attributes]") {
        auto t = data.t;

        array_1d<index_t> ref{1, 1, 1, 1, 1, 2, 3, 5};
        auto res = attribute_area(t);
        REQUIRE((ref == res));

        array_1d<index_t> leaf_area{2, 1, 1, 3, 2};
        array_1d<index_t> ref2{2, 1, 1, 3, 2, 3, 6, 9};
        auto res2 = attribute_area(t, leaf_area);
        REQUIRE((ref2 == res2));
    }

    TEST_CASE("tree attribute volume", "[tree_attributes]") {
        auto t = data.t;

        array_1d<index_t> node_area{2, 1, 1, 3, 2, 3, 6, 9};
        array_1d<double> node_altitude{0, 0, 0, 0, 0, 2, 1, 4};
        array_1d<index_t> ref{0, 0, 0, 0, 0, 6, 18, 24};
        auto res = attribute_volume(t, node_altitude, node_area);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute depth", "[tree_attributes]") {
        auto t = data.t;

        array_1d<index_t> ref{2, 2, 2, 2, 2, 1, 1, 0};
        auto res = attribute_depth(t);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attributes area, volume and depth on an empty tree", "[tree_attributes]") {
        hg::tree t(hg::array<index_t>{});
        REQUIRE(attribute_depth(t).size() == 0);
        auto area = attribute_area(t);
        REQUIRE(area.size() == 0);
        REQUIRE(attribute_volume(t, hg::array<double>::from_shape({0}), area).size() == 0);
    }

    TEST_CASE("tree attribute height", "[tree_attributes]") {
        hg::tree t(hg::array<index_t>{7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11});
        SECTION("increasing") {
            array_1d<double> node_altitude{0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 5, 8};
            array_1d<index_t> ref{0, 0, 0, 0, 0, 0, 0, 2, 3, 7, 6, 7};
            auto res = attribute_height(t, node_altitude, true);
            REQUIRE((ref == res));
        }SECTION("decreasing") {
            array_1d<double> node_altitude{0, 0, 0, 0, 0, 0, 0, 8, 5, 9, 4, 1};
            array_1d<index_t> ref{0, 0, 0, 0, 0, 0, 0, 4, 1, 8, 7, 8};
            auto res = attribute_height(t, node_altitude, false);
            REQUIRE((ref == res));
        }
    }

    TEST_CASE("tree attribute extrema", "[tree_attributes]") {
        tree t(hg::array<index_t>{11, 11, 9, 9, 8, 8, 13, 13, 10, 10, 12, 12, 14, 14, 14});

        array_1d<double> node_altitude{0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 4, 8, 10};
        array_1d<bool> ref{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0};
        auto res = attribute_extrema(t, node_altitude);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute extrema2", "[tree_attributes]") {
        auto graph = get_4_adjacency_implicit_graph({4, 4});
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});

        auto res = component_tree_max_tree(graph, vertex_weights);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        auto extrema = attribute_extrema(tree, altitudes);
        array_1d<bool> expected_extrema({0, 0, 0, 0,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
        REQUIRE((expected_extrema == extrema));
    }

    TEST_CASE("tree attribute extinction value", "[tree_attributes]") {
        // same as dynamics
        tree t(hg::array<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});

        array_1d<double> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10};
        array_1d<double> attribute{0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 4, 2, 10};

        array_1d<index_t> ref{3, 3, 0, 10, 10, 2, 2, 10, 3, 10, 10, 2, 10};
        auto res = attribute_extinction_value(t, altitudes, attribute, true);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute extinction value2", "[tree_attributes]") {
        // area extinction on a max tree
        auto graph = get_4_adjacency_implicit_graph({4, 4});
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});

        auto res = component_tree_max_tree(graph, vertex_weights);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        auto area = attribute_area(tree);
        auto ext = attribute_extinction_value(tree, altitudes, area, false);

        array_1d<double> expected_ext({0, 0, 0, 0,
                                     1, 0, 0, 4,
                                     0, 0, 0, 0,
                                     16, 0, 0, 1,
                                     16, 16, 4, 1, 1, 16, 4, 4, 16, 16, 16, 16, 16});
        REQUIRE((ext == expected_ext));
    }

    TEST_CASE("tree attribute dynamics", "[tree_attributes]") {
        tree t(hg::array<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});

        array_1d<double> node_altitude{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10};
        array_1d<index_t> ref{3, 3, 0, 10, 10, 2, 2, 10, 3, 10, 10, 2, 10};
        auto res = attribute_dynamics(t, node_altitude, true);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute dynamics2", "[tree_attributes]") {
        tree t(hg::array<index_t>{11, 11, 9, 9, 8, 8, 13, 13, 10, 10, 12, 12, 14, 14, 14});

        array_1d<double> node_altitude{0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 4, 8, 10};
        array_1d<index_t> ref{3, 3, 0, 0, 10, 10, 2, 2, 10, 0, 10, 3, 10, 2, 10};
        auto res = attribute_dynamics(t, node_altitude, true);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute siblings", "[tree_attributes]") {
        auto t = data.t;

        array_1d<index_t> ref{1, 0, 3, 4, 2, 6, 5, 7};
        auto res = attribute_sibling(t);
        REQUIRE((ref == res));

        array_1d<index_t> ref2{1, 0, 4, 2, 3, 6, 5, 7};
        auto res2 = attribute_sibling(t, -1);
        REQUIRE((ref2 == res2));
    }

    TEST_CASE("tree attribute contour length component tree", "[tree_attributes]") {
        auto g = get_4_adjacency_graph({4, 4});

        /* for reference, tree is a max tree on the following image
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});
        */

        array_1d<index_t> parents({28, 27, 24, 24,
                                   20, 23, 22, 18,
                                   26, 25, 24, 27,
                                   16, 17, 21, 19,
                                   17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28});
        /*
         array_1d<double> altitudes({0, 1, 4, 4,
                                     7, 5, 6, 8,
                                     2, 3, 4, 1,
                                     9, 8, 6, 7, 
                                     9, 8, 8, 7, 7, 6, 6, 5, 4, 3, 2, 1, 0});
         */

        tree t(parents, tree_category::component_tree);

        array_1d<double> vertex_perimeters({num_vertices(g)}, 4);
        array_1d<double> edge_length({num_edges(g)}, 1);

        auto res = attribute_contour_length_component_tree(t, g, vertex_perimeters, edge_length);

        array_1d<double> ref{4, 4, 4, 4,
                             4, 4, 4, 4,
                             4, 4, 4, 4,
                             4, 4, 4, 4,
                             4, 6, 4, 4, 4, 10, 6, 10, 22, 20, 18, 16, 16};

        REQUIRE(hg::allclose(ref, res));
    }

    TEST_CASE("tree attribute child number", "[tree_attributes]") {
        auto t = data.t;

        array_1d<index_t> ref{0, 1, 0, 1, 2, 0, 1, -1};
        auto res = attribute_child_number(t);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute smallest enclosing shape ", "[tree_attributes]") {
        array_1d<index_t> pt1{8, 8, 9, 9, 9, 10, 10, 11, 13, 12, 11, 12, 13, 13};
        tree t1(pt1);

        array_1d<index_t> pt2{11, 11, 8, 8, 10, 9, 9, 9, 10, 12, 11, 12, 12};
        tree t2(pt2);

        array_1d<index_t> ref{0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 9, 12, 12};
        auto res = attribute_smallest_enclosing_shape(t1, t2);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute children pair sum product scalar", "[tree_attributes]") {
        auto t = data.t; //{5, 5, 6, 6, 6, 7, 7, 7}

        array_1d<double> node_weights{2, 3, 4, 5, 6, 7, 8, 9};
        auto res = attribute_children_pair_sum_product(t, node_weights);
        array_1d<double> ref{0, 0, 0, 0, 0, 6, 74, 56};
        REQUIRE(hg::allclose(ref, res));
    }

    TEST_CASE("tree attribute children pair sum product integer overflow", "[tree_attributes]") {
        tree t(array_1d<index_t>{3, 3, 3, 3}); // three leaves, one root: pairs (0, 1), (0, 2), (1, 2)

        // product computed in the result type
        array_1d<int> w_int{1 << 16, 1 << 16, 0, 0};
        auto r_double = attribute_children_pair_sum_product<tree, int, double>(t, w_int);
        REQUIRE(r_double(3) == 4294967296.0);
        auto r_int64 = attribute_children_pair_sum_product<tree, int, int64_t>(t, w_int);
        REQUIRE(r_int64(3) == (int64_t) 1 << 32);

        // integer results wrap around modulo 2^n
        auto r_int = attribute_children_pair_sum_product(t, w_int);
        REQUIRE(r_int(3) == 0);
        array_1d<int16_t> w_int16{300, 300, -3, 0};
        auto r_int16 = attribute_children_pair_sum_product(t, w_int16);
        REQUIRE(r_int16(3) == (int16_t) (90000 - 900 - 900 - 65536));
        array_1d<uint16_t> w_uint16{65535, 65535, 0, 0};
        REQUIRE(attribute_children_pair_sum_product(t, w_uint16)(3) == 1);
        array_1d<int64_t> w_int64{(int64_t) 1 << 62, 4, 0, 0};
        REQUIRE(attribute_children_pair_sum_product(t, w_int64)(3) == 0);

        // non-integer weights with an integer result: product computed before the conversion
        array_1d<double> w_double{1.5, 1.5, 0, 0};
        REQUIRE(attribute_children_pair_sum_product<tree, double, int>(t, w_double)(3) == 2);
    }

    TEST_CASE("tree attribute children pair sum product boolean result", "[tree_attributes]") {
        tree t(array_1d<index_t>{6, 6, 6, 6, 7, 7, 7, 7}); // node 6: children 0..3, root 7: children 4, 5, 6

        // several pairs of true children: the result stays true (it is not a sum modulo 2)
        array_1d<bool> flags{true, true, true, true, false, false, true, false};
        auto r = attribute_children_pair_sum_product(t, flags);
        REQUIRE((r == array_1d<bool>{false, false, false, false, false, false, true, false}));
        array_1d<bool> one_true{true, false, false, false, true, true, false, false};
        REQUIRE((attribute_children_pair_sum_product(t, one_true) ==
                 array_1d<bool>{false, false, false, false, false, false, false, true}));

        // explicit boolean result from integer weights: a product that overflows int is still non-zero
        array_1d<int> integers{1 << 16, 1 << 16, 0, 0, 2, 0, 1, 0};
        REQUIRE((attribute_children_pair_sum_product<tree, int, bool>(t, integers) ==
                 array_1d<bool>{false, false, false, false, false, false, true, true}));

        // floating point weights: product computed in the weight type, then converted
        array_1d<float> floats{1.5f, 1.5f, 0, 0, 0, 0, 0, 0};
        REQUIRE((attribute_children_pair_sum_product<tree, float, bool>(t, floats) ==
                 array_1d<bool>{false, false, false, false, false, false, true, false}));
    }

    TEST_CASE("tree attribute children pair sum product vectorial", "[tree_attributes]") {
        auto t = data.t; //{5, 5, 6, 6, 6, 7, 7, 7}

        array_nd<double> node_weights{2, 9, 3, 8, 4, 7, 5, 6, 6, 5, 7, 4, 8, 3, 9, 2};
        node_weights.reshape({8, 2});
        auto res = attribute_children_pair_sum_product(t, node_weights);
        array_nd<double> ref{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 72, 74, 107, 56, 12};
        ref.reshape({8, 2});
        REQUIRE(hg::allclose(ref, res));
    }

    TEST_CASE("tree attribute height non binary, value types and inputs", "[tree_attributes]") {
        hg::tree t(array_1d<index_t>{7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11});
        SECTION("increasing") {
            array_1d<int> node_altitude{1, 0, 2, 0, 0, 0, 0, 3, 2, 1, 5, 8};
            array_1d<int> ref{0, 0, 0, 0, 0, 0, 0, 2, 3, 7, 6, 7};
            auto res = attribute_height(t, node_altitude, true);
            static_assert(std::is_same<decltype(res), array_1d<int>>::value, "");
            REQUIRE((ref == res));
            // views as inputs
            array_1d<int> wide = hg::zeros<int>({24});
            hg::assign(wide.slice(5, 17), node_altitude);
            REQUIRE((ref == attribute_height(t, wide.slice(5, 17), true)));
            REQUIRE((ref == attribute_height(t, node_altitude.view(), true)));
        }
        SECTION("decreasing") {
            array_1d<float> node_altitude{0, 9, 0, 0, 0, 0, 0, 8, 5, 9, 4, 1};
            array_1d<float> ref{0, 0, 0, 0, 0, 0, 0, 4, 1, 8, 7, 8};
            auto res = attribute_height(t, node_altitude, false);
            static_assert(std::is_same<decltype(res), array_1d<float>>::value, "");
            REQUIRE((ref == res));
        }
    }

    TEST_CASE("tree attribute children pair sum product nd and inputs", "[tree_attributes]") {
        auto t = data.t; //{5, 5, 6, 6, 6, 7, 7, 7}
        array_nd<int> node_weights = hg::arange<int>(32);
        node_weights.reshape({8, 2, 2});
        array_nd<int> ref = hg::zeros<int>({8, 2, 2});
        hg::assign(ref.slice(5, 8), array_3d<int>{{{0, 5}, {12, 21}}, {{416, 491}, {572, 659}}, {{480, 525}, {572, 621}}});
        auto res = attribute_children_pair_sum_product(t, node_weights);
        static_assert(std::is_same<decltype(res), array_nd<int>>::value, "");
        REQUIRE((ref == res));
        array_3d<int> node_weights3 = node_weights;
        REQUIRE((ref == attribute_children_pair_sum_product(t, node_weights3)));
        // view input: rows 2 to 9 of a larger array
        array_3d<int> wide = hg::zeros<int>({12, 2, 2});
        hg::assign(wide.slice(2, 10), node_weights3);
        REQUIRE((ref == attribute_children_pair_sum_product(t, wide.slice(2, 10))));
    }

    TEMPLATE_TEST_CASE("tree attribute children pair sum product with a wider value type", "[tree_attributes]",
                       int, float) {
        auto t = data.t; //{5, 5, 6, 6, 6, 7, 7, 7}

        array_1d<TestType> node_weights{2, 3, 4, 5, 6, 7, 8, 9};
        auto res = attribute_children_pair_sum_product<hg::tree, TestType, double>(t, node_weights);
        static_assert(std::is_same<decltype(res), array_nd<double>>::value, "");
        array_1d<double> ref{0, 0, 0, 0, 0, 6, 74, 56};
        REQUIRE((ref == res));

        array_nd<TestType> node_weights_v{2, 9, 3, 8, 4, 7, 5, 6, 6, 5, 7, 4, 8, 3, 9, 2};
        node_weights_v.reshape({8, 2});
        auto res_v = attribute_children_pair_sum_product<hg::tree, TestType, double>(t, node_weights_v);
        array_nd<double> ref_v{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 72, 74, 107, 56, 12};
        ref_v.reshape({8, 2});
        REQUIRE((ref_v == res_v));
    }

    TEST_CASE("tree attribute children pair sum product, products computed in the weight type", "[tree_attributes]") {
        auto t = data.t; //{5, 5, 6, 6, 6, 7, 7, 7}
        // each product is computed in float, then converted to double
        array_1d<float> node_weights{1.1f, 1.3f, 0.7f, 1.9f, 2.3f, 0.3f, 1.7f, 2.9f};
        auto res = attribute_children_pair_sum_product<hg::tree, float, double>(t, node_weights);
        array_1d<double> ref = hg::zeros<double>({8});
        ref(5) = (double) (node_weights(0) * node_weights(1));
        ref(6) = (double) (node_weights(2) * node_weights(3)) + (double) (node_weights(2) * node_weights(4)) +
                 (double) (node_weights(3) * node_weights(4));
        ref(7) = (double) (node_weights(5) * node_weights(6));
        REQUIRE((ref == res));
    }
}
