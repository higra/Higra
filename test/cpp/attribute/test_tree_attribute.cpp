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

        _data() : t(xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
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

    TEST_CASE("tree attribute height", "[tree_attributes]") {
        hg::tree t(xt::xarray<index_t>{7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11});
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
        tree t(xt::xarray<index_t>{11, 11, 9, 9, 8, 8, 13, 13, 10, 10, 12, 12, 14, 14, 14});

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
        tree t(xt::xarray<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});

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
        tree t(xt::xarray<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});

        array_1d<double> node_altitude{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10};
        array_1d<index_t> ref{3, 3, 0, 10, 10, 2, 2, 10, 3, 10, 10, 2, 10};
        auto res = attribute_dynamics(t, node_altitude, true);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute dynamics2", "[tree_attributes]") {
        tree t(xt::xarray<index_t>{11, 11, 9, 9, 8, 8, 13, 13, 10, 10, 12, 12, 14, 14, 14});

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

    TEST_CASE("tree attribute perimeter length component tree", "[tree_attributes]") {
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

        auto res = attribute_perimeter_length_component_tree(t, g, vertex_perimeters, edge_length);

        array_1d<double> ref{4, 4, 4, 4,
                             4, 4, 4, 4,
                             4, 4, 4, 4,
                             4, 4, 4, 4,
                             4, 6, 4, 4, 4, 10, 6, 10, 22, 20, 18, 16, 16};

        REQUIRE(xt::allclose(ref, res));
    }

}