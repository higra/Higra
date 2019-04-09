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
        auto t = data.t;

        array_1d<double> node_altitude{1, 2, 0, 3, 2, 5, 9, 12};
        array_1d<index_t> ref{0, 0, 0, 0, 0, 4, 9, 12};
        auto res = attribute_height(t, node_altitude);
        REQUIRE((ref == res));
    }

    TEST_CASE("tree attribute dynamics", "[tree_attributes]") {
        tree t(xt::xarray<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});

        array_1d<double> node_altitude{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10};
        array_1d<index_t> ref{0, 0, 0, 0, 0, 0, 0, 10, 3, 10, 10, 2, 10};
        auto res = attribute_dynamics(t, node_altitude);
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

}