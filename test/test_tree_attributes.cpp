/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/attributes/tree_attributes.hpp"

BOOST_AUTO_TEST_SUITE(tree_attributes);

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<long>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;


    BOOST_AUTO_TEST_CASE(test_attribute_sum) {
        auto t = data.t;

        array_1d<long> ref{1, 1, 1, 1, 1, 2, 3, 5};
        auto res = attribute_area(t);
        BOOST_CHECK(ref == res);

        array_1d<long> leaf_area{2, 1, 1, 3, 2};
        array_1d<long> ref2{2, 1, 1, 3, 2, 3, 6, 9};
        auto res2 = attribute_area(t, leaf_area);
        BOOST_CHECK(ref2 == res2);
    }

    BOOST_AUTO_TEST_CASE(test_attribute_volume) {
        auto t = data.t;

        array_1d<long> node_area{2, 1, 1, 3, 2, 3, 6, 9};
        array_1d<double> node_altitude{0, 0, 0, 0, 0, 2, 1, 4};
        array_1d<long> ref{4, 2, 1, 3, 2, 12, 24, 36};
        auto res = attribute_volume(t, node_altitude, node_area);
        BOOST_CHECK(ref == res);
    }

    BOOST_AUTO_TEST_CASE(test_attribute_depth) {
        auto t = data.t;

        array_1d<long> ref{2, 2, 2, 2, 2, 1, 1, 0};
        auto res = attribute_depth(t);
        BOOST_CHECK(ref == res);
    }

    BOOST_AUTO_TEST_CASE(test_attribute_height) {
        auto t = data.t;

        array_1d<double> node_altitude{1, 2, 0, 3, 2, 5, 9, 12};
        array_1d<long> ref{0, 0, 0, 0, 0, 4, 9, 12};
        auto res = attribute_height(t, node_altitude);
        BOOST_CHECK(ref == res);
    }

    BOOST_AUTO_TEST_CASE(test_attribute_extinction) {
        auto t = data.t;

        array_1d<double> base_attribute{0, 0, 0, 0, 0, 4, 9, 12};
        array_1d<long> ref{4, 4, 12, 12, 12, 4, 12, 12};
        auto res = attribute_extinction(t, base_attribute);
        BOOST_CHECK(ref == res);
    }



BOOST_AUTO_TEST_SUITE_END();