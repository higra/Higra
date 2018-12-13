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
#include "higra/image/graph_image.hpp"
#include "higra/algo/graph_weights.hpp"
#include "../test_utils.hpp"

BOOST_AUTO_TEST_SUITE(graphWeights);

    using namespace hg;
    using namespace std;

    BOOST_AUTO_TEST_CASE(graphWeightingScalar) {

        auto g = get_4_adjacency_graph({2, 2});

        xt::xarray<double> data{0, 1, 2, 3};

        xt::xarray<double> ref1{0.5, 1, 2, 2.5};
        auto r1 = weight_graph(g, data, hg::weight_functions::mean);
        BOOST_CHECK(xt::allclose(ref1, r1));

        xt::xarray<double> ref2{0, 0, 1, 2};
        auto r2 = weight_graph(g, data, hg::weight_functions::min);
        BOOST_CHECK(xt::allclose(ref2, r2));

        xt::xarray<double> ref3{1, 2, 3, 3};
        auto r3 = weight_graph(g, data, hg::weight_functions::max);
        BOOST_CHECK(xt::allclose(ref3, r3));

        xt::xarray<double> ref4{1, 2, 2, 1};
        auto r4 = weight_graph(g, data, hg::weight_functions::L1);
        BOOST_CHECK(xt::allclose(ref4, r4));

        xt::xarray<double> ref5{std::sqrt(1), 2, 2, std::sqrt(1)};
        auto r5 = weight_graph(g, data, hg::weight_functions::L2);
        BOOST_CHECK(xt::allclose(ref5, r5));

        xt::xarray<double> ref6{1, 2, 2, 1};
        auto r6 = weight_graph(g, data, hg::weight_functions::L_infinity);
        BOOST_CHECK(xt::allclose(ref6, r6));

        xt::xarray<double> ref7{1, 4, 4, 1};
        auto r7 = weight_graph(g, data, hg::weight_functions::L2_squared);
        BOOST_CHECK(xt::allclose(ref7, r7));

        xt::xarray<double> data2{0, 0, 2, 0};
        xt::xarray<double> ref8{0, 1, 0, 1};
        auto r8 = weight_graph(g, data2, hg::weight_functions::L0);
        BOOST_CHECK(xt::allclose(ref8, r8));

        xt::xarray<double> ref9{0, 0, 1, 2};
        auto r9 = weight_graph(g, data, hg::weight_functions::source);
        BOOST_CHECK(xt::allclose(ref9, r9));

        xt::xarray<double> ref10{1, 2, 3, 3};
        auto r10 = weight_graph(g, data, hg::weight_functions::target);
        BOOST_CHECK(xt::allclose(ref10, r10));
    }

    BOOST_AUTO_TEST_CASE(graphWeightingVectorial) {

        auto g = get_4_adjacency_graph({2, 2});

        xt::xarray<double> data{{0, 1},
                                {2, 3},
                                {4, 5},
                                {6, 7}};

        xt::xarray<double> ref4{4, 8, 8, 4};
        auto r4 = weight_graph(g, data, hg::weight_functions::L1);
        BOOST_CHECK(xt::allclose(ref4, r4));

        xt::xarray<double> ref5{std::sqrt(8), std::sqrt(32), std::sqrt(32), std::sqrt(8)};
        auto r5 = weight_graph(g, data, hg::weight_functions::L2);
        BOOST_CHECK(xt::allclose(ref5, r5));

        xt::xarray<double> ref6{2, 4, 4, 2};
        auto r6 = weight_graph(g, data, hg::weight_functions::L_infinity);
        BOOST_CHECK(xt::allclose(ref6, r6));

        xt::xarray<double> ref7{8, 32, 32, 8};
        auto r7 = weight_graph(g, data, hg::weight_functions::L2_squared);
        BOOST_CHECK(xt::allclose(ref7, r7));

    }


BOOST_AUTO_TEST_SUITE_END();