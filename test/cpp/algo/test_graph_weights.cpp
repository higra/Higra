/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/image/graph_image.hpp"
#include "higra/algo/graph_weights.hpp"
#include "../test_utils.hpp"


using namespace hg;
using namespace std;

namespace test_graph_weights {
    TEST_CASE("graph edge weighting scalar", "[graph_weights]") {

        auto g = get_4_adjacency_graph({2, 2});

        array_1d<double> data{0, 1, 2, 3};

        array_1d<double> ref1{0.5, 1, 2, 2.5};
        auto r1 = weight_graph(g, data, hg::weight_functions::mean);
        REQUIRE(xt::allclose(ref1, r1));

        array_1d<double> ref2{0, 0, 1, 2};
        auto r2 = weight_graph(g, data, hg::weight_functions::min);
        REQUIRE(xt::allclose(ref2, r2));

        array_1d<double> ref3{1, 2, 3, 3};
        auto r3 = weight_graph(g, data, hg::weight_functions::max);
        REQUIRE(xt::allclose(ref3, r3));

        array_1d<double> ref4{1, 2, 2, 1};
        auto r4 = weight_graph(g, data, hg::weight_functions::L1);
        REQUIRE(xt::allclose(ref4, r4));

        array_1d<double> ref5{std::sqrt(1), 2, 2, std::sqrt(1)};
        auto r5 = weight_graph(g, data, hg::weight_functions::L2);
        REQUIRE(xt::allclose(ref5, r5));

        array_1d<double> ref6{1, 2, 2, 1};
        auto r6 = weight_graph(g, data, hg::weight_functions::L_infinity);
        REQUIRE(xt::allclose(ref6, r6));

        array_1d<double> ref7{1, 4, 4, 1};
        auto r7 = weight_graph(g, data, hg::weight_functions::L2_squared);
        REQUIRE(xt::allclose(ref7, r7));

        array_1d<double> data2{0, 0, 2, 0};
        array_1d<double> ref8{0, 1, 0, 1};
        auto r8 = weight_graph(g, data2, hg::weight_functions::L0);
        REQUIRE(xt::allclose(ref8, r8));

        array_1d<double> ref9{0, 0, 1, 2};
        auto r9 = weight_graph(g, data, hg::weight_functions::source);
        REQUIRE(xt::allclose(ref9, r9));

        array_1d<double> ref10{1, 2, 3, 3};
        auto r10 = weight_graph(g, data, hg::weight_functions::target);
        REQUIRE(xt::allclose(ref10, r10));
    }

    TEST_CASE("graph edge weighting vectorial", "[graph_weights]") {

        auto g = get_4_adjacency_graph({2, 2});

        array_2d<double> data{{0, 1},
                              {2, 3},
                              {4, 5},
                              {6, 7}};

        array_1d<double> ref4{4, 8, 8, 4};
        auto r4 = weight_graph(g, data, hg::weight_functions::L1);
        REQUIRE(xt::allclose(ref4, r4));

        array_1d<double> ref5{std::sqrt(8), std::sqrt(32), std::sqrt(32), std::sqrt(8)};
        auto r5 = weight_graph(g, data, hg::weight_functions::L2);
        REQUIRE(xt::allclose(ref5, r5));

        array_1d<double> ref6{2, 4, 4, 2};
        auto r6 = weight_graph(g, data, hg::weight_functions::L_infinity);
        REQUIRE(xt::allclose(ref6, r6));

        array_1d<double> ref7{8, 32, 32, 8};
        auto r7 = weight_graph(g, data, hg::weight_functions::L2_squared);
        REQUIRE(xt::allclose(ref7, r7));

        array_2d<double> data2{{0, 1},
                              {2, 3},
                              {0, 1},
                              {6, 7}};

        array_1d<double> ref8{1, 0, 1, 1};
        auto r8 = weight_graph(g, data2, hg::weight_functions::L0);
        REQUIRE(xt::allclose(ref8, r8));
    }
}
