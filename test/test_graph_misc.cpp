//
// Created by user on 4/2/18.
//

#include <boost/test/unit_test.hpp>

#include "graph.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE(boost_neighbouringgraph);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(test4AdjGraphExplicit) {

        auto g = get_4_adjacency_graph({2, 3});

        vector<vector<pair<ulong, ulong>>> outListsRef{{{0, 1}, {0, 3}},
                                                       {{1, 0}, {1, 2}, {1, 4}},
                                                       {{2, 1}, {2, 5}},
                                                       {{3, 0}, {3, 4}},
                                                       {{4, 1}, {4, 3}, {4, 5}},
                                                       {{5, 2}, {5, 4}}
        };
        vector<vector<pair<ulong, ulong>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }

    }

    BOOST_AUTO_TEST_CASE(test4AdjGraphImplicit) {

        auto g = get_4_adjacency_implicit_graph({2, 3});

        vector<vector<pair<ulong, ulong>>> outListsRef{{{0, 1}, {0, 3}},
                                                       {{1, 0}, {1, 2}, {1, 4}},
                                                       {{2, 1}, {2, 5}},
                                                       {{3, 0}, {3, 4}},
                                                       {{4, 1}, {4, 3}, {4, 5}},
                                                       {{5, 2}, {5, 4}}
        };
        vector<vector<pair<ulong, ulong>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }

    }

    BOOST_AUTO_TEST_CASE(test8AdjGraphExplicit) {

        auto g = get_8_adjacency_graph({2, 3});

        vector<vector<pair<ulong, ulong>>> outListsRef{{{0, 1}, {0, 3}, {0, 4}},
                                                       {{1, 0}, {1, 2}, {1, 3}, {1, 4}, {1, 5}},
                                                       {{2, 1}, {2, 4}, {2, 5}},
                                                       {{3, 0}, {3, 1}, {3, 4}},
                                                       {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 5}},
                                                       {{5, 1}, {5, 2}, {5, 4}}
        };
        vector<vector<pair<ulong, ulong>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }

    }

    BOOST_AUTO_TEST_CASE(test8AdjGraphImplicit) {

        auto g = get_8_adjacency_implicit_graph({2, 3});

        vector<vector<pair<ulong, ulong>>> outListsRef{{{0, 1}, {0, 3}, {0, 4}},
                                                       {{1, 0}, {1, 2}, {1, 3}, {1, 4}, {1, 5}},
                                                       {{2, 1}, {2, 4}, {2, 5}},
                                                       {{3, 0}, {3, 1}, {3, 4}},
                                                       {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 5}},
                                                       {{5, 1}, {5, 2}, {5, 4}}
        };
        vector<vector<pair<ulong, ulong>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }

    }


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

    BOOST_AUTO_TEST_CASE(graph2d2khalimsky) {


        auto g = get_4_adjacency_graph({4, 5});

        xt::xarray<int> data{0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 1, 1, 1, 2, 0, 3, 0, 0, 0, 0, 2,
                             3};

        xt::xarray<int> ref{{0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {1, 1, 1, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 0, 0, 2, 0, 3, 0}};
        auto r = contour2d_2_khalimsky(g, {4, 5}, data);
        BOOST_CHECK(xt::allclose(ref, r));

    }



BOOST_AUTO_TEST_SUITE_END();