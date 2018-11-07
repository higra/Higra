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
#include "../test_utils.hpp"

BOOST_AUTO_TEST_SUITE(graphImage);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(test4AdjGraphExplicit) {

        auto g = get_4_adjacency_graph({2, 3});

        vector<vector<pair<unsigned long, unsigned long>>> outListsRef{{{0, 1}, {0, 3}},
                                                                       {{1, 0}, {1, 2}, {1, 4}},
                                                                       {{2, 1}, {2, 5}},
                                                                       {{3, 0}, {3, 4}},
                                                                       {{4, 1}, {4, 3}, {4, 5}},
                                                                       {{5, 2}, {5, 4}}
        };
        vector<vector<pair<unsigned long, unsigned long>>> outListsTest;

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

        vector<vector<pair<unsigned long, unsigned long>>> outListsRef{{{0, 1}, {0, 3}},
                                                                       {{1, 0}, {1, 2}, {1, 4}},
                                                                       {{2, 1}, {2, 5}},
                                                                       {{3, 0}, {3, 4}},
                                                                       {{4, 1}, {4, 3}, {4, 5}},
                                                                       {{5, 2}, {5, 4}}
        };
        vector<vector<pair<unsigned long, unsigned long>>> outListsTest;

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

        vector<vector<pair<unsigned long, unsigned long>>> outListsRef{{{0, 1}, {0, 3}, {0, 4}},
                                                                       {{1, 0}, {1, 2}, {1, 3}, {1, 4}, {1, 5}},
                                                                       {{2, 1}, {2, 4}, {2, 5}},
                                                                       {{3, 0}, {3, 1}, {3, 4}},
                                                                       {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 5}},
                                                                       {{5, 1}, {5, 2}, {5, 4}}
        };
        vector<vector<pair<unsigned long, unsigned long>>> outListsTest;

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

        vector<vector<pair<unsigned long, unsigned long>>> outListsRef{{{0, 1}, {0, 3}, {0, 4}},
                                                                       {{1, 0}, {1, 2}, {1, 3}, {1, 4}, {1, 5}},
                                                                       {{2, 1}, {2, 4}, {2, 5}},
                                                                       {{3, 0}, {3, 1}, {3, 4}},
                                                                       {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 5}},
                                                                       {{5, 1}, {5, 2}, {5, 4}}
        };
        vector<vector<pair<unsigned long, unsigned long>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }

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
        auto r = graph_4_adjacency_2_khalimsky(g, {4, 5}, data);
        BOOST_CHECK(xt::allclose(ref, r));

        xt::xarray<int> ref2{{0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {1, 1, 1, 1, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0}};
        auto r2 = graph_4_adjacency_2_khalimsky(g, {4, 5}, data, true);
        BOOST_CHECK(xt::allclose(ref2, r2));
    }


    BOOST_AUTO_TEST_CASE(khalimsky2graph) {


        array_1d<std::size_t> ref_shape{4, 5};

        xt::xarray<int> data{0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 1, 1, 1, 2, 0, 3, 0, 0, 0, 0, 2,
                             3};

        xt::xarray<int> ref{{0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {1, 1, 1, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 0, 0, 2, 0, 3, 0}};
        auto r = khalimsky_2_graph_4_adjacency(ref);
        //auto & graph = std::get<0>(r);
        auto &embedding = std::get<1>(r);
        auto &weights = std::get<2>(r);
        BOOST_CHECK(xt::allclose(embedding.shape(), ref_shape));
        BOOST_CHECK(xt::allclose(data, weights));

        xt::xarray<int> ref2{{0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0},
                             {1, 1, 1, 1, 1, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0},
                             {0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0}};
        auto r2 = khalimsky_2_graph_4_adjacency(ref2, true);
        //auto & graph2 = std::get<0>(r2);
        auto &embedding2 = std::get<1>(r2);
        auto &weights2 = std::get<2>(r2);
        BOOST_CHECK(xt::allclose(embedding2.shape(), ref_shape));
        BOOST_CHECK(xt::allclose(data, weights2));
    }



BOOST_AUTO_TEST_SUITE_END();