//
// Created by user on 4/5/18.
//

//
// Created by user on 4/2/18.
//

#include <boost/test/unit_test.hpp>
#include "higra/algo/graph_image.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE(hierarchyCore);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(testBPTtrivial) {

        auto graph = get_4_adjacency_graph({1, 2});

        xt::xarray<double> edge_weights{2};

        auto res = bptCanonical(graph, edge_weights);
        auto tree = std::get<0>(res);
        auto altitude = std::get<1>(res);
        auto mst = std::get<2>(res);
        BOOST_CHECK(num_vertices(tree) == 3);
        BOOST_CHECK(num_edges(tree) == 2);
        BOOST_CHECK(xt::allclose(tree.parents(), xt::xarray<uint>({2, 2, 2})));
        BOOST_CHECK(xt::allclose(altitude, xt::xarray<double>({0, 0, 2})));
        BOOST_CHECK(num_vertices(mst) == 2);
        BOOST_CHECK(num_edges(mst) == 1);

    }


    BOOST_AUTO_TEST_CASE(testBPT) {

        auto graph = get_4_adjacency_graph({2, 3});

        xt::xarray<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

        auto res = bptCanonical(graph, edge_weights);
        auto tree = std::get<0>(res);
        auto altitude = std::get<1>(res);
        auto mst = std::get<2>(res);
        BOOST_CHECK(num_vertices(tree) == 11);
        BOOST_CHECK(num_edges(tree) == 10);
        BOOST_CHECK(xt::allclose(tree.parents(), xt::xarray<uint>({6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10})));
        BOOST_CHECK(xt::allclose(altitude, xt::xarray<double>({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2})));
        BOOST_CHECK(num_vertices(mst) == 6);
        BOOST_CHECK(num_edges(mst) == 5);
        std::vector<std::pair<ulong, ulong> > ref = {{0, 3},
                                                     {0, 1},
                                                     {1, 4},
                                                     {2, 5},
                                                     {1, 2}};
        for (std::size_t i = 0; i < ref.size(); i++) {
            auto e = edge(i, mst);
            BOOST_CHECK(e == ref[i]);
        }
    }


BOOST_AUTO_TEST_SUITE_END();