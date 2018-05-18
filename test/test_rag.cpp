//
// Created by perretb on 17/05/18.
//

#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/algo/rag.hpp"
#include "higra/algo/graph_image.hpp"


using namespace hg;

BOOST_AUTO_TEST_SUITE(region_adjacency_graph_test);

    struct fixture {

        region_adjacency_graph data;

        fixture() {
            auto g = hg::get_4_adjacency_graph({4, 4});
            array_1d<int> vertex_labels{1, 1, 5, 5,
                                        1, 1, 5, 5,
                                        1, 1, 3, 3,
                                        1, 1, 10, 10};

            data = make_region_adjacency_graph(g, vertex_labels);
        }
    };


    BOOST_AUTO_TEST_CASE(rag_simple) {

        fixture d;

        auto &rag = d.data.rag;
        auto &vertex_map = d.data.vertex_map;
        auto &edge_map = d.data.edge_map;

        BOOST_CHECK(num_vertices(rag) == 4);
        BOOST_CHECK(num_edges(rag) == 5);

        std::vector<std::pair<std::size_t, std::size_t>> expected_edges = {{0, 1},
                                                                           {1, 2},
                                                                           {0, 2},
                                                                           {2, 3},
                                                                           {0, 3}};
        index_t i = 0;
        for (auto e: edge_iterator(rag)) {
            BOOST_CHECK(e == expected_edges[i++]);
        }

        array_1d<index_t> expected_vertex_map{0, 0, 1, 1,
                                              0, 0, 1, 1,
                                              0, 0, 2, 2,
                                              0, 0, 3, 3};
        BOOST_CHECK(vertex_map == expected_vertex_map);

        auto iv = invalid_index;
        array_1d<index_t> expected_edge_map{iv, iv, 0, iv, iv, iv, iv,
                                            iv, iv, 0, iv, iv, 1, 1,
                                            iv, iv, 2, iv, iv, 3, 3,
                                            iv, 4, iv};
        BOOST_CHECK(edge_map == expected_edge_map);

    }

    BOOST_AUTO_TEST_CASE(back_project_vertex_weights) {

        fixture d;
        auto &vertex_map = d.data.vertex_map;

        array_nd<double> rag_vertex_weights{5, 7, 1, 3};
        auto vertex_weights = rag_back_project_weights(vertex_map, rag_vertex_weights);
        array_nd<double> expected_vertex_weights{5, 5, 7, 7,
                                                 5, 5, 7, 7,
                                                 5, 5, 1, 1,
                                                 5, 5, 3, 3};
        BOOST_CHECK(vertex_weights == expected_vertex_weights);

        array_nd<double> rag_vertex_weights_vec{{5, 2},
                                                {7, 1},
                                                {1, 9},
                                                {3, -2}};
        auto vertex_weights_vec = rag_back_project_weights(vertex_map, rag_vertex_weights_vec);
        array_nd<double> expected_vertex_weights_vec{{5, 2},
                                                     {5, 2},
                                                     {7, 1},
                                                     {7, 1},
                                                     {5, 2},
                                                     {5, 2},
                                                     {7, 1},
                                                     {7, 1},
                                                     {5, 2},
                                                     {5, 2},
                                                     {1, 9},
                                                     {1, 9},
                                                     {5, 2},
                                                     {5, 2},
                                                     {3, -2},
                                                     {3, -2}};
        BOOST_CHECK(vertex_weights_vec == expected_vertex_weights_vec);
    }

    BOOST_AUTO_TEST_CASE(back_project_edge_weights) {

        fixture d;
        auto &edge_map = d.data.edge_map;

        array_nd<double> rag_edge_weights{5, 7, 1, 3, 2};
        auto edge_weights = rag_back_project_weights(edge_map, rag_edge_weights);
        double iv = 0;
        array_nd<double> expected_edge_weights{iv, iv, 5, iv, iv, iv, iv,
                                               iv, iv, 5, iv, iv, 7, 7,
                                               iv, iv, 1, iv, iv, 3, 3,
                                               iv, 2, iv};

        BOOST_CHECK(edge_weights == expected_edge_weights);

        array_nd<double> rag_edge_weights_vec{{5, 1},
                                              {7, 1},
                                              {1, 9},
                                              {3, -4},
                                              {2, 8}};
        auto edge_weights_vec = rag_back_project_weights(edge_map, rag_edge_weights_vec);
        array_nd<double> expected_edge_weights_vec{{0, 0},
                                                   {0, 0},
                                                   {5, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {0, 0},
                                                   {0, 0},
                                                   {0, 0},
                                                   {0, 0},
                                                   {5, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {7, 1},
                                                   {7, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 9},
                                                   {0, 0},
                                                   {0, 0},
                                                   {3, -4},
                                                   {3, -4},
                                                   {0, 0},
                                                   {2, 8},
                                                   {0, 0}};
        BOOST_CHECK(edge_weights_vec == expected_edge_weights_vec);
    }

BOOST_AUTO_TEST_SUITE_END();