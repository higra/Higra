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
#include "higra/algo/rag.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

namespace test_rag {
    struct fixture {

        region_adjacency_graph data;

        fixture() {
            auto g = hg::get_4_adjacency_graph({4, 4});
            array_1d<int> vertex_labels{1, 1, 5, 5,
                                        1, 1, 5, 5,
                                        1, 1, 3, 3,
                                        1, 1, 10, 10};

            data = make_region_adjacency_graph_from_labelisation(g, vertex_labels);
        }
    };

    TEST_CASE("simple rag", "[rag]") {

        fixture d;
        auto &rag = d.data.rag;
        auto &vertex_map = d.data.vertex_map;
        auto &edge_map = d.data.edge_map;

        REQUIRE(num_vertices(rag) == 4);
        REQUIRE(num_edges(rag) == 5);

        std::vector<ugraph::edge_descriptor> expected_edges = {
                {0, 1, 0},
                {1, 2, 1},
                {0, 2, 2},
                {2, 3, 3},
                {0, 3, 4}
        };
        index_t i = 0;
        for (auto e: edge_iterator(rag)) {
            REQUIRE(e == expected_edges[i++]);
        }

        array_1d<index_t> expected_vertex_map{
                0, 0, 1, 1,
                0, 0, 1, 1,
                0, 0, 2, 2,
                0, 0, 3, 3
        };
        REQUIRE(xt::equal(vertex_map, expected_vertex_map)());

        auto iv = invalid_index;
        array_1d<index_t> expected_edge_map{
                iv, iv, 0, iv, iv, iv, iv,
                iv, iv, 0, iv, iv, 1, 1,
                iv, iv, 2, iv, iv, 3, 3,
                iv, 4, iv
        };

        REQUIRE((edge_map == expected_edge_map));

    }

    TEST_CASE("simple rag 2", "[rag]") {

        auto g = get_4_adjacency_graph({3, 3});
        array_1d<int> labels{0, 1, 2, 3, 4, 2, 3, 4, 2};
        auto rag = make_region_adjacency_graph_from_labelisation(g, labels);

        REQUIRE(num_vertices(rag.rag) == 5);
        REQUIRE(num_edges(rag.rag) == 6);

        array_1d<int> edge_weights{1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1};
        auto rag2 = make_region_adjacency_graph_from_graph_cut(g, edge_weights);

        REQUIRE(num_vertices(rag2.rag) == 5);
        REQUIRE(num_edges(rag2.rag) == 6);

        REQUIRE((rag.vertex_map == rag2.vertex_map));
        REQUIRE((rag.edge_map == rag2.edge_map));

    }


    TEST_CASE("rag from graph cut", "[rag]") {

        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{0, 0, 1, 0, 0, 0, 0,
                                   0, 0, 1, 0, 0, 1, 1,
                                   0, 0, 1, 0, 0, 1, 1,
                                   0, 1, 0};

        auto data = make_region_adjacency_graph_from_graph_cut(g, edge_weights);
        auto &rag = data.rag;
        auto &vertex_map = data.vertex_map;
        auto &edge_map = data.edge_map;

        REQUIRE(num_vertices(rag) == 4);
        REQUIRE(num_edges(rag) == 5);

        std::vector<ugraph::edge_descriptor> expected_edges = {
                {0, 1, 0},
                {1, 2, 1},
                {0, 2, 2},
                {2, 3, 3},
                {0, 3, 4}
        };
        index_t i = 0;
        for (auto e: edge_iterator(rag)) {
            REQUIRE(e == expected_edges[i++]);
        }

        array_1d<index_t> expected_vertex_map{
                0, 0, 1, 1,
                0, 0, 1, 1,
                0, 0, 2, 2,
                0, 0, 3, 3
        };
        REQUIRE((vertex_map == expected_vertex_map));

        auto iv = invalid_index;
        array_1d<index_t> expected_edge_map{
                iv, iv, 0, iv, iv, iv, iv,
                iv, iv, 0, iv, iv, 1, 1,
                iv, iv, 2, iv, iv, 3, 3,
                iv, 4, iv
        };
        REQUIRE((edge_map == expected_edge_map));

    }

    TEST_CASE("rag back project vertex weights", "[rag]") {

        fixture d;
        auto &vertex_map = d.data.vertex_map;

        array_nd<double> rag_vertex_weights{5, 7, 1, 3};
        auto vertex_weights = rag_back_project_weights(vertex_map, rag_vertex_weights);
        array_nd<double> expected_vertex_weights{
                5, 5, 7, 7,
                5, 5, 7, 7,
                5, 5, 1, 1,
                5, 5, 3, 3
        };
        REQUIRE((vertex_weights == expected_vertex_weights));

        array_nd<double> rag_vertex_weights_vec{
                {5, 2},
                {7, 1},
                {1, 9},
                {3, -2}
        };
        auto vertex_weights_vec = rag_back_project_weights(vertex_map, rag_vertex_weights_vec);
        array_nd<double> expected_vertex_weights_vec{
                {5, 2},
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
                {3, -2}
        };
        REQUIRE((vertex_weights_vec == expected_vertex_weights_vec));
    }

    TEST_CASE("rag back project edge weights", "[rag]") {

        fixture d;
        auto &edge_map = d.data.edge_map;

        array_nd<double> rag_edge_weights{5, 7, 1, 3, 2};
        auto edge_weights = rag_back_project_weights(edge_map, rag_edge_weights);
        double iv = 0;
        array_nd<double> expected_edge_weights{
                iv, iv, 5, iv, iv, iv, iv,
                iv, iv, 5, iv, iv, 7, 7,
                iv, iv, 1, iv, iv, 3, 3,
                iv, 2, iv
        };

        REQUIRE((edge_weights == expected_edge_weights));

        array_nd<double> rag_edge_weights_vec{
                {5, 1},
                {7, 1},
                {1, 9},
                {3, -4},
                {2, 8}
        };
        auto edge_weights_vec = rag_back_project_weights(edge_map, rag_edge_weights_vec);
        array_nd<double> expected_edge_weights_vec{
                {0, 0},
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
                {0, 0}
        };
        REQUIRE((edge_weights_vec == expected_edge_weights_vec));
    }

    TEST_CASE("rag accumulate vertex weights", "[rag]") {

        fixture d;

        auto &vertex_map = d.data.vertex_map;

        array_1d<int> vertex_weights({16}, 1);

        auto rag_vertex_weights = rag_accumulate(vertex_map, vertex_weights, accumulator_sum());
        array_nd<double> expected_rag_vertex_weights{8, 4, 2, 2};
        REQUIRE((rag_vertex_weights == expected_rag_vertex_weights));

        array_2d<int> vertex_weights_vec({16, 2}, 1);
        auto rag_vertex_weights_vec = rag_accumulate(vertex_map, vertex_weights_vec, accumulator_sum());
        array_nd<double> expected_rag_vertex_weights_vec{
                {8, 8},
                {4, 4},
                {2, 2},
                {2, 2}
        };
        REQUIRE((rag_vertex_weights_vec == expected_rag_vertex_weights_vec));
    }

    TEST_CASE("rag accumulate edge weights", "[rag]") {

        fixture d;

        auto &edge_map = d.data.edge_map;

        array_1d<int> edge_weights({24}, 1);

        auto rag_edge_weights = rag_accumulate(edge_map, edge_weights, accumulator_sum());
        array_nd<double> expected_rag_edge_weights{2, 2, 1, 2, 1};
        REQUIRE((rag_edge_weights == expected_rag_edge_weights));

        array_2d<int> edge_weights_vec({24, 2}, 1);
        auto rag_edge_weights_vec = rag_accumulate(edge_map, edge_weights_vec, accumulator_sum());
        array_nd<double> expected_rag_edge_weights_vec{
                {2, 2},
                {2, 2},
                {1, 1},
                {2, 2},
                {1, 1}
        };
        REQUIRE((rag_edge_weights_vec == expected_rag_edge_weights_vec));
    }

}