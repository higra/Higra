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
#include "higra/hierarchy/binary_partition_tree.hpp"
#include "higra/image/graph_image.hpp"
#include "xtensor/generators/xrandom.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/algo/tree.hpp"


using namespace hg;
using namespace std;

namespace binary_partition_tree {

    TEST_CASE("single linkage clustering simple", "[binary_partition_tree]") {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<double> edge_weights({1, 9, 6, 7, 5, 8, 12, 4, 10, 11, 2, 3});
        auto res = binary_partition_tree_min_linkage(graph, edge_weights);
        auto &tree = res.tree;
        auto &levels = res.altitudes;

        array_1d<index_t> expected_parents({9, 9, 13, 15, 12, 12, 10, 10, 11, 14, 11, 16, 13, 14, 15, 16, 16});
        array_1d<double> expected_levels({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 10});

        REQUIRE((expected_parents == tree.parents()));
        REQUIRE((expected_levels == levels));
    }

    TEST_CASE("single linkage clustering", "[binary_partition_tree]") {
        long size = 100;
        auto graph = get_4_adjacency_graph({size, size});
        array_1d<double> edge_weights = xt::random::rand<double>({num_edges(graph)});
        auto res = binary_partition_tree_min_linkage(graph, edge_weights);
        auto &tree = res.tree;

        auto res2 = hg::bpt_canonical(graph, edge_weights);
        auto &tree2 = res2.tree;

        REQUIRE(hg::test_tree_isomorphism(tree, tree2));
    }

    TEST_CASE("complete linkage clustering multiple edges", "[binary_partition_tree]") {
        auto graph = get_4_adjacency_graph({3, 3});
        index_t ne = (index_t)num_edges(graph);
        for (index_t i = 0; i < ne; i++) {
            auto e = edge_from_index(i, graph);
            add_edge(source(e, graph), target(e, graph), graph);
        }
        array_1d<double> edge_weights({1, 8, 2, 10, 15, 3, 11, 4, 12, 13, 5, 6,
                                       1, 8, 2, 10, 15, 3, 11, 4, 12, 13, 5, 6});


        auto res = binary_partition_tree_complete_linkage(graph, edge_weights);
        auto &tree = res.tree;
        auto &levels = res.altitudes;

        array_1d<index_t> expected_parents({9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 16, 12, 15, 14, 15, 16, 16});
        array_1d<double> expected_levels({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 13, 15});

        REQUIRE((expected_parents == tree.parents()));
        REQUIRE((expected_levels == levels));
    }

    TEST_CASE("complete linkage clustering simple", "[binary_partition_tree]") {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<double> edge_weights({1, 8, 2, 10, 15, 3, 11, 4, 12, 13, 5, 6});
        auto res = binary_partition_tree_complete_linkage(graph, edge_weights);
        auto &tree = res.tree;
        auto &levels = res.altitudes;

        array_1d<index_t> expected_parents({9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 16, 12, 15, 14, 15, 16, 16});
        array_1d<double> expected_levels({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 13, 15});

        REQUIRE((expected_parents == tree.parents()));
        REQUIRE((expected_levels == levels));
    }

    TEST_CASE("average linkage clustering simple", "[binary_partition_tree]") {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<double> edge_weights({1, 7, 2, 10, 16, 3, 11, 4, 12, 14, 5, 6});
        array_1d<double> edge_weight_weights({7, 1, 7, 3, 2, 8, 2, 2, 2, 1, 5, 9});
        auto res = binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights);
        auto &tree = res.tree;
        auto &levels = res.altitudes;

        array_1d<index_t> expected_parents({9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 15, 12, 15, 14, 16, 16, 16});
        array_1d<double> expected_levels({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 11.5, 12});

        REQUIRE((expected_parents == tree.parents()));
        REQUIRE((expected_levels == levels));
    }

    TEST_CASE("ward linkage clustering", "[binary_partition_tree]") {
        ugraph graph(5);

        array_1d<index_t> sources{0, 0, 0, 1, 2, 2, 3};
        array_1d<index_t> targets{1, 2, 3, 2, 3, 4, 4};
        add_edges(sources, targets, graph);
        array_2d<double> vertex_centroids{
                {0,  0},
                {1,  1},
                {1,  3},
                {-3, 4},
                {-1, 5}};

        array_1d<double> vertex_sizes{1,
                                      1,
                                      1,
                                      2,
                                      1};

        auto res = binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes);

        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> expected_parents{5, 5, 7, 6, 6, 7, 8, 8, 8};
        array_1d<double> expected_altitudes{0., 0., 0., 0., 0.,
                                            1., 3.333333, 4.333333, 27.};
        REQUIRE((expected_parents == parents(tree)));
        REQUIRE(xt::allclose(expected_altitudes, altitudes));
    }

    TEST_CASE("ward linkage non increasing", "[binary_partition_tree]") {
        ugraph graph(3);

        array_1d<index_t> sources{0, 1};
        array_1d<index_t> targets{2, 2};
        add_edges(sources, targets, graph);
        array_2d<double> vertex_centroids{{0},
                                          {1},
                                          {5}};

        array_1d<double> vertex_sizes{1, 1, 1};

        auto res = binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes);

        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> expected_parents{4, 3, 3, 4, 4};
        array_1d<double> expected_altitudes{0., 0., 0., 8, 8};
        REQUIRE((expected_parents == parents(tree)));
        REQUIRE(xt::allclose(expected_altitudes, altitudes));

        auto res2 = binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes, "none");

        auto &tree2 = res2.tree;
        auto &altitudes2 = res2.altitudes;

        array_1d<index_t> expected_parents2{4, 3, 3, 4, 4};
        array_1d<double> expected_altitudes2{0., 0., 0., 8, 6};
        REQUIRE((expected_parents2 == parents(tree2)));
        REQUIRE(xt::allclose(expected_altitudes2, altitudes2));
    }

    TEST_CASE("average linkage clustering", "[binary_partition_tree]") {
        ugraph graph(10);
        array_1d<index_t> sources{0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 7, 7};
        array_1d<index_t> targets{3, 6, 4, 2, 5, 3, 6, 9, 7, 3, 8, 5, 9, 4, 6, 9, 7, 8, 6, 9, 8};
        add_edges(sources, targets, graph);
        array_1d<double> edge_weights{
                0.87580029, 0.60123697, 0.79924759, 0.74221387, 0.75418382, 0.66159356,
                1.31856839, 0.76080612, 1.08881471, 0.98557615, 0.61454158, 0.50913424,
                0.63556478, 0.64684775, 1.14865302, 0.81741018, 2.1591071, 0.60563004,
                2.06636665, 1.35617725, 0.83085949
        };
        array_1d<double> edge_weight_weights = xt::ones_like(edge_weights);
        auto res = binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights);

        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> expected_parents{11, 14, 10, 13, 15, 10, 11, 18, 12, 13, 12, 17, 16, 14, 15, 16, 17, 18, 18};
        array_1d<double> expected_altitudes{
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.509134, 0.601237, 0.610086,
                0.635565, 0.661594, 0.732129, 0.810695, 1.241727, 1.35874
        };
        REQUIRE((expected_parents == parents(tree)));
        REQUIRE(xt::allclose(expected_altitudes, altitudes));
    }

    TEST_CASE("exponential linkage clustering", "[binary_partition_tree]") {
        ugraph graph(5);
        array_1d<index_t> sources{0, 0, 1, 2, 2, 3};
        array_1d<index_t> targets{1, 2, 4, 3, 4, 4};
        add_edges(sources, targets, graph);

        array_1d<double> edge_weights{1, 3, 5, 2, 4, 6};
        array_1d<double> edge_weight_weights{2, 2, 1, 3, 3, 1};

        auto r = binary_partition_tree_exponential_linkage(graph, edge_weights, -1, edge_weight_weights);
        auto &tree = r.tree;
        auto &altitudes = r.altitudes;

        array_1d<index_t> ref_parents{5, 5, 6, 6, 8, 7, 7, 8, 8};
        array_1d<double> ref_altitudes{0., 0., 0., 0., 0., 1.,
                                       2., 3., 4.182275};
        
        REQUIRE(tree.parents() == ref_parents);
        // large tolerance due to numerical instabilities, especially with fastmath
        REQUIRE(xt::allclose(altitudes, ref_altitudes, 1e-5, 1e-1));
    }

    TEST_CASE("exponential linkage clustering equiv", "[binary_partition_tree]") {
        xt::random::seed(10);
        auto g = get_4_adjacency_graph({5, 5});
        array_1d<double> edge_weights = xt::random::rand<double>({num_edges(g)});
        array_1d<double> edge_weight_weights = xt::random::randint<int>({num_edges(g)}, 1, 10);

        auto r1 = binary_partition_tree_exponential_linkage(g, edge_weights, 0, edge_weight_weights);
        auto r1_ref = binary_partition_tree_average_linkage(g, edge_weights, edge_weight_weights);

        REQUIRE(r1.tree.parents() == r1_ref.tree.parents());
        REQUIRE(xt::allclose(r1.altitudes, r1_ref.altitudes));

        auto r2 = binary_partition_tree_exponential_linkage(g, edge_weights, 200, edge_weight_weights);
        auto r2_ref = binary_partition_tree_complete_linkage(g, edge_weights);

        REQUIRE(r2.tree.parents() == r2_ref.tree.parents());

        auto r3 = binary_partition_tree_exponential_linkage(g, edge_weights, -600, edge_weight_weights);
        auto r3_ref = binary_partition_tree_min_linkage(g, edge_weights);

        REQUIRE(r3.tree.parents() == r3_ref.tree.parents());
    }

}
