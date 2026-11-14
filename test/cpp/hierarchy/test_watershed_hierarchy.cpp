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
#include "higra/hierarchy/watershed_hierarchy.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/algo/tree.hpp"

namespace watershed_hierarchy {

    using namespace hg;
    using namespace std;


    TEST_CASE("watershed hierarchy by area", "[watershed_hierarchy]") {

        auto g = hg::get_4_adjacency_graph({2, 3});
        array_1d<int> edge_weights{1, 0, 1, 0, 0, 0, 1};
        // x1x1x
        // 0 0 0
        // x0x1x

        auto res = watershed_hierarchy_by_area(g, edge_weights);
        auto &t = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> ref_parents{6, 7, 8, 6, 7, 8, 9, 9, 10, 10, 10 };

        array_1d<int> ref_altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};

        REQUIRE((ref_parents == t.parents()));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("watershed hierarchy by area 2", "[watershed_hierarchy]") {

        auto g = hg::get_4_adjacency_graph({1, 19});
        array_1d<int> edge_weights{0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0};

        auto res = watershed_hierarchy_by_area(g, edge_weights);
        auto &t = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> ref_parents{19, 19, 20, 20, 21, 22, 22, 23, 24, 25, 26, 27, 27, 28, 29, 30, 31, 31, 32,
                                      33, 21, 33, 23, 24, 25, 26, 34, 28, 29, 30, 35, 32, 35, 34, 36, 36, 36};
        array_1d<int> ref_altitudes  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 5};

        REQUIRE((ref_parents == t.parents()));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("watershed hierarchy by area 3", "[watershed_hierarchy]") {

        auto g = hg::get_4_adjacency_graph({3, 3});
        array_1d<double> edge_weights{0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0};

        auto res = watershed_hierarchy_by_area(g, edge_weights);

        REQUIRE((hg::sum(res.altitudes) == 0));
    }

    TEST_CASE("watershed hierarchy by dynamics", "[watershed_hierarchy]") {

        auto g = hg::get_4_adjacency_graph({1, 7});
        array_1d<int> edge_weights{1, 4, 1, 0, 10, 8};

        auto res = watershed_hierarchy_by_dynamics(g, edge_weights);
        auto &t = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> ref_parents{8, 8, 9, 7, 7, 10, 10,
                                      9, 12, 11, 11, 12, 12};
        array_1d<int> ref_altitudes{0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 2, 3};

        REQUIRE((ref_parents == t.parents()));
        REQUIRE((altitudes == ref_altitudes));
    }


    TEST_CASE("watershed hierarchy by volume", "[watershed_hierarchy]") {

        auto g = hg::get_4_adjacency_graph({1, 7});
        array_1d<int> edge_weights{1, 4, 1, 0, 10, 8};

        auto res = watershed_hierarchy_by_volume(g, edge_weights);
        auto &t = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> ref_parents{8, 8, 9, 7, 7, 10, 10,
                                      9, 12, 11, 11, 12, 12};
        array_1d<int> ref_altitudes{0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 4, 6};

        REQUIRE((ref_parents == t.parents()));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("watershed hierarchy by minima ordering", "[watershed_hierarchy]") {

        auto g = hg::get_4_adjacency_graph({1, 7});
        array_1d<int> edge_weights{1, 4, 1, 0, 10, 8};
        //same as dynamics
        array_1d<int> minima_ranking{2, 2, 0, 3, 3, 1, 1};

        auto res = watershed_hierarchy_by_minima_ordering(g, edge_weights, minima_ranking);
        auto &t = res.tree;
        auto &altitudes = res.altitudes;

        array_1d<index_t> ref_parents{8, 8, 9, 7, 7, 10, 10,
                                      9, 12, 11, 11, 12, 12};
        array_1d<int> ref_altitudes{0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 1, 2};

        REQUIRE((ref_parents == t.parents()));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("watershed hierarchy by area and volume default vertex area", "[watershed_hierarchy]") {
        auto graph = get_4_adjacency_graph({4, 5});
        array_1d<double> edge_weights{1, 5, 2, 0, 3, 6, 4, 2, 8, 1, 7, 3, 2, 5, 9, 1, 4, 6, 3, 2, 7, 1, 8, 4, 2,
                                      6, 3, 5, 1, 9, 2};
        array_1d<index_t> ones = hg::ones<index_t>({num_vertices(graph)});

        auto res1 = watershed_hierarchy_by_area(graph, edge_weights);
        auto ref1 = watershed_hierarchy_by_area(graph, edge_weights, ones);
        REQUIRE((res1.tree.parents() == ref1.tree.parents()));
        REQUIRE((res1.altitudes == ref1.altitudes));

        auto res2 = watershed_hierarchy_by_volume(graph, edge_weights);
        auto ref2 = watershed_hierarchy_by_volume(graph, edge_weights, ones);
        REQUIRE((res2.tree.parents() == ref2.tree.parents()));
        REQUIRE((res2.altitudes == ref2.altitudes));
    }
}
