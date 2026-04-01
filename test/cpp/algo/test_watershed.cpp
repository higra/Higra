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
#include "higra/algo/watershed.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

namespace test_watershed {

    TEST_CASE("watershed cut simple", "[watershed_cut]") {

        // Fig 4 of Watershed Cuts: Minimum Spanning Forests and the
        // Drop of Water Principle
        // Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 5, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 3, 5, 4, 0, 7, 0, 3, 4, 0};

        auto labels = hg::labelisation_watershed(g, edge_weights);

        array_1d<index_t> expected{1, 1, 1, 2,
                                   1, 1, 2, 2,
                                   1, 1, 3, 3,
                                   1, 1, 3, 3};
        REQUIRE((labels == expected));
    }

    TEST_CASE("watershed cut simple 2", "[watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({3, 3});
        array_1d<int> edge_weights{1, 1, 0, 0, 0, 1, 0, 0, 2, 2, 0, 2};

        auto labels = hg::labelisation_watershed(g, edge_weights);

        array_1d<index_t> expected{1, 1, 1,
                                   2, 1, 1,
                                   2, 2, 1};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed 1", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0};
        array_1d<int> seeds{1, 1, 9, 9,
                            1, 9, 9, 9,
                            9, 9, 9, 9,
                            1, 1, 2, 2};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds, 9);

        array_1d<int> expected{1, 1, 2, 2,
                               1, 1, 2, 2,
                               1, 1, 2, 2,
                               1, 1, 2, 2};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed 2", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0};
        array_1d<int> seeds{1, 1, 0, 0,
                            1, 0, 0, 0,
                            0, 0, 0, 0,
                            2, 2, 3, 3};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);

        array_1d<int> expected{1, 1, 3, 3,
                               1, 1, 3, 3,
                               2, 2, 3, 3,
                               2, 2, 3, 3};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed split minima", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 4});
        array_1d<int> edge_weights{0, 1, 0, 2, 0, 2, 0, 1, 2, 1};
        /* x0x0x0x
         * 1 2 2 0
         * x1x2x1x */
        array_1d<int> seeds{1, 0, 0, 2,
                            0, 0, 0, 0};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
        /*
        * other possible results:
        * ((1, 1, 2, 2),
        * (1, 1, 2, 2))
        * or
        * ((1, 2, 2, 2),
        * (1, 1, 2, 2)) */
        array_1d<int> expected{1, 1, 1, 2,
                               1, 1, 2, 2};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed disconnected seed", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 3});
        array_1d<int> edge_weights{1, 0, 2, 0, 0, 1, 2};
        /* x1x2x
         * 0 0 0
         * x1x2x */
        array_1d<int> seeds{5, 7, 5,
                            0, 0, 0};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);

        array_1d<int> expected{5, 7, 5,
                               5, 7, 5};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed seed not in minima", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 4});
        array_1d<int> edge_weights{0, 2, 0, 2, 1, 2, 2, 1, 0, 0};
        /* x0x0x1x
         * 2 2 2 2
         * x1x0x0x */
        array_1d<int> seeds{0, 0, 0, 1,
                            2, 0, 0, 0};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);

        array_1d<int> expected{1, 1, 1, 1,
                               2, 2, 2, 2};
        REQUIRE((labels == expected));
    }

    TEST_CASE("incremental watershed cut basic", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        // Add seeds matching seeded watershed test 2
        array_1d<index_t> seed_v1{0, 1, 4};
        array_1d<index_t> seed_l1{1, 1, 1};
        iws.add_seeds(seed_v1, seed_l1);

        array_1d<index_t> seed_v2{12, 13};
        array_1d<index_t> seed_l2{2, 2};
        iws.add_seeds(seed_v2, seed_l2);

        array_1d<index_t> seed_v3{14, 15};
        array_1d<index_t> seed_l3{3, 3};
        iws.add_seeds(seed_v3, seed_l3);

        auto labels = iws.get_labeling();

        // Check that each seed vertex has the correct label
        REQUIRE(labels(0) == 1);
        REQUIRE(labels(1) == 1);
        REQUIRE(labels(4) == 1);
        REQUIRE(labels(12) == 2);
        REQUIRE(labels(13) == 2);
        REQUIRE(labels(14) == 3);
        REQUIRE(labels(15) == 3);

        // All vertices should be labeled (no background)
        for (index_t i = 0; i < 16; i++) {
            REQUIRE(labels(i) != 0);
        }
    }

    TEST_CASE("incremental watershed cut remove seed", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 3});
        array_1d<int> edge_weights{1, 0, 2, 0, 0, 1, 2};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        array_1d<index_t> sv1{0};
        array_1d<index_t> sl1{1};
        iws.add_seeds(sv1, sl1);

        array_1d<index_t> sv2{2};
        array_1d<index_t> sl2{2};
        iws.add_seeds(sv2, sl2);

        array_1d<index_t> sv3{4};
        array_1d<index_t> sl3{3};
        iws.add_seeds(sv3, sl3);

        // Three seeds => three regions
        auto labels3 = iws.get_labeling();
        REQUIRE(labels3(0) == 1);
        REQUIRE(labels3(2) == 2);
        REQUIRE(labels3(4) == 3);

        // Remove seed at vertex 4 => two regions remain
        array_1d<index_t> rm{4};
        iws.remove_seeds(rm);

        auto labels2 = iws.get_labeling();
        REQUIRE(labels2(0) == 1);
        REQUIRE(labels2(2) == 2);
        // Vertex 4 should now be in one of the remaining regions
        REQUIRE((labels2(4) == 1 || labels2(4) == 2));
    }

    TEST_CASE("incremental watershed cut shared labels", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 3});
        array_1d<int> edge_weights{1, 0, 2, 0, 0, 1, 2};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        // Two seeds with the same label (label 5) and one with label 7
        array_1d<index_t> sv{0, 2, 1};
        array_1d<index_t> sl{5, 5, 7};
        iws.add_seeds(sv, sl);

        auto labels = iws.get_labeling();
        REQUIRE(labels(0) == 5);
        REQUIRE(labels(2) == 5);
        REQUIRE(labels(1) == 7);
    }

    TEST_CASE("incremental watershed cut no seeds", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 2});
        array_1d<int> edge_weights{1, 2, 3, 4};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        auto labels = iws.get_labeling();
        for (index_t i = 0; i < 4; i++) {
            REQUIRE(labels(i) == 0);
        }
    }

    TEST_CASE("incremental watershed cut single seed", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 2});
        array_1d<int> edge_weights{1, 2, 3, 4};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        array_1d<index_t> sv{0};
        array_1d<index_t> sl{1};
        iws.add_seeds(sv, sl);

        auto labels = iws.get_labeling();
        for (index_t i = 0; i < 4; i++) {
            REQUIRE(labels(i) == 1);
        }
    }

}
