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
#include "higra/graph.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/algo/horizontal_cuts.hpp"

using namespace hg;
namespace test_horizontal_cuts {
    TEST_CASE("horizontal cut explorer constructor asserts", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{4, 4, 5, 5, 6, 6, 6}
        };
        array_1d<int> altitudes{1, 0, 0, 0, 2, 3, 4};
        REQUIRE_THROWS_AS(make_horizontal_cut_explorer(tree, altitudes), std::runtime_error);

        array_1d<int> altitudes2{0, 0, 0, 0, 2, 3, -1};
        REQUIRE_THROWS_AS(make_horizontal_cut_explorer(tree, altitudes2), std::runtime_error);
    }

    TEST_CASE("horizontal cut explorer indexed accessor", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        REQUIRE(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {18},
                {17, 13, 14},
                {11, 16, 13, 14},
                {0,  1,  2,  3, 4, 5, 13, 9, 10}
        };

        std::vector<int> alt_cuts{3, 2, 1, 0};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_index(i);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
            REQUIRE(c.altitude == alt_cuts[i]);
        }
    }

    TEST_CASE("horizontal cut explorer indexed accessor on sorted tree", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{5, 5, 5, 6, 6, 7, 7, 7}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 1, 2, 3};

        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        REQUIRE(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {7},
                {5, 6},
                {5, 3, 4},
                {0, 1, 2, 3, 4}
        };

        std::vector<int> alt_cuts{3, 2, 1, 0};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_index(i);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
            REQUIRE(c.altitude == alt_cuts[i]);
        }
    }

    TEST_CASE("horizontal cut explorer altitudes accessor", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        REQUIRE(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {18},
                {17, 13, 14},
                {11, 16, 13, 14},
                {0,  1,  2,  3, 4, 5, 13, 9, 10}
        };

        std::vector<int> alt_cuts{3, 2, 1, 0};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_altitude(alt_cuts[i]);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
            REQUIRE(c.altitude == alt_cuts[i]);
        }
    }

    TEST_CASE("horizontal cut explorer altitudes accessor sorted tree", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{5, 5, 5, 6, 6, 7, 7, 7}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        REQUIRE(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {7},
                {5, 6},
                {5, 3, 4},
                {0, 1, 2, 3, 4}
        };

        std::vector<int> alt_cuts{3, 2, 1, 0};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_altitude(alt_cuts[i]);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
            REQUIRE(c.altitude == alt_cuts[i]);
        }
    }

    TEST_CASE("horizontal cut explorer number of regions accessor", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        REQUIRE(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {18},
                {17, 13, 14},
                {11, 16, 13, 14},
                {0,  1,  2,  3, 4, 5, 13, 9, 10}
        };

        std::vector<int> k_cuts{1, 3, 4, 9};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_num_regions(k_cuts[i]);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
        }

        std::vector<int> k_cuts2{1, 2, 4, 5};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_num_regions(k_cuts2[i]);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
        }
    }

    TEST_CASE("horizontal cut explorer number of regions accessor sorted tree", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{5, 5, 5, 6, 6, 7, 7, 7}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        REQUIRE(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {7},
                {5, 6},
                {5, 3, 4},
                {0, 1, 2, 3, 4}
        };

        std::vector<int> k_cuts{1, 2, 3, 4};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_num_regions(k_cuts[i]);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
        }

        std::vector<int> k_cuts2{1, 2, 3, 5};

        for (index_t i = 0; i < (index_t) hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_num_regions(k_cuts2[i]);
            REQUIRE(vectorSame(c.nodes, cut_nodes[i]));
        }
    }

    TEST_CASE("horizontal cut explorer consistency", "[horizontal_cuts]") {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        for (index_t i = 0; i <= 3; i++) {
            auto c = hch.horizontal_cut_from_altitude((int)i);
            auto r1 = c.labelisation_leaves(tree);
            auto r2 = labelisation_horizontal_cut_from_threshold(tree, altitudes, i);
            REQUIRE((r1 == r2));
        }

    }

    TEST_CASE("horizontal cut nodes functions", "[horizontal_cuts]") {

        auto g = get_4_adjacency_graph({1, 11});
        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        auto c = hch.horizontal_cut_from_num_regions(3);

        auto lbls = c.labelisation_leaves(tree);
        array_1d<index_t> ref_lbls{17, 17, 17, 17, 17, 17, 13, 13, 13, 14, 14};
        REQUIRE((lbls == ref_lbls));

        auto rec = c.reconstruct_leaf_data(tree, altitudes);
        array_1d<int> ref_rec{2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1};
        REQUIRE((rec == ref_rec));

        auto cut = c.graph_cut(tree, g);
        array_1d<int> ref_cut{0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
        REQUIRE((cut == ref_cut));
    }
}

