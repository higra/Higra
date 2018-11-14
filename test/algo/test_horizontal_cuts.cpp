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
#include "../test_utils.hpp"
#include "higra/graph.hpp"
#include "higra/algo/horizontal_cuts.hpp"


using namespace hg;

BOOST_AUTO_TEST_SUITE(test_horinzontal_cuts);


    BOOST_AUTO_TEST_CASE(test_horizontal_cut_explorer_indexed) {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        BOOST_CHECK(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {18},
                {17, 13, 14},
                {11, 16, 13, 14},
                {0,  1,  2,  3, 4, 5, 13, 9, 10}
        };

        std::vector<int> alt_cuts{3, 2, 1, 0};

        for (index_t i = 0; i < hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_index(i);
            BOOST_CHECK(vectorSame(c.nodes, cut_nodes[i]));
            BOOST_CHECK(c.altitude == alt_cuts[i]);
        }
    }

    BOOST_AUTO_TEST_CASE(test_horizontal_cut_explorer_altitudes) {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        BOOST_CHECK(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {18},
                {17, 13, 14},
                {11, 16, 13, 14},
                {0,  1,  2,  3, 4, 5, 13, 9, 10}
        };

        std::vector<int> alt_cuts{3, 2, 1, 0};

        for (index_t i = 0; i < hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_altitude(alt_cuts[i]);
            BOOST_CHECK(vectorSame(c.nodes, cut_nodes[i]));
            BOOST_CHECK(c.altitude == alt_cuts[i]);
        }
    }

    BOOST_AUTO_TEST_CASE(test_horizontal_cut_explorer_num_regions) {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        BOOST_CHECK(hch.num_cuts() == 4);

        std::vector<array_1d<index_t>> cut_nodes{
                {18},
                {17, 13, 14},
                {11, 16, 13, 14},
                {0,  1,  2,  3, 4, 5, 13, 9, 10}
        };

        std::vector<int> k_cuts{1, 3, 4, 9};

        for (index_t i = 0; i < hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_num_regions(k_cuts[i]);
            BOOST_CHECK(vectorSame(c.nodes, cut_nodes[i]));
        }

        std::vector<int> k_cuts2{1, 2, 4, 5};

        for (index_t i = 0; i < hch.num_cuts(); i++) {
            auto c = hch.horizontal_cut_from_num_regions(k_cuts2[i]);
            BOOST_CHECK(vectorSame(c.nodes, cut_nodes[i]));
        }
    }

    BOOST_AUTO_TEST_CASE(test_horizontal_cut_explorer_consistency) {

        hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
        };
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        auto hch = make_horizontal_cut_explorer(tree, altitudes);

        for (index_t i = 0; i <= 3; i++) {
            auto c = hch.horizontal_cut_from_altitude(i);
            auto r1 = c.labelisation_leaves();
            auto r2 = labelisation_horizontal_cut_from_threshold(tree, altitudes, i);
            BOOST_CHECK(r1 == r2);
        }

    }


BOOST_AUTO_TEST_SUITE_END();