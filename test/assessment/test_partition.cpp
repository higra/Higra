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
#include "higra/assessment/partition.hpp"
#include "../test_utils.hpp"

using namespace hg;

BOOST_AUTO_TEST_SUITE(test_assessment_partition);

    BOOST_AUTO_TEST_CASE(test_card_intersection_parition) {
        array_1d<int> candidate{0, 0, 0, 1, 1, 1, 2, 2, 2};
        array_1d<int> gt1{0, 0, 1, 1, 1, 2, 2, 3, 3};
        array_1d<int> gt2{0, 0, 0, 0, 1, 1, 1, 1, 1};

        auto r = card_intersections(candidate, xt::stack(xt::xtuple(gt1, gt2)));

        std::vector<array_2d<index_t>> ref =
                {{{2, 1, 0, 0},
                         {0, 2, 1, 0},
                         {0, 0, 1, 2}},
                 {{3, 0},
                         {1, 2},
                         {0, 3}}};
        BOOST_CHECK(r.size() == ref.size());
        for (int i = 0; i < ref.size(); i++) {
            BOOST_CHECK(ref[i] == r[i]);
        }
    }


BOOST_AUTO_TEST_SUITE_END();