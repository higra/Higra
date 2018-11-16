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

    BOOST_AUTO_TEST_CASE(test_assess_partition_BCE) {
        array_1d<int> candidate{0, 0, 0, 1, 1, 1, 2, 2, 2};
        array_1d<int> gt1{0, 0, 1, 1, 1, 2, 2, 3, 3};
        array_1d<int> gt2{0, 0, 0, 0, 1, 1, 1, 1, 1};

        auto bce1 = assess_partition(candidate, gt1, scorer_partition_BCE());
        double s1 = 5.0 / 9;
        BOOST_CHECK(s1 == bce1);

        auto bce2 = assess_partition(candidate, gt2, scorer_partition_BCE());
        double s2 = (9.0 / 4 + 1.0 / 4 + 4.0 / 5 + 9.0 / 5) / 9;
        BOOST_CHECK(s2 == bce2);

        auto bce = assess_partition(candidate, xt::stack(xt::xtuple(gt1, gt2)), scorer_partition_BCE());
        BOOST_CHECK((s1 + s2) / 2.0 == bce);
    }

    BOOST_AUTO_TEST_CASE(test_assess_partition_DHaming) {
        array_1d<int> candidate{0, 0, 0, 1, 1, 1, 2, 2, 2};
        array_1d<int> gt1{0, 0, 1, 1, 1, 2, 2, 3, 3};
        array_1d<int> gt2{0, 0, 0, 0, 1, 1, 1, 1, 1};

        auto dh1 = assess_partition(candidate, gt1, scorer_partition_DHamming());
        double s1 = 6.0 / 9;
        BOOST_CHECK(s1 == dh1);

        auto dh2 = assess_partition(candidate, gt2, scorer_partition_DHamming());
        double s2 = 8.0 / 9;
        BOOST_CHECK(s2 == dh2);

        auto dh = assess_partition(candidate, xt::stack(xt::xtuple(gt1, gt2)), scorer_partition_DHamming());
        BOOST_CHECK(almost_equal((s1 + s2) / 2.0, dh));
    }

    BOOST_AUTO_TEST_CASE(test_assess_partition_DCovering) {
        array_1d<int> candidate{0, 0, 0, 1, 1, 1, 2, 2, 2};
        array_1d<int> gt1{0, 0, 1, 1, 1, 2, 2, 3, 3};
        array_1d<int> gt2{0, 0, 0, 0, 1, 1, 1, 1, 1};

        auto cov1 = assess_partition(candidate, gt1, scorer_partition_DCovering());
        double s1 = (2.0 + 1.5 + 2.0) / 9;
        BOOST_CHECK(s1 == cov1);

        auto cov2 = assess_partition(candidate, gt2, scorer_partition_DCovering());
        double s2 = (9.0 / 4 + 1 + 9.0 / 5) / 9;
        BOOST_CHECK(s2 == cov2);

        auto cov = assess_partition(candidate, xt::stack(xt::xtuple(gt1, gt2)), scorer_partition_DCovering());
        BOOST_CHECK(almost_equal((s1 + s2) / 2.0, cov));
    }


BOOST_AUTO_TEST_SUITE_END();