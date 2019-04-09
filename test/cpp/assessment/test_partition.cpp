/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/assessment/partition.hpp"
#include "../test_utils.hpp"

using namespace hg;

namespace assessment_partition {
 
    TEST_CASE("cardinal of intersections", "[assessment_partition]") {
            array_1d<int> candidate{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };
            array_1d<int> gt1{ 0, 0, 1, 1, 1, 2, 2, 3, 3 };
            array_1d<int> gt2{ 0, 0, 0, 0, 1, 1, 1, 1, 1 };

            auto r = card_intersections(candidate, xt::stack(xt::xtuple(gt1, gt2)));

            std::vector<array_2d<index_t>> ref =
            {
                {
                    { 2, 1, 0, 0 },
                    { 0, 2, 1, 0 },
                    { 0, 0, 1, 2 }
                },
                {
                    { 3, 0 },
                    { 1, 2 },
                    { 0, 3 }
                }
            };
            REQUIRE(r.size() == ref.size());
            for (int i = 0; i < (int)ref.size(); i++) {
                REQUIRE((ref[i] == r[i]));
            }
    }

    TEST_CASE("assess partition BCE", "[assessment_partition]") {
            array_1d<int> candidate{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };
            array_1d<int> gt1{ 0, 0, 1, 1, 1, 2, 2, 3, 3 };
            array_1d<int> gt2{ 0, 0, 0, 0, 1, 1, 1, 1, 1 };

            auto bce1 = assess_partition(candidate, gt1, scorer_partition_BCE());
            double s1 = 5.0 / 9;
            REQUIRE(s1 == bce1);

            auto bce2 = assess_partition(candidate, gt2, scorer_partition_BCE());
            double s2 = (9.0 / 4 + 1.0 / 4 + 4.0 / 5 + 9.0 / 5) / 9;
            REQUIRE(s2 == bce2);

            auto bce = assess_partition(candidate, xt::stack(xt::xtuple(gt1, gt2)), scorer_partition_BCE());
            REQUIRE((s1 + s2) / 2.0 == bce);
    }

    TEST_CASE("assess partition DHaming", "[assessment_partition]") {
            array_1d<int> candidate{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };
            array_1d<int> gt1{ 0, 0, 1, 1, 1, 2, 2, 3, 3 };
            array_1d<int> gt2{ 0, 0, 0, 0, 1, 1, 1, 1, 1 };

            auto dh1 = assess_partition(candidate, gt1, scorer_partition_DHamming());
            double s1 = 6.0 / 9;
            REQUIRE(s1 == dh1);

            auto dh2 = assess_partition(candidate, gt2, scorer_partition_DHamming());
            double s2 = 8.0 / 9;
            REQUIRE(s2 == dh2);

            auto dh = assess_partition(candidate, xt::stack(xt::xtuple(gt1, gt2)), scorer_partition_DHamming());
            REQUIRE(almost_equal((s1 + s2) / 2.0, dh));
    }

    TEST_CASE("assess partition DCovering", "[assessment_partition]") {
            array_1d<int> candidate{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };
            array_1d<int> gt1{ 0, 0, 1, 1, 1, 2, 2, 3, 3 };
            array_1d<int> gt2{ 0, 0, 0, 0, 1, 1, 1, 1, 1 };

            auto cov1 = assess_partition(candidate, gt1, scorer_partition_DCovering());
            double s1 = (2.0 + 1.5 + 2.0) / 9;
            REQUIRE(s1 == cov1);

            auto cov2 = assess_partition(candidate, gt2, scorer_partition_DCovering());
            double s2 = (9.0 / 4 + 1 + 9.0 / 5) / 9;
            REQUIRE(s2 == cov2);

            auto cov = assess_partition(candidate, xt::stack(xt::xtuple(gt1, gt2)), scorer_partition_DCovering());
            REQUIRE(almost_equal((s1 + s2) / 2.0, cov));
    }

}
