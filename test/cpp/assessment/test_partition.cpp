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

            auto r = card_intersections(candidate, hg_test::stack(gt1, gt2));

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

            auto bce = assess_partition(candidate, hg_test::stack(gt1, gt2), scorer_partition_BCE());
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

            auto dh = assess_partition(candidate, hg_test::stack(gt1, gt2), scorer_partition_DHamming());
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

            auto cov = assess_partition(candidate, hg_test::stack(gt1, gt2), scorer_partition_DCovering());
            REQUIRE(almost_equal((s1 + s2) / 2.0, cov));
    }

    TEST_CASE("assess partition non contiguous labels", "[assessment_partition]") {
        // same values as the Python test test_assess_partition_measures_non_contiguous_labels
        array_1d<int> candidate{0, 0, 1, 1, 1, 2, 2, 0, 2, 1, 1, 2};
        array_1d<int> gt1{1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 1, 1}; // labels 0 and 2 unused
        array_1d<int> gt2{0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2}; // label 1 unused
        array_2d<int> gts = hg_test::stack(gt1, gt2);

        auto check = [&](const auto &scorer, double ref1, double ref2, double ref12) {
            REQUIRE(almost_equal(assess_partition(candidate, gt1, scorer), ref1, 1e-14));
            REQUIRE(almost_equal(assess_partition(candidate, gt2, scorer), ref2, 1e-14));
            REQUIRE(almost_equal(assess_partition(candidate, gts, scorer), ref12, 1e-14));
            // ground truths given as a view (rows 1 and 2 of a larger array)
            array_1d<int> gt0 = hg::zeros<int>({12});
            array_2d<int> gts_wide = hg_test::stack<int>({gt0, gt1, gt2, gt0});
            REQUIRE(almost_equal(assess_partition(candidate, gts_wide.slice(1, 3), scorer), ref12, 1e-14));
        };
        check(scorer_partition_BCE(), 0.3652777777777778, 0.4666666666666666, 0.4159722222222222);
        check(scorer_partition_DHamming(), 0.5, 0.75, 0.625);
        check(scorer_partition_DCovering(), 0.3134920634920635, 0.4523809523809524, 0.3829365079365079);
    }

    TEST_CASE("partition scorers on views and gathered matrices", "[assessment_partition]") {
        auto card_intersections = hg::card_intersections<double>(array_1d<int>{0, 0, 1, 1, 1, 2, 2, 0, 2, 1, 1, 2},
                                                                 array_1d<int>{1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 1, 1});
        const array_2d<double> &ci = card_intersections[0];
        // the matrix as rows 1 to 3 of a larger one
        const index_t nr = ci.shape(0), nc = ci.shape(1);
        array_2d<double> big = hg::zeros<double>({nr + 2, nc});
        hg::assign(big.slice(1, nr + 1), ci);
        auto rows_view = big.slice(1, nr + 1);
        // rows gathered in another order, and the same matrix built element by element
        array_1d<index_t> rows{2, 0, 1};
        auto gathered = hg::take(ci, rows);
        array_2d<double> gathered_copy = hg::empty<double>({3, nc});
        for (index_t i = 0; i < 3; i++) {
            for (index_t j = 0; j < nc; j++) {
                gathered_copy(i, j) = ci(rows(i), j);
            }
        }

        auto check = [&](const auto &scorer, double ref) {
            REQUIRE(almost_equal(scorer.score(ci), ref, 1e-14));
            REQUIRE(scorer.score(rows_view) == scorer.score(ci));
            REQUIRE(scorer.score(ci.view()) == scorer.score(ci));
            REQUIRE(scorer.score(gathered) == scorer.score(gathered_copy));
        };
        check(scorer_partition_BCE(), 0.3652777777777778);
        check(scorer_partition_DHamming(), 0.5);
        check(scorer_partition_DCovering(), 0.3134920634920635);
    }
}
