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

namespace test_partition {

    using namespace hg;
 
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

    TEMPLATE_TEST_CASE("partition scorers on small unsigned count matrices", "[assessment_partition]",
                       uint8_t, uint16_t) {
        // row and column sums exceed the element type: same scores as with int64_t
        std::vector<array_2d<int64_t>> matrices{
                {{200, 100}, {50, 200}},
                {{255, 255}, {255, 255}},
                {{200, 0},   {0, 250}},
                {{3, 0, 1},  {0, 4, 0}, {2, 0, 5}}};
        if (std::numeric_limits<TestType>::max() > 255) {
            matrices.push_back({{40000, 30000}, {1000, 40000}});
        }
        for (const auto &m: matrices) {
            array_2d<TestType> mt = hg::cast<TestType>(m);
            REQUIRE(scorer_partition_BCE::score(mt.view()) == scorer_partition_BCE::score(m.view()));
            REQUIRE(scorer_partition_DCovering::score(mt.view()) == scorer_partition_DCovering::score(m.view()));
            REQUIRE(scorer_partition_DHamming::score(mt.view()) == scorer_partition_DHamming::score(m.view()));
        }
        // integer division, as for other integer types
        array_2d<TestType> diagonal{{200, 0}, {0, 250}};
        REQUIRE(scorer_partition_DHamming::score(diagonal.view()) == 1);
        array_2d<TestType> m{{200, 100}, {50, 200}};
        REQUIRE(scorer_partition_BCE::score(m.view()) == 0);
        REQUIRE(scorer_partition_DCovering::score(m.view()) == 0);
        REQUIRE(scorer_partition_DHamming::score(m.view()) == 0);
        // DHamming returns the type of T / T
        static_assert(std::is_same<decltype(scorer_partition_DHamming::score(m.view())), int>::value, "");
    }

    TEST_CASE("partition scorers result values and types", "[assessment_partition]") {
        // integer types: integer division
        array_2d<int32_t> mi{{200, 100}, {50, 200}};
        array_2d<int64_t> ml{{3, 0, 1}, {0, 4, 0}, {2, 0, 5}};
        array_2d<uint64_t> mu{{200, 0}, {0, 250}};
        REQUIRE(scorer_partition_BCE::score(mi.view()) == 0);
        REQUIRE(scorer_partition_DCovering::score(mi.view()) == 0);
        REQUIRE(scorer_partition_DHamming::score(mi.view()) == 0);
        REQUIRE(scorer_partition_BCE::score(ml.view()) == 4.0 / 15);
        REQUIRE(scorer_partition_DCovering::score(ml.view()) == 4.0 / 15);
        REQUIRE(scorer_partition_DHamming::score(ml.view()) == 0);
        REQUIRE(scorer_partition_DHamming::score(mu.view()) == 1);
        static_assert(std::is_same<decltype(scorer_partition_DHamming::score(mi.view())), int32_t>::value, "");
        static_assert(std::is_same<decltype(scorer_partition_DHamming::score(ml.view())), int64_t>::value, "");
        static_assert(std::is_same<decltype(scorer_partition_DHamming::score(mu.view())), uint64_t>::value, "");
        static_assert(std::is_same<decltype(scorer_partition_BCE::score(mi.view())), double>::value, "");
        static_assert(std::is_same<decltype(scorer_partition_DCovering::score(mi.view())), double>::value, "");

        // floating point types: accumulated in their own type
        array_2d<double> md{{200, 100}, {50, 200}};
        array_2d<float> mf{{200, 100}, {50, 200}};
        REQUIRE(scorer_partition_BCE::score(md.view()) == 0.56363636363636349);
        REQUIRE(scorer_partition_DCovering::score(md.view()) == 0.5714285714285714);
        REQUIRE(scorer_partition_DHamming::score(md.view()) == 0.72727272727272729);
        REQUIRE(scorer_partition_BCE::score(mf.view()) == 0.56363641912286933);
        REQUIRE(scorer_partition_DCovering::score(mf.view()) == 0.57142855557528405);
        REQUIRE(scorer_partition_DHamming::score(mf.view()) == 0.72727274894714355f);
        static_assert(std::is_same<decltype(scorer_partition_DHamming::score(mf.view())), float>::value, "");
    }

    TEMPLATE_TEST_CASE("cardinal of intersections of empty unsigned labelisations", "[assessment_partition]",
                       uint8_t, uint64_t) {
        // one region (the lowest value of the type plus one) for empty labelisations
        array_1d<TestType> empty = hg::empty<TestType>({0});
        auto r = card_intersections(empty, empty);
        REQUIRE(r.size() == 1);
        REQUIRE((r[0] == array_2d<index_t>{{0}}));
    }

    TEMPLATE_TEST_CASE("partition scorers with empty rows or columns", "[assessment_partition]",
                       int64_t, uint8_t, double) {
        // the same partition, with an unused candidate label, an unused ground-truth label, or both
        std::vector<array_2d<TestType>> matrices{
                {{2, 0}, {0, 0}, {0, 2}},
                {{2, 0, 0}, {0, 0, 2}},
                {{2, 0, 0}, {0, 0, 0}, {0, 0, 2}}};
        for (const auto &m: matrices) {
            REQUIRE(scorer_partition_BCE::score(m.view()) == 1);
            REQUIRE(scorer_partition_DCovering::score(m.view()) == 1);
            REQUIRE(scorer_partition_DHamming::score(m.view()) == 1);
        }

        // empty rows and columns do not change the score of a non-trivial matrix
        array_2d<TestType> base{{3, 1}, {1, 5}};
        array_2d<TestType> padded{{3, 0, 1}, {0, 0, 0}, {1, 0, 5}};
        REQUIRE(scorer_partition_BCE::score(padded.view()) == scorer_partition_BCE::score(base.view()));
        REQUIRE(scorer_partition_DCovering::score(padded.view()) == scorer_partition_DCovering::score(base.view()));
        REQUIRE(scorer_partition_DHamming::score(padded.view()) == scorer_partition_DHamming::score(base.view()));

        // a matrix whose total is 0 is rejected, including matrices without rows or columns
        for (const auto &z: {array_2d<TestType>{{0, 0}, {0, 0}}, array_2d<TestType>{{0}},
                             array_2d<TestType>::from_shape({2, 0}), array_2d<TestType>::from_shape({0, 2}),
                             array_2d<TestType>::from_shape({0, 0})}) {
            REQUIRE_THROWS(scorer_partition_BCE::score(z.view()));
            REQUIRE_THROWS(scorer_partition_DCovering::score(z.view()));
            REQUIRE_THROWS(scorer_partition_DHamming::score(z.view()));
        }
    }

    TEST_CASE("assess partition degenerate inputs", "[assessment_partition]") {
        array_1d<int> candidate{0, 0, 2, 2};
        array_1d<int> ground_truth{0, 0, 1, 1};
        REQUIRE(assess_partition(candidate, ground_truth, scorer_partition_BCE()) == 1);
        REQUIRE(assess_partition(candidate, ground_truth, scorer_partition_DHamming()) == 1);
        REQUIRE(assess_partition(candidate, ground_truth, scorer_partition_DCovering()) == 1);
        REQUIRE(assess_partition(ground_truth, candidate, scorer_partition_BCE()) == 1);
        REQUIRE(assess_partition(ground_truth, candidate, scorer_partition_DHamming()) == 1);
        REQUIRE(assess_partition(ground_truth, candidate, scorer_partition_DCovering()) == 1);

        // several ground truths, one with unused labels
        array_2d<int> ground_truths{{0, 0, 1, 1}, {0, 0, 3, 3}};
        REQUIRE(assess_partition(candidate, ground_truths, scorer_partition_BCE()) == 1);
        REQUIRE(assess_partition(candidate, ground_truths, scorer_partition_DHamming()) == 1);
        REQUIRE(assess_partition(candidate, ground_truths, scorer_partition_DCovering()) == 1);

        // no ground truth
        array_2d<int> no_ground_truth = array_2d<int>::from_shape({0, 4});
        REQUIRE_THROWS(assess_partition(candidate, no_ground_truth, scorer_partition_BCE()));
        REQUIRE_THROWS(assess_partition(candidate, no_ground_truth, scorer_partition_DHamming()));
        REQUIRE_THROWS(assess_partition(candidate, no_ground_truth, scorer_partition_DCovering()));
        std::vector<array_2d<index_t>> no_matrix;
        REQUIRE_THROWS(assess_partition(no_matrix, scorer_partition_BCE()));
        REQUIRE_THROWS(assess_partition(no_matrix, scorer_partition_DHamming()));
        REQUIRE_THROWS(assess_partition(no_matrix, scorer_partition_DCovering()));

        array_1d<int> empty = hg::empty<int>({0});
        REQUIRE_THROWS(assess_partition(empty, empty, scorer_partition_BCE()));
        REQUIRE_THROWS(assess_partition(empty, empty, scorer_partition_DHamming()));
        REQUIRE_THROWS(assess_partition(empty, empty, scorer_partition_DCovering()));
    }
}
