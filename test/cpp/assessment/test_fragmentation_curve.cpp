/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/assessment/fragmentation_curve.hpp"
#include "higra/assessment/partition.hpp"
#include "higra/image/graph_image.hpp"
#include "../test_utils.hpp"

using namespace hg;

namespace assessment_fragmentation_curve {

    TEST_CASE("fragmentation curve BCE optimal cut", "[fragmentation_curve]") {
            tree t(array_1d<index_t>{ 8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14 });
            array_1d<char> ground_truth{ 0, 0, 1, 1, 1, 2, 2, 2 };

            assesser_fragmentation_optimal_cut assesser(t, ground_truth, optimal_cut_measure::BCE);

            REQUIRE(assesser.optimal_number_of_regions() == 3);
            REQUIRE(almost_equal(assesser.optimal_score(), (2 + 4.0 / 3 + 2.5) / num_leaves(t)));

            auto res = assesser.fragmentation_curve();
            auto &res_scores = res.scores();
            auto &res_k = res.num_regions();
            REQUIRE(3 == res.num_regions_ground_truth());
            REQUIRE(almost_equal(res.optimal_score(), (2 + 4.0 / 3 + 2.5) / num_leaves(t)));
            REQUIRE(res.optimal_number_of_regions() == 3);

            array_1d<double> ref_scores{
                2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3,
                        2 + 4.0 / 3 + 4.0 / 3, 4, 3
            };
            array_1d<size_t> ref_k{ 1, 2, 3, 4, 5, 6, 7, 8 };

            REQUIRE(res_scores == (hg_test::divided(ref_scores, num_leaves(t))));
            REQUIRE(res_k == ref_k);
    }

    TEST_CASE("fragmentation curve BCE optimal cut non contiguous labels", "[fragmentation_curve]") {
        tree t(array_1d<index_t>{ 8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14 });
        array_1d<char> ground_truth{ 1, 1, 2, 2, 2, 5, 5, 5 };

        assesser_fragmentation_optimal_cut assesser(t, ground_truth, optimal_cut_measure::BCE);

        REQUIRE(assesser.optimal_number_of_regions() == 3);
        REQUIRE(almost_equal(assesser.optimal_score(), (2 + 4.0 / 3 + 2.5) / num_leaves(t)));

        auto res = assesser.fragmentation_curve();
        auto &res_scores = res.scores();
        auto &res_k = res.num_regions();
        REQUIRE(3 == res.num_regions_ground_truth());
        REQUIRE(almost_equal(res.optimal_score(), (2 + 4.0 / 3 + 2.5) / num_leaves(t)));
        REQUIRE(res.optimal_number_of_regions() == 3);

        array_1d<double> ref_scores{
                2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3,
                2 + 4.0 / 3 + 4.0 / 3, 4, 3
        };
        array_1d<size_t> ref_k{ 1, 2, 3, 4, 5, 6, 7, 8 };

        REQUIRE(res_scores == (hg_test::divided(ref_scores, num_leaves(t))));
        REQUIRE(res_k == ref_k);
    }

    TEST_CASE("fragmentation curve BCE optimal cut on rag", "[fragmentation_curve]") {
         array_1d<index_t> vertex_map{ 0, 0, 1, 1, 2, 2, 3, 4 };
            tree t(array_1d<index_t>{ 6, 6, 5, 5, 7, 7, 8, 8, 8 });
            array_1d<char> ground_truth{ 0, 0, 1, 1, 1, 2, 2, 2 };

            assesser_fragmentation_optimal_cut assesser(t, ground_truth, optimal_cut_measure::BCE, vertex_map);

            REQUIRE(assesser.optimal_number_of_regions() == 3);
            REQUIRE(almost_equal(assesser.optimal_score(), (2 + 4.0 / 3 + 2.5) / num_leaves(t)));

            auto res = assesser.fragmentation_curve();
            auto &res_scores = res.scores();
            auto &res_k = res.num_regions();
            REQUIRE(3 == res.num_regions_ground_truth());

            array_1d<double> ref_scores{ 2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3 };
            array_1d<size_t> ref_k{ 1, 2, 3, 4, 5 };

            REQUIRE(hg::allclose(res_scores, hg_test::divided(ref_scores, num_leaves(t))));
            REQUIRE(res_k == ref_k);
    }

    TEST_CASE("fragmentation curve DHaming optimal cut", "[fragmentation_curve]") {
            tree t(array_1d<index_t>{ 8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14 });
            array_1d<char> ground_truth{ 0, 0, 1, 1, 1, 2, 2, 2 };

            assesser_fragmentation_optimal_cut assesser(t, ground_truth, optimal_cut_measure::DHamming);

            REQUIRE(assesser.optimal_number_of_regions() == 6);
            REQUIRE(almost_equal(assesser.optimal_score(), (8.0) / num_leaves(t)));

            auto res = assesser.fragmentation_curve();
            auto &res_scores = res.scores();
            auto res_k = res.num_regions_normalized();
            REQUIRE(3 == res.num_regions_ground_truth());

            array_1d<double> ref_scores{ 3, 5, 7, 7, 7, 8, 8, 8 };
            array_1d<double> ref_k{ 1, 2, 3, 4, 5, 6, 7, 8 };

            REQUIRE(res_scores == (hg_test::divided(ref_scores, num_leaves(t))));
            REQUIRE(hg::allclose(res_k, hg_test::divided(ref_k, 3)));
    }

    TEST_CASE("fragmentation curve Covering optimal cut", "[fragmentation_curve]") {
            tree t(array_1d<index_t>{ 8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14 });
            array_1d<char> ground_truth{ 0, 0, 1, 1, 1, 2, 2, 2 };

            assesser_fragmentation_optimal_cut assesser(t, ground_truth, optimal_cut_measure::DCovering);

            REQUIRE(assesser.optimal_number_of_regions() == 3);
            REQUIRE(almost_equal(assesser.optimal_score(), (5 + 4.0 / 3) / num_leaves(t)));

            auto res = assesser.fragmentation_curve();
            auto &res_scores = res.scores();
            auto &res_k = res.num_regions();
            REQUIRE(3 == res.num_regions_ground_truth());

            array_1d<double> ref_scores{ 3, 5, 5 + 4.0 / 3, 5 + 2.0 / 3, 4 + 2.0 / 3, 2 + 8.0 / 3, 4, 3 };
            array_1d<size_t> ref_k{ 1, 2, 3, 4, 5, 6, 7, 8 };

            REQUIRE(hg::allclose(res_scores, (hg_test::divided(ref_scores, num_leaves(t)))));
            REQUIRE(res_k == ref_k);
    }

    TEST_CASE("fragmentation curve BCE optimal cut, optimal partition", "[fragmentation_curve]") {
         tree t(array_1d<index_t>{ 8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14 });
            array_1d<char> ground_truth{ 0, 0, 1, 1, 1, 2, 2, 2 };

            assesser_fragmentation_optimal_cut assesser(t, ground_truth, optimal_cut_measure::BCE);

            std::vector<array_1d<index_t>> optimal_partitions
            {
                { 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 1, 1, 1, 1 },
                { 0, 0, 1, 1, 2, 2, 2, 2 },
                { 0, 0, 1, 1, 2, 2, 2, 3 },
                { 0, 0, 1, 1, 2, 2, 3, 4 },
                { 0, 0, 1, 1, 2, 3, 4, 5 },
                { 0, 0, 1, 2, 3, 4, 5, 6 },
                { 0, 1, 2, 3, 4, 5, 6, 7 }
            };

            REQUIRE(is_in_bijection(optimal_partitions[2], assesser.optimal_partition()));

            for (index_t i = 0; i < (index_t) optimal_partitions.size(); i++) {
                REQUIRE(is_in_bijection(optimal_partitions[i], assesser.optimal_partition(i + 1)));
            }
    }

    TEST_CASE("straightened altitudes BCE optimal cut", "[fragmentation_curve]") {

            tree t(array_1d<index_t>{ 8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14 });
            array_1d<char> ground_truth{ 0, 0, 1, 1, 1, 2, 2, 2 };

            assesser_fragmentation_optimal_cut assesser(t, ground_truth, optimal_cut_measure::BCE);

            auto altitudes = assesser.straightened_altitudes(false);
            array_1d<double> ref_scores{
                2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3,
                        2 + 4.0 / 3 + 4.0 / 3, 4, 3
            };

            std::vector<array_1d<index_t>> optimal_partitions
            {
                { 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 1, 1, 1, 1 },
                { 0, 0, 1, 1, 2, 2, 2, 2 },
                { 0, 0, 1, 1, 2, 2, 2, 3 },
                { 0, 0, 1, 1, 2, 2, 3, 4 },
                { 0, 0, 1, 1, 2, 2, 3, 4 },
                { 0, 0, 1, 2, 3, 4, 5, 6 },
                { 0, 1, 2, 3, 4, 5, 6, 7 }
            };

            auto sorted = hg_test::sort<double>(altitudes);

            for (index_t i = 0; i < (index_t) optimal_partitions.size(); i++) {
                auto tmp = labelisation_horizontal_cut_from_threshold(t,
                                                                      altitudes,
                                                                      sorted(root(t) - i));
                REQUIRE(is_in_bijection(optimal_partitions[i], tmp));
            }
    }

    TEST_CASE("fragmentation curve DHaming horizontal cut", "[fragmentation_curve]") {
         hg::tree tree{
                array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}
            };
            array_1d<int> altitudes{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3 };
            array_1d<int> ground_truth{ 3, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2 };


            auto res = assess_fragmentation_horizontal_cut(tree,
            altitudes,
            ground_truth,
            scorer_partition_DHamming());
            auto &res_scores = res.scores();
            auto &res_k = res.num_regions();

            array_1d<double> ref_scores{ 4.0, 8.0, 9.0, 10.0 };
            array_1d<double> ref_k{ 1, 3, 4, 9 };

            REQUIRE(res.num_regions_ground_truth() == 3);
            REQUIRE(hg::allclose(res_scores, hg_test::divided(ref_scores, num_leaves(tree))));
            REQUIRE(res_k == (ref_k));
    }

    TEST_CASE("fragmentation curve DHaming horizontal cut on rag", "[fragmentation_curve]") {
            hg::tree tree{
                array_1d<index_t>{9, 9, 9, 10, 10, 13, 12, 11, 11, 14, 13, 12, 15, 14, 15, 15}
            };
            array_1d<int> altitudes{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 1, 2, 3 };
            array_1d<int> ground_truth{ 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 };
            array_1d<index_t> vertex_map{ 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8 };

            auto res = assess_fragmentation_horizontal_cut(tree,
            altitudes,
            ground_truth,
            scorer_partition_DHamming(),
            vertex_map);
            auto &res_scores = res.scores();
            auto &res_k = res.num_regions();

            array_1d<double> ref_scores{ 4.0, 8.0, 9.0, 10.0 };
            array_1d<double> ref_k{ 1, 3, 4, 9 };

            REQUIRE(hg::allclose(res_scores, hg_test::divided(ref_scores, 11)));
            REQUIRE(res_k == ref_k);
    }

    TEST_CASE("fragmentation curve all measures horizontal cut", "[fragmentation_curve]") {
        // same values as the Python test test_assess_fragmentation_curve_all_measures_horizontal_cut
        hg::tree tree{array_1d<index_t>{11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18}};
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3};
        array_1d<int> ground_truth{0, 0, 0, 0, 3, 3, 3, 1, 1, 1, 1}; // label 2 unused
        array_1d<double> ref_k{1, 3, 4, 9};

        auto check = [&](const auto &scorer, const array_1d<double> &ref_scores) {
            auto res = assess_fragmentation_horizontal_cut(tree, altitudes, ground_truth, scorer);
            REQUIRE(res.num_regions_ground_truth() == 3);
            REQUIRE(res.num_regions() == ref_k);
            REQUIRE(hg::allclose(hg_test::multiplied(res.scores(), 11.0), ref_scores, 1e-14));
            auto normalized = res.num_regions_normalized();
            static_assert(std::is_same<decltype(normalized), array_1d<double>>::value, "");
            REQUIRE(normalized == array_1d<double>{1.0 / 3, 3.0 / 3, 4.0 / 3, 9.0 / 3});
            auto pos = std::max_element(ref_scores.begin(), ref_scores.end()) - ref_scores.begin();
            REQUIRE(res.optimal_score() == res.scores()(pos));
            REQUIRE(res.optimal_number_of_regions() == ref_k(pos));
        };
        check(scorer_partition_BCE(), {3.727272727272727, 5.666666666666666, 6.166666666666666, 3.5});
        check(scorer_partition_DHamming(), {4, 8, 9, 10});
        check(scorer_partition_DCovering(), {4, 6.199999999999999, 5.95, 3.366666666666667});
    }

    TEST_CASE("fragmentation curve all measures optimal cut", "[fragmentation_curve]") {
        // same values as the Python test test_assess_fragmentation_curve_all_measures_optimal_cut_non_contiguous_labels
        hg::tree t{array_1d<index_t>{8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14}};
        array_1d<int> ground_truth{0, 0, 3, 3, 3, 1, 1, 1}; // label 2 unused
        array_1d<double> ref_k{1, 2, 3, 4, 5, 6, 7, 8};

        auto check = [&](optimal_cut_measure measure, const array_1d<double> &ref_scores, size_t optimal_k) {
            assesser_fragmentation_optimal_cut assesser(t, ground_truth, measure);
            auto res = assesser.fragmentation_curve();
            REQUIRE(res.num_regions_ground_truth() == 3);
            REQUIRE(res.num_regions() == ref_k);
            REQUIRE(hg::allclose(hg_test::multiplied(res.scores(), 8.0), ref_scores, 1e-14));
            REQUIRE(hg::allclose(res.num_regions_normalized(), hg_test::divided(ref_k, 3.0), 1e-15));
            REQUIRE((size_t) assesser.optimal_number_of_regions() == optimal_k);
            REQUIRE(res.optimal_number_of_regions() == optimal_k);
            REQUIRE(assesser.optimal_score() == res.optimal_score());
        };
        check(optimal_cut_measure::BCE,
              {2.75, 4.5, 5.833333333333333, 5.333333333333333, 4.666666666666666, 4.666666666666666, 4, 3}, 3);
        check(optimal_cut_measure::DHamming, {3, 5, 7, 7, 7, 8, 8, 8}, 6);
        check(optimal_cut_measure::DCovering,
              {3, 5, 6.333333333333333, 5.666666666666666, 4.666666666666666, 4.666666666666666, 4, 3}, 3);
    }

    TEST_CASE("fragmentation curve optimal cut on rag, all measures", "[fragmentation_curve]") {
        // a tree on a rag whose vertex map is the identity gives the same curves as the tree on the pixels
        hg::tree t{array_1d<index_t>{8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14}};
        array_1d<int> ground_truth{0, 0, 3, 3, 3, 1, 1, 1};
        array_1d<index_t> vertex_map = hg::arange<index_t>(8);

        for (auto measure: {optimal_cut_measure::BCE, optimal_cut_measure::DHamming, optimal_cut_measure::DCovering}) {
            assesser_fragmentation_optimal_cut a_rag(t, ground_truth, measure, vertex_map);
            assesser_fragmentation_optimal_cut a_pixels(t, ground_truth, measure);
            auto c_rag = a_rag.fragmentation_curve();
            auto c_pixels = a_pixels.fragmentation_curve();
            REQUIRE(c_rag.num_regions() == c_pixels.num_regions());
            REQUIRE(c_rag.scores() == c_pixels.scores());
            REQUIRE(a_rag.optimal_number_of_regions() == a_pixels.optimal_number_of_regions());
            REQUIRE(a_rag.straightened_altitudes() == a_pixels.straightened_altitudes());
        }
    }

    TEMPLATE_TEST_CASE("fragmentation curve optimal score, empty curve and NaN", "[fragmentation_curve]",
                       float, double) {
        using curve_t = hg::fragmentation_curve<TestType>;
        // empty curve: the lowest value of the type
        curve_t empty_curve(hg::empty<TestType>({0}), hg::empty<TestType>({0}), 3);
        REQUIRE(empty_curve.optimal_score() == std::numeric_limits<TestType>::lowest());

        curve_t curve(array_1d<TestType>{1, 2, 3}, array_1d<TestType>{0.5, 0.75, 0.25}, 3);
        REQUIRE(curve.optimal_score() == TestType(0.75));

        // a NaN score propagates, wherever it is
        const TestType nan = std::numeric_limits<TestType>::quiet_NaN();
        for (const auto &scores: {array_1d<TestType>{0.5, nan, 0.25},
                                  array_1d<TestType>{nan, 0.5, 0.25},
                                  array_1d<TestType>{0.5, 0.25, nan}}) {
            curve_t nan_curve(array_1d<TestType>{1, 2, 3}, array_1d<TestType>(scores), 3);
            REQUIRE(std::isnan(nan_curve.optimal_score()));
        }
    }
}
