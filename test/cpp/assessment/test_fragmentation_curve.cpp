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

            REQUIRE(res_scores == (ref_scores / num_leaves(t)));
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

            REQUIRE(xt::allclose(res_scores, ref_scores / num_leaves(t)));
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

            REQUIRE(res_scores == (ref_scores / num_leaves(t)));
            REQUIRE(xt::allclose(res_k, ref_k / 3));
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

            REQUIRE(xt::allclose(res_scores, (ref_scores / num_leaves(t))));
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

            auto sorted = xt::sort(altitudes);

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
            array_1d<int> ground_truth{ 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 };


            auto res = assess_fragmentation_horizontal_cut(tree,
            altitudes,
            ground_truth,
            scorer_partition_DHamming());
            auto &res_scores = res.scores();
            auto &res_k = res.num_regions();

            array_1d<double> ref_scores{ 4.0, 8.0, 9.0, 10.0 };
            array_1d<double> ref_k{ 1, 3, 4, 9 };

            REQUIRE(xt::allclose(res_scores, ref_scores / num_leaves(tree)));
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

            REQUIRE(xt::allclose(res_scores, ref_scores / 11));
            REQUIRE(res_k == ref_k);
    }
}