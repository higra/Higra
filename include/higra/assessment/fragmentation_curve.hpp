/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "../graph.hpp"
#include "../attribute/tree_attribute.hpp"
#include "../algo/tree.hpp"
#include <xtensor/xsort.hpp>

namespace hg {

    struct k_curve {
        array_1d<size_t> k;
        array_1d<double> scores;
    };

    enum class optimal_cut_measure {
        BCE,
        DHamming,
        Covering
    };

    namespace fragmentation_curve_internal {
        struct dynamic_node {
            size_t k; // number of regions
            double score;
            size_t back_track_k_left; // number of regions coming from left/first child
            size_t back_track_k_right; // number of regions coming from right/second  child
        };

    }

    /**
     * This class is used to assess the optimal cuts of a hierarchy of partitions with respect to
     * a given ground-truth labelisation of its base graph and the BCE measure.
     */
    class assesser_optimal_cut {
    public:

        /**
         * Create an assesser for hierarchy optimal cuts w.r.t. a given ground-truth partition of hierarchy
         * leaves and the BCE quality measure. The algorithms will explore optimal cuts containing at most
         * max_regions regions.
         *
         * The ground truth labelisation must be normalized (i.e. its labels must be positive integers
         * in the interval [0, num_regions[).
         *
         * @tparam tree_t tree type
         * @tparam T type of labels
         * @param tree input hierarchy
         * @param xground_truth ground truth labelisation of the tree leaves
         * @param max_regions maximum number of regions in the considered cuts.
         */
        template<typename tree_t, typename T>
        assesser_optimal_cut(
                const tree_t &tree,
                const xt::xexpression<T> &xground_truth,
                optimal_cut_measure measure,
                size_t max_regions = 200):
                m_tree(tree) {
            auto &ground_truth = xground_truth.derived_cast();
            hg_assert_leaf_weights(tree, ground_truth);
            hg_assert_1d_array(ground_truth);
            hg_assert_integral_value_type(ground_truth);

            max_regions = std::min(max_regions, num_leaves(tree));

            m_num_regions_ground_truth = xt::amax(ground_truth)() + 1;

            array_1d<index_t> region_gt_areas({m_num_regions_ground_truth}, 0);
            for (auto v:ground_truth) {
                region_gt_areas[v]++;
            }

            auto region_tree_area = attribute_area(tree);

            // for a tree node i, a gt region j: card_intersection(i, j) is the number of pixels in R_i cap R_j
            array_2d<index_t> card_intersection_leaves{{num_leaves(tree), region_gt_areas.size()}, 0};
            for (auto i: leaves_iterator(tree)) {
                card_intersection_leaves(i, ground_truth(i))++;
            }
            array_2d<double> card_intersection = accumulate_sequential(tree, card_intersection_leaves,
                                                                       accumulator_sum());

            array_1d<double> scores;

            switch (measure) {
                case optimal_cut_measure::BCE:
                    scores = xt::eval(xt::sum(
                            card_intersection *
                            xt::minimum(card_intersection / region_gt_areas,
                                        card_intersection / xt::view(region_tree_area, xt::all(), xt::newaxis())),
                            {1}));
                    break;
                case optimal_cut_measure::DHamming:
                    scores = xt::eval(xt::amax(card_intersection, {1}));
                    break;
                case optimal_cut_measure::Covering:
                    auto card_union = xt::eval(-card_intersection + region_gt_areas +
                                               xt::view(region_tree_area, xt::all(), xt::newaxis()));
                    scores = xt::eval(xt::amax(card_intersection / card_union, {1}) * region_tree_area);
                    break;
            }

            // initialize scoring for single region partitions (hte node itself)
            for (auto i: leaves_to_root_iterator(tree)) {
                backtracking.push_back({{1, scores(i), 0, 0}});
            }

            for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                hg_assert(num_children(i, tree) == 2, "Only binary trees are supported.");

                auto &backtrack_i = backtracking[i];

                auto c1 = child(0, i, tree);
                auto c2 = child(1, i, tree);
                auto &backtrack_c1 = backtracking[c1];
                auto &backtrack_c2 = backtracking[c2];

                auto max_regions_combination = backtrack_c1.size() + backtrack_c2.size(); // unlimited combinations
                max_regions_combination = std::min(max_regions, max_regions_combination);

                backtrack_i.resize(max_regions_combination);

                for (size_t k_c1 = 0; k_c1 < std::min(backtrack_c1.size(), max_regions); k_c1++) {
                    auto &match_k_c1 = backtrack_c1[k_c1];
                    for (size_t k_c2 = 0;
                         k_c2 < std::min(backtrack_c2.size(), max_regions_combination - k_c1 - 1); k_c2++) {
                        auto &match_k_c2 = backtrack_c2[k_c2];
                        size_t fusion_num_regions = k_c1 + k_c2 + 2; // +2 for indexing

                        auto fusion_score = match_k_c1.score + match_k_c2.score;
                        if (fusion_score > backtrack_i[fusion_num_regions - 1].score) {
                            backtrack_i[fusion_num_regions - 1] = {fusion_num_regions, fusion_score, k_c1 + 1,
                                                                   k_c2 + 1};
                        }
                    }
                }
            }
        }

        /**
         * Fragmentation curve, i.e. for each number of region k between 1 and max_regions,
         * the BCE score of the optimal cut with k regions.
         * @return a k_curve
         */
        auto fragmentation_curve() const {
            auto &backtrack_root = backtracking[root(m_tree)];
            array_1d<double> final_scores({backtrack_root.size()}, 0);
            for (index_t i = 0; i < backtrack_root.size(); i++) {
                final_scores(i) = backtrack_root[i].score;
            }
            return k_curve{xt::arange<size_t>(1, final_scores.size() + 1),
                           xt::eval(final_scores / (double) num_leaves(m_tree))};
        }

        /**
         * Number of regions in the ground truth labelisation of the hierarchy base graph
         * @return
         */
        auto number_of_region_ground_truth() const {
            return m_num_regions_ground_truth;
        }

        /**
         * Number of regions in the optimal cut
         * @return
         */
        auto optimal_number_of_regions() const {
            auto &backtrack_root = backtracking[root(m_tree)];
            return 1 + std::distance(
                    backtrack_root.begin(),
                    std::max_element(backtrack_root.begin(),
                                     backtrack_root.end(),
                                     [](const auto &a, const auto &b) { return a.score < b.score; }));
        }

        /**
         * Score of the optimal cut
         * @return
         */
        auto optimal_score() const {
            auto &backtrack_root = backtracking[root(m_tree)];
            return std::max_element(backtrack_root.begin(),
                                    backtrack_root.end(),
                                    [](const auto &a, const auto &b) { return a.score < b.score; }
            )->score / (double) num_leaves(m_tree);
        }

        /**
         * Labelisation of the base graph that corresponds to the optimal cut with
         * the given number of regions. If the number of regions is equal to 0 (default),
         * the global optimal cut it returned (it will contain get_optimal_number_of_regions regions).
         *
         * @param num_regions
         * @return
         */
        auto optimal_partition(size_t num_regions = 0) const {
            if (num_regions == 0) {
                num_regions = optimal_number_of_regions();
            }
            array_1d<bool> non_cut_nodes({num_vertices(m_tree)}, true);
            std::stack<std::pair<index_t, size_t>> s;
            s.push({root(m_tree), num_regions});
            while (!s.empty()) {
                index_t n;
                size_t k_n;
                std::tie(n, k_n) = s.top();
                s.pop();
                auto &node = backtracking[n][k_n - 1];
                non_cut_nodes[n] = false;

                if (node.back_track_k_left != 0) { // && node.back_track_k_right != 0
                    s.push({child(0, n, m_tree), node.back_track_k_left});
                    s.push({child(1, n, m_tree), node.back_track_k_right});
                }

            }
            return reconstruct_leaf_data(m_tree,
                                         xt::arange(num_vertices(m_tree)),
                                         non_cut_nodes);
        }

        /**
         * Compute tree node altitudes such that the horizontal cut of the resulting vertex valued hierarchy
         * corresponds to the optimal cut of the tree.
         *
         * @param gain_only If true, optimal cuts with a number of regions greater than optimal_number_of_regions() are ignored
         * @param normalize_result Ensure that altitudes[tree(root)] == optimal_score
         * @return
         */
        auto straightened_altitudes(bool gain_only = false, bool normalize_result = true) const {
            auto &backtrack_root = backtracking[root(m_tree)];
            auto score_max = backtrack_root[0].score;

            array_1d<double> altitudes({num_vertices(m_tree)}, 0);
            stackv<std::pair<index_t, size_t>> stack;

            auto back_track_cut = [&altitudes, &stack, this](size_t k, double score_gain) {
                stack.push({root(m_tree), k});
                while (!stack.empty()) {
                    index_t n;
                    size_t k_n;
                    std::tie(n, k_n) = stack.top();
                    stack.pop();
                    if (parent(n, m_tree) != n) {
                        altitudes(parent(n, m_tree)) += score_gain;
                    }
                    if (!is_leaf(n, m_tree)) {
                        auto &node = backtracking[n][k_n - 1];
                        if (node.back_track_k_left != 0) { // && node.back_track_k_right != 0
                            stack.push({child(0, n, m_tree), node.back_track_k_left});
                            stack.push({child(1, n, m_tree), node.back_track_k_right});
                        }
                    }
                }
            };

            for (index_t i = 1; i < backtrack_root.size(); i++) {
                double score = backtrack_root[i].score;
                double previous_score = backtrack_root[i - 1].score;
                double score_gain = score - previous_score;
                if (!gain_only || score_gain > 0) {
                    score_max = std::max(score_max, score);
                    back_track_cut(i + 1, std::abs(score_gain) * 0.5); // *0.5 summation from two children
                }
            }
            if (normalize_result) {
                back_track_cut(backtrack_root.size(),
                               (score_max - altitudes[root(m_tree)]) * 0.5); // *0.5 summation from two children
            }

            return altitudes;
        }


    private:
        std::vector<std::vector<fragmentation_curve_internal::dynamic_node>> backtracking;
        const hg::tree m_tree;
        size_t m_num_regions_ground_truth;
    };

}
