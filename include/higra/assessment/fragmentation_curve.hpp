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

    namespace fragmentation_curve_internal {
        struct dynamic_node {
            size_t k; // number of regions
            double score;
            size_t back_track_k_left; // number of regions coming from left/first child
            size_t back_track_k_right; // number of regions coming from right/second  child
        };

    }

    class assesser_optimal_cut_BCE {
    public:

        template<typename tree_t, typename T>
        assesser_optimal_cut_BCE(
                const tree_t &tree,
                const xt::xexpression<T> &xground_truth,
                size_t max_regions = 200):
                m_tree(tree) {
            auto &ground_truth = xground_truth.derived_cast();
            hg_assert_leaf_weights(tree, ground_truth);
            hg_assert_1d_array(ground_truth);
            hg_assert_integral_value_type(ground_truth);

            max_regions = std::min(max_regions, num_leaves(tree));

            size_t nb_regions_gt = xt::amax(ground_truth)() + 1;
            array_1d<index_t> region_gt_areas({nb_regions_gt}, 0);
            for (auto v:ground_truth) {
                region_gt_areas[v]++;
            }

            auto region_tree_area = attribute_area(tree);

            // for a tree node i, a gt region j: card_intersection(i, j) is the number of pixels in R_i cap R_j
            array_2d<index_t> card_intersection_leaves{{num_leaves(tree), nb_regions_gt}, 0};
            for (auto i: leaves_iterator(tree)) {
                card_intersection_leaves(i, ground_truth(i))++;
            }
            array_2d<double> card_intersection = accumulate_sequential(tree, card_intersection_leaves,
                                                                       accumulator_sum());
            //auto card_union = xt::eval(-card_intersection + region_gt_areas + xt::view(region_tree_area, xt::all(), xt::newaxis()));

            auto scores = xt::eval(xt::sum(
                    card_intersection *
                    xt::minimum(card_intersection / region_gt_areas,
                                card_intersection / xt::view(region_tree_area, xt::all(), xt::newaxis())),
                    {1}));

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

        auto get_fragmentation_curve() const{
            auto &backtrack_root = backtracking[root(m_tree)];
            array_1d<double> final_scores({backtrack_root.size()}, 0);
            for (index_t i = 0; i < backtrack_root.size(); i++) {
                final_scores(i) = backtrack_root[i].score;
            }
            return k_curve{xt::arange<size_t>(1, final_scores.size() + 1),
                           xt::eval(final_scores / (double) num_leaves(m_tree))};
        }

        auto get_optimal_number_of_regions() const {
            auto &backtrack_root = backtracking[root(m_tree)];
            return 1 + std::distance(
                    backtrack_root.begin(),
                    std::max_element(backtrack_root.begin(),
                                     backtrack_root.end(),
                                     [](const auto &a, const auto &b) { return a.score < b.score; }));
        }

        auto get_optimal_partition(size_t num_regions = 0) const{
            if (num_regions == 0) {
                num_regions = get_optimal_number_of_regions();
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

    private:
        std::vector<std::vector<fragmentation_curve_internal::dynamic_node>> backtracking;
        const hg::tree &m_tree;
    };

    template<typename tree_t, typename T>
    auto assess_fragmentation_curve_BCE_optimal_cut(
            const tree_t &tree,
            const xt::xexpression<T> &xground_truth,
            size_t max_regions = 200) {
        assesser_optimal_cut_BCE assesser(tree, xground_truth, max_regions);
        return assesser.get_fragmentation_curve();
    }
}
