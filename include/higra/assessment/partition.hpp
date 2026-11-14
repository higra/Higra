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

#include "../structure/array.hpp"
#include <limits>
#include <vector>

namespace hg {

    enum class partition_measure {
        BCE,
        DHamming,
        DCovering
    };

    namespace partition_internal {

        /**
         * Largest value of a view, or the lowest value of the type if the view is empty.
         */
        template<typename V>
        V max_value(const array_view<const V> &values) {
            V res = std::numeric_limits<V>::lowest();
            const V *d = values.data();
            for (index_t i = 0; i < values.size(); i++) {
                if (!(res > d[i])) {
                    res = d[i];
                }
            }
            return res;
        }

        /**
         * Smaller of two values: b if a < b is false (NaN in b propagates).
         */
        template<typename V>
        inline V select_min(const V &a, const V &b) {
            return (a < b) ? a : b;
        }

        /**
         * Larger of two values: b if a > b is false (NaN in b propagates).
         */
        template<typename V>
        inline V select_max(const V &a, const V &b) {
            return (a > b) ? a : b;
        }

        /**
         * Contiguous view of a 2d matrix, its number of rows and
         * columns, the sum of each row (candidate region areas) and of each column (ground-truth
         * region areas). Sums are computed sequentially by increasing indices.
         */
        template<typename T>
        struct card_intersection_matrix {
            using value_type = T;

            /**
             * The matrix refers to the elements of card_intersection, which must outlive it.
             */
            explicit card_intersection_matrix(array_view<const T> card_intersection) {
                const auto &v = card_intersection;
                hg_assert(v.dimension() == 2, "The card intersection matrix must be 2d.");
                num_rows = v.shape(0);
                num_cols = v.shape(1);
                data = v.data();
                row_sums.assign(num_rows, value_type(0));
                col_sums.assign(num_cols, value_type(0));
                for (index_t j = 0; j < num_cols; j++) {
                    for (index_t i = 0; i < num_rows; i++) {
                        col_sums[j] += data[i * num_cols + j];
                    }
                }
                for (index_t i = 0; i < num_rows; i++) {
                    for (index_t j = 0; j < num_cols; j++) {
                        row_sums[i] += data[i * num_cols + j];
                    }
                }
            }

            value_type operator()(index_t i, index_t j) const {
                return data[i * num_cols + j];
            }

            value_type total_row_sums() const {
                value_type res = 0;
                for (auto v: row_sums) {
                    res += v;
                }
                return res;
            }

            const value_type *data;
            index_t num_rows;
            index_t num_cols;
            std::vector<value_type> row_sums;
            std::vector<value_type> col_sums;
        };
    }

    template<typename value_type=index_t, typename T1, typename T2>
    auto card_intersections(array_view<const T1> candidate,
                            array_view<const T2> ground_truths) {

        hg_assert_integral_value_type(candidate);
        hg_assert_integral_value_type(ground_truths);

        std::vector<array_2d<value_type>> result;
        auto num_regions_candidate = partition_internal::max_value(candidate) + 1;

        auto compute = [&candidate, &result, num_regions_candidate](const auto &ground_truth) {
            hg_assert_same_shape(candidate, ground_truth);
            auto num_regions_ground_truth = partition_internal::max_value(ground_truth) + 1;
            array_shape shape_result{(index_t) num_regions_candidate, (index_t) num_regions_ground_truth};
            result.emplace_back(shape_result, 0);
            auto &r = result.back();

            const auto *cf = candidate.data();
            const auto *gtf = ground_truth.data();
            for (index_t i = 0; i < candidate.size(); i++) {
                r(cf[i], gtf[i])++;
            }
        };

        if (hg::same_shape(candidate.shape(), ground_truths.shape())) {
            compute(ground_truths);
        } else {
            for (index_t i = 0; i < ground_truths.shape(0); i++) {
                compute(ground_truths.row(i));
            }
        }

        return result;
    }

    struct scorer_partition_BCE {
        template<typename T>
        static
        auto score(array_view<const T> xcard_intersection) {
            const partition_internal::card_intersection_matrix<T> ci(xcard_intersection);
            using value_type = typename partition_internal::card_intersection_matrix<T>::value_type;

            // sum of the row sums
            value_type sum = 0;
            for (index_t i = 0; i < ci.num_rows; i++) {
                value_type row_sum = 0;
                for (index_t j = 0; j < ci.num_cols; j++) {
                    const value_type c = ci(i, j);
                    row_sum += c * partition_internal::select_min<value_type>(c / ci.col_sums[j], c / ci.row_sums[i]);
                }
                sum += row_sum;
            }
            double score = sum;

            return score / ci.total_row_sums();
        }
    };

    struct scorer_partition_DHamming {
        template<typename T>
        static
        auto score(array_view<const T> xcard_intersection) {
            const partition_internal::card_intersection_matrix<T> ci(xcard_intersection);
            using value_type = typename partition_internal::card_intersection_matrix<T>::value_type;

            value_type sum_max = 0;
            for (index_t i = 0; i < ci.num_rows; i++) {
                value_type row_max = std::numeric_limits<value_type>::lowest();
                for (index_t j = 0; j < ci.num_cols; j++) {
                    row_max = partition_internal::select_max(row_max, ci(i, j));
                }
                sum_max += row_max;
            }

            return sum_max / ci.total_row_sums();
        }
    };

    struct scorer_partition_DCovering {
        template<typename T>
        static
        auto score(array_view<const T> xcard_intersection) {
            const partition_internal::card_intersection_matrix<T> ci(xcard_intersection);
            using value_type = typename partition_internal::card_intersection_matrix<T>::value_type;

            value_type sum = 0;
            for (index_t i = 0; i < ci.num_rows; i++) {
                value_type row_max = std::numeric_limits<value_type>::lowest();
                for (index_t j = 0; j < ci.num_cols; j++) {
                    const value_type c = ci(i, j);
                    const value_type card_union = -c + ci.col_sums[j] + ci.row_sums[i];
                    row_max = partition_internal::select_max<value_type>(row_max, c / card_union);
                }
                sum += row_max * ci.row_sums[i];
            }
            double score = sum;

            return score / ci.total_row_sums();
        }
    };

    template<typename T, typename scorer_t>
    auto assess_partition(const std::vector<T> &card_intersections, const scorer_t &scorer) {
        double score = 0;
        for (const auto &card_intersection: card_intersections) {
            score += scorer.score(card_intersection);
        }
        return score / card_intersections.size();
    }

    template<typename T1, typename T2, typename scorer_t>
    auto assess_partition(array_view<const T1> xcandidate,
                          array_view<const T2> xground_truths,
                          const scorer_t &scorer) {
        auto card_intersections = hg::card_intersections<double>(xcandidate, xground_truths);
        return assess_partition(card_intersections, scorer);
    }

}
