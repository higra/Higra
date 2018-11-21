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
#include <vector>
#include <xtensor/xview.hpp>

namespace hg {

    enum class partition_measure {
        BCE,
        DHamming,
        DCovering
    };

    template<typename value_type=index_t, typename T1, typename T2>
    auto card_intersections(const xt::xexpression<T1> &xcandidate,
                            const xt::xexpression<T2> &xground_truths) {
        auto &candidate = xcandidate.derived_cast();
        auto &ground_truths = xground_truths.derived_cast();

        hg_assert_integral_value_type(candidate);
        hg_assert_integral_value_type(ground_truths);

        std::vector<array_2d<value_type>> result;
        auto num_regions_candidate = xt::amax(candidate)() + 1;

        auto compute = [&candidate, &result, num_regions_candidate](const auto &ground_truth) {
            hg_assert_same_shape(candidate, ground_truth);
            auto num_regions_ground_truth = xt::amax(ground_truth)() + 1;
            typename array_2d<value_type>::shape_type shape_result{(size_t) num_regions_candidate,
                                                                   (size_t) num_regions_ground_truth};
            result.emplace_back(shape_result, 0);
            auto &r = result.back();

            const auto &cf = xt::flatten(candidate);
            const auto &gtf = xt::flatten(ground_truth);
            for (index_t i = 0; i < (index_t)cf.size(); i++) {
                r(cf(i), gtf(i))++;
            }
        };

        if (xt::same_shape(candidate.shape(), ground_truths.shape())) {
            compute(ground_truths);
        } else {
            for (index_t i = 0; i < (index_t)ground_truths.shape()[0]; i++) {
                compute(xt::view(ground_truths, i));
            }
        }

        return result;
    }

    struct scorer_partition_BCE {
        template<typename T>
        static
        auto score(const xt::xexpression<T> &xcard_intersection) {
            auto card_intersection = xcard_intersection.derived_cast();
            auto candidate_regions_area = xt::sum(card_intersection, {1});

            double score = xt::sum(
                    card_intersection *
                    xt::minimum(card_intersection / xt::sum(card_intersection, {0}),
                                card_intersection / xt::view(candidate_regions_area, xt::all(), xt::newaxis())))();

            return score / xt::sum(candidate_regions_area)();
        }
    };

    struct scorer_partition_DHamming {
        template<typename T>
        static
        auto score(const xt::xexpression<T> &xcard_intersection) {
            auto card_intersection = xcard_intersection.derived_cast();

            return (xt::sum(xt::amax(card_intersection, {1}))() / xt::sum(card_intersection)());
        }
    };

    struct scorer_partition_DCovering {
        template<typename T>
        static
        auto score(const xt::xexpression<T> &xcard_intersection) {
            auto card_intersection = xcard_intersection.derived_cast();
            auto candidate_regions_area = xt::sum(card_intersection, {1});

            auto card_union = -card_intersection + xt::sum(card_intersection, {0}) +
                              xt::view(candidate_regions_area, xt::all(), xt::newaxis());

            double score = xt::eval(xt::sum(xt::amax(card_intersection / card_union, {1}) * candidate_regions_area))();

            return score / xt::sum(candidate_regions_area)();
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
    auto assess_partition(const xt::xexpression<T1> &xcandidate,
                          const xt::xexpression<T2> &xground_truths,
                          const scorer_t &scorer) {
        auto card_intersections = hg::card_intersections<double>(xcandidate, xground_truths);
        return assess_partition(card_intersections, scorer);
    }

}
