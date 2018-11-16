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
            for (index_t i = 0; i < cf.size(); i++) {
                r(cf(i), gtf(i))++;
            }
        };

        if (xt::same_shape(candidate.shape(), ground_truths.shape())) {
            compute(ground_truths);
        } else {
            for (index_t i = 0; i < ground_truths.shape()[0]; i++) {
                compute(xt::view(ground_truths, i));
            }
        }

        return result;
    }

    template<typename T1, typename T2>
    auto assess_partition_BCE(const xt::xexpression<T1> &xcandidate,
                              const xt::xexpression<T2> &xground_truths) {
        auto &candidate = xcandidate.derived_cast();
        auto card_intersections = hg::card_intersections<double>(xcandidate, xground_truths);
        double score = 0;
        auto candidate_regions_area = xt::sum(card_intersections[0], {1});
        for (const auto &card_intersection: card_intersections) {
            score += xt::sum(
                    card_intersection *
                    xt::minimum(card_intersection / xt::sum(card_intersection, {0}),
                                card_intersection / xt::view(candidate_regions_area, xt::all(), xt::newaxis())))();
        }
        return (score / candidate.size()) / card_intersections.size();
    }

    template<typename T1, typename T2>
    auto assess_partition_DHamming(const xt::xexpression<T1> &xcandidate,
                                   const xt::xexpression<T2> &xground_truths) {
        auto &candidate = xcandidate.derived_cast();
        auto card_intersections = hg::card_intersections<double>(xcandidate, xground_truths);
        double score = 0;
        for (const auto &card_intersection: card_intersections) {
            score += xt::sum(xt::amax(card_intersection, {1}))();
        }
        return (score / candidate.size()) / card_intersections.size();
    }

    template<typename T1, typename T2>
    auto assess_partition_DCovering(const xt::xexpression<T1> &xcandidate,
                                   const xt::xexpression<T2> &xground_truths) {
        auto &candidate = xcandidate.derived_cast();
        auto card_intersections = hg::card_intersections<double>(xcandidate, xground_truths);
        double score = 0;
        auto candidate_regions_area = xt::sum(card_intersections[0], {1});
        for (const auto &card_intersection: card_intersections) {
            auto card_union = -card_intersection + xt::sum(card_intersection, {0}) +
                              xt::view(candidate_regions_area, xt::all(), xt::newaxis());
            score += xt::eval(xt::sum(xt::amax(card_intersection / card_union, {1}) * candidate_regions_area))();

        }
        return (score / candidate.size()) / card_intersections.size();
    }

}
