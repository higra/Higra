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


    template<typename T1, typename T2>
    auto card_intersections(const xt::xexpression<T1> &xcandidate,
                            const xt::xexpression<T2> &xground_truths) {
        auto &candidate = xcandidate.derived_cast();
        auto &ground_truths = xground_truths.derived_cast();

        hg_assert_integral_value_type(candidate);
        hg_assert_integral_value_type(ground_truths);

        std::vector<array_2d<index_t>> result;
        auto num_regions_candidate = xt::amax(candidate)() + 1;

        auto compute = [&candidate, &result, num_regions_candidate](const auto &ground_truth) {
            hg_assert_same_shape(candidate, ground_truth);
            auto num_regions_ground_truth = xt::amax(ground_truth)() + 1;
            array_2d<index_t>::shape_type shape_result{(size_t) num_regions_candidate,
                                                       (size_t) num_regions_ground_truth};
            result.emplace_back(shape_result, 0);
            auto & r = result.back();

            const auto &cf = xt::flatten(candidate);
            const auto &gtf = xt::flatten(ground_truth);
            for (index_t i = 0; i < cf.size(); i++) {
                r(cf(i), gtf(i))++;
            }
        };

        if (xt::same_shape(candidate, ground_truths)) {
            compute(ground_truths);
        } else {
            for (index_t i = 0; i < ground_truths.shape()[0]; i++) {
                compute(xt::view(ground_truths, i));
            }
        }

        return result;
    }

}
