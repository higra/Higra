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
#include "accumulator.hpp"
#include "../structure/details/light_axis_view.hpp"

namespace hg {

    namespace at_accumulator_internal {
        template<bool vectorial,
                typename V,
                typename accumulator_t,
                typename output_t = V>
        auto
        at_accumulate(array_view<const index_t> indices,
                      array_view<const V> weights,
                      const accumulator_t &accumulator) {
            HG_TRACE();
            hg_assert(weights.dimension() > 0 && weights.shape(0) == indices.size(),
                      "Weights dimension does not match rag map dimension.");

            // M = max(indices); indices equal to -1 are ignored, so M + 1 = 0 if every index is ignored
            index_t size = 0;
            for (index_t i = 0; i < indices.size(); ++i) {
                hg_assert(indices[i] >= invalid_index,
                          "Indices must be greater than or equal to -1, found " + std::to_string(indices[i]) +
                          " at position " + std::to_string(i) + ".");
                size = (std::max)(size, indices[i] + 1);
            }
            auto data_shape = std::vector<size_t>(weights.shape().begin() + 1, weights.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), size);
            array_nd<V> res = array_nd<V>::from_shape(array_shape::from(output_shape));

            auto input_view = make_light_axis_view<vectorial>(weights);
            auto output_view = make_light_axis_view<vectorial>(res);

            std::vector<decltype(accumulator.template make_accumulator<vectorial>(output_view))> accs;
            accs.reserve(size);


            for (index_t i = 0; i < size; ++i) {
                output_view.set_position(i);
                accs.push_back(accumulator.template make_accumulator<vectorial>(output_view));
                accs[i].initialize();
            }

            index_t map_size = indices.size();
            for (index_t i = 0; i < map_size; ++i) {
                if (indices[i] != invalid_index) {
                    input_view.set_position(i);
                    accs[indices[i]].accumulate(input_view.begin());
                }
            }

            for (auto &acc: accs) {
                acc.finalize();
            }

            return res;
        }
    }

    /**
     * Accumulate the given weights located at given indices.
     *
     * Let :math:`M = max(indices)`. For all :math:`i \in \{0, \ldots, M\}`
     *
     * .. math::
     *
     *      result[i] = accumulator(\{weights[j, :] \mid indices[j] = i  \})
     *
     * @tparam TI
     * @tparam T
     * @tparam accumulator_t
     * @tparam output_t
     * @param indices a 1d array of indices (entries equal to :math:`-1` are ignored; any other negative entry
     *     throws an exception)
     * @param weights a nd-array of shape :math:`(s_1, \ldots, s_n)` such that :math:`s_1=indices.size()`
     * @param accumulator
     * @return a nd-array of size :math:`(M + 1, s_2, \ldots, s_n)`
     */
    template<typename TI, typename T, typename accumulator_t, typename output_t = T>
    auto accumulate_at(array_view<const TI> indices,
                       array_view<const T> weights,
                       const accumulator_t &accumulator) {
        static_assert(std::is_integral<TI>::value,
                      "Indices must have integral value type.");
        if constexpr (!std::is_same<TI, index_t>::value) {
            return accumulate_at(array_view<const index_t>(hg::cast<index_t>(indices)), weights, accumulator);
        } else if (weights.dimension() == 1) {
            return at_accumulator_internal::at_accumulate<false>(indices,
                                                                 weights,
                                                                 accumulator);
        } else {
            return at_accumulator_internal::at_accumulate<true>(indices,
                                                                weights,
                                                                accumulator);
        }
    };

}
