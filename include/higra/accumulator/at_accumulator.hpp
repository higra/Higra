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
                typename T,
                typename accumulator_t,
                typename output_t = typename T::value_type>
        auto
        at_accumulate(const array_1d<index_t> &indices,
                      const xt::xexpression<T> &xweights,
                      const accumulator_t &accumulator) {
            HG_TRACE();
            auto &weights = xweights.derived_cast();
            hg_assert(weights.shape()[0] == indices.size(), "Weights dimension does not match rag map dimension.");

            index_t size = xt::amax(indices)() + 1;
            auto data_shape = std::vector<size_t>(weights.shape().begin() + 1, weights.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), size);
            array_nd<typename T::value_type> res = array_nd<typename T::value_type>::from_shape(output_shape);

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
                if (indices.data()[i] != invalid_index) {
                    input_view.set_position(i);
                    accs[indices.data()[i]].accumulate(input_view.begin());
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
     * @tparam T
     * @tparam accumulator_t
     * @tparam output_t
     * @param indices a 1d array of indices (entry equals to :math:`-1` are ignored)
     * @param xweights a nd-array of shape :math:`(s_1, \ldots, s_n)` such that :math:`s_1=indices.size()`
     * @param accumulator
     * @return a nd-array of size :math:`(M, s_2, \ldots, s_n)`
     */
    template<typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_at(const array_1d<index_t> &indices,
                       const xt::xexpression<T> &xweights,
                       const accumulator_t &accumulator) {
        if (xweights.derived_cast().dimension() == 1) {
            return at_accumulator_internal::at_accumulate<false, T, accumulator_t, output_t>(indices,
                                                                                             xweights,
                                                                                             accumulator);
        } else {
            return at_accumulator_internal::at_accumulate<true, T, accumulator_t, output_t>(indices,
                                                                                            xweights,
                                                                                            accumulator);
        }
    };

}
