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
#include "../structure/array/xtensor_bridge.hpp"

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
                size = (std::max)(size, indices[i] + 1);
            }
            auto data_shape = std::vector<size_t>(weights.shape().begin() + 1, weights.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), size);
            array_nd<V> res = array_nd<V>::from_shape(output_shape);

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
     * @param xindices a 1d array of indices (entry equals to :math:`-1` are ignored)
     * @param xweights a nd-array of shape :math:`(s_1, \ldots, s_n)` such that :math:`s_1=indices.size()`
     * @param accumulator
     * @return a nd-array of size :math:`(M + 1, s_2, \ldots, s_n)`
     */
    template<typename TI, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_at(const xt::xexpression<TI> &xindices,
                       const xt::xexpression<T> &xweights,
                       const accumulator_t &accumulator) {
        static_assert(std::is_integral<std::decay_t<typename TI::value_type>>::value,
                      "Indices must have integral value type.");
        const auto indices = [&xindices]() {
            if constexpr (std::is_same<std::decay_t<typename TI::value_type>, index_t>::value) {
                return hg::bridge::as_view(xindices);
            } else {
                return bridge::view_holder<index_t>(hg::bridge::to_array(xt::cast<index_t>(xindices.derived_cast())));
            }
        }();
        const auto weights = hg::bridge::as_view(xweights);
        if (weights.view.dimension() == 1) {
            return at_accumulator_internal::at_accumulate<false>(indices.view,
                                                                 weights.view,
                                                                 accumulator);
        } else {
            return at_accumulator_internal::at_accumulate<true>(indices.view,
                                                                weights.view,
                                                                accumulator);
        }
    };

}
