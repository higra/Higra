/***************************************************************************
* Copyright ESIEE Paris (2021)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

#include <pybind11/pybind11.h>

// TODO: remove with future xtensor_python release
#pragma once


namespace pybind11 {
    namespace detail {
        // Type caster for casting xt::xstrided_view to ndarray
        template<class CT, class S, xt::layout_type L, class FST>
        struct type_caster<xt::xstrided_view<CT, S, L, FST>>
                : xtensor_type_caster_base<xt::xstrided_view<CT, S, L, FST>> {
        };

        // Type caster for casting xt::xarray_adaptor to ndarray
        template<class EC, xt::layout_type L, class SC, class Tag>
        struct type_caster<xt::xarray_adaptor<EC, L, SC, Tag>>
                : xtensor_type_caster_base<xt::xarray_adaptor<EC, L, SC, Tag>> {
        };

        // Type caster for casting xt::xtensor_adaptor to ndarray
        template<class EC, std::size_t N, xt::layout_type L, class Tag>
        struct type_caster<xt::xtensor_adaptor<EC, N, L, Tag>>
                : xtensor_type_caster_base<xt::xtensor_adaptor<EC, N, L, Tag>> {
        };
    }
}