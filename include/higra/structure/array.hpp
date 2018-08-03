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

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xeval.hpp"

#include "point.hpp"

namespace hg {

    template<typename value_t>
    using array_1d = xt::xtensor<value_t, 1>;

    template<typename value_t>
    using array_2d = xt::xtensor<value_t, 2>;

    template<typename value_t>
    using array_3d = xt::xtensor<value_t, 3>;

    template<typename value_t>
    using array_3d = xt::xtensor<value_t, 3>;

    template<typename value_t>
    using array_4d = xt::xtensor<value_t, 4>;

    template<typename value_t>
    using array_nd = xt::xarray<value_t>;
}
