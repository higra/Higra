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

#include "../utils.hpp"
#include "xtensor/xfixed.hpp"

namespace hg {
    template<typename value_t, unsigned int dim>
    using point = xt::xtensor_fixed <value_t, xt::xshape<dim>>;

    using point_1d_f = point<double, 1>;
    using point_1d_i = point<index_t, 1>;

    using point_2d_f = point<double, 2>;
    using point_2d_i = point<index_t, 2>;

    using point_3d_f = point<double, 3>;
    using point_3d_i = point<index_t, 3>;

    using point_4d_f = point<double, 4>;
    using point_4d_i = point<index_t, 4>;
}