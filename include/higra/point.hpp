//
// Created by user on 4/4/18.
//

#pragma once

#include "xtensor/xfixed.hpp"

namespace hg {
    template<typename value_t, uint dim>
    using point = xt::xtensorf <value_t, xt::xshape<dim>>;

    using point_1d_f = point<double, 1>;
    using point_1d_i = point<long, 1>;

    using point_2d_f = point<double, 2>;
    using point_2d_i = point<long, 2>;

    using point_3d_f = point<double, 3>;
    using point_3d_i = point<long, 3>;

    using point_4d_f = point<double, 4>;
    using point_4d_i = point<long, 4>;
}