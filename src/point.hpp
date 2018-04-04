//
// Created by user on 4/4/18.
//

#pragma once

#include "xtensor/xfixed.hpp"

namespace hg {
    template<typename value_t, uint dim>
    using point = xt::xtensorf <value_t, xt::xshape<dim>>;

    using point2d = point<double, 2>;
    using point2i = point<long, 2>;

    using point3d = point<double, 3>;
    using point3i = point<long, 3>;

    using point4d = point<double, 4>;
    using point4i = point<long, 4>;
}