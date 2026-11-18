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

#include "array/all.hpp"
#include "point.hpp"

namespace hg {

    /*
     * Names kept for continuity: the rank is not part of the type, every alias is hg::array<value_t>.
     * Code that needs a given rank checks it at run time (hg_assert_1d_array, dimension()).
     */

    template<typename value_t>
    using array_1d = array<value_t>;

    template<typename value_t>
    using array_2d = array<value_t>;

    template<typename value_t>
    using array_3d = array<value_t>;

    template<typename value_t>
    using array_4d = array<value_t>;

    template<typename value_t>
    using array_nd = array<value_t>;
}
