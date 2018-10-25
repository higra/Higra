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

#include "../graph.hpp"
#include "xtensor/xsort.hpp"

namespace hg {
    /**
    * Given two labelisations, a fine and a coarse one, of a same set of elements.
    * Find for each label (ie. region) of the fine labelisation, the label of the region in the
    * coarse labelisation that maximises the intersection with the "fine" region.
    *
    * Pre-condition:
    *  range(xlabelisation_fine) = [0..num_regions_fine[
    *  range(xlabelisation_coarse) = [0..num_regions_coarse[
    *
    * If num_regions_fine or num_regions_coarse are not provided, they will
    * be determined as max(xlabelisation_fine) + 1 and max(xlabelisation_coarse) + 1
    * @tparam T1
    * @tparam T2
    * @param xlabelisation_fine
    * @param num_regions_fine
    * @param xlabelisation_coarse
    * @param num_regions_coarse
    * @return a 1d array of size num_regions_fine
    */
    template<typename T1, typename T2>
    auto project_fine_to_coarse_labelisation
            (const xt::xexpression<T1> &xlabelisation_fine,
             const xt::xexpression<T2> &xlabelisation_coarse,
             size_t num_regions_fine = 0,
             size_t num_regions_coarse = 0) {

        auto &labelisation_fine = xlabelisation_fine.derived_cast();
        auto &labelisation_coarse = xlabelisation_coarse.derived_cast();

        hg_assert_integral_value_type(labelisation_fine);
        hg_assert_integral_value_type(labelisation_coarse);
        hg_assert_1d_array(labelisation_fine);
        hg_assert_1d_array(labelisation_coarse);
        hg_assert(labelisation_fine.size() == labelisation_coarse.size(),
                  "Labelisations must have the same size.");

        if(num_regions_fine == 0){
            num_regions_fine = xt::amax(labelisation_fine)(0) + 1;
        }

        if(num_regions_coarse == 0){
            num_regions_coarse = xt::amax(labelisation_coarse)(0) + 1;
        }

        array_2d <size_t> intersections = xt::zeros<size_t>({num_regions_fine, num_regions_coarse});

        for (index_t i = 0; i < labelisation_fine.size(); i++) {
            intersections(labelisation_fine(i), labelisation_coarse(i))++;
        }

        return xt::eval(xt::argmax(intersections, 1));
    }
}
