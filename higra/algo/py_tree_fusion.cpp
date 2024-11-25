/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_fusion.hpp"
#include "higra/algo/tree_fusion.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py_tree_fusion {

    template<typename T>
    using pyarray = xt::pyarray<T>;

    namespace py = pybind11;
    using namespace hg;

    void py_init_tree_fusion(pybind11::module &m) {
        //xt::import_numpy();

        m.def("_tree_fusion_depth_map", [](const std::vector<tree *> &trees) {
            return tree_fusion_depth_map(trees);
        });

    }

}



