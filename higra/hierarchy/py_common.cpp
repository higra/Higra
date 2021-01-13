/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_common.hpp"

#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/graph.hpp"
#include "higra/hierarchy/common.hpp"
#include <string>

template<typename T>
using pyarray = xt::pyarray<T>;

using namespace hg;
namespace py = pybind11;

void py_init_common_hierarchy(pybind11::module &m) {
    xt::import_numpy();
}
