/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Wonder Alexandre Luz Alves                              *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "pybind11/pybind11.h"

namespace py_component_tree_dual_filter {
    void py_init_component_tree_dual_filter(pybind11::module &m);
}
