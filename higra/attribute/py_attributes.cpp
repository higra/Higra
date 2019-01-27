/***************************************************************************
* Copyright ESIEE Paris (2019)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_attributes.hpp"
#include "../py_common.hpp"
#include "higra/attribute/tree_attribute.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

void py_init_attributes(pybind11::module &m) {
    xt::import_numpy();
    m.def("_attribute_sibling",
          [](const hg::tree &tree, hg::index_t skip) {
              return hg::attribute_sibling(tree, skip);
          },
          "Attribute sibling.",
          pybind11::arg("tree"),
          pybind11::arg("skip") = 1);

    m.def("_attribute_depth",
          [](const hg::tree &tree) {
              return hg::attribute_depth(tree);
          },
          "Attribute depth.",
          pybind11::arg("tree"));
}
