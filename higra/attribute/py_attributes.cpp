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

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

struct def_perimeter_length_component_tree {
    template<typename T>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_attribute_perimeter_length_component_tree",
              [](const hg::tree &tree,
                 const hg::ugraph &graph,
                 const pyarray<T> &vertex_perimeter,
                 const pyarray<T> &edge_length) {
                  return hg::attribute_perimeter_length_component_tree(
                          tree,
                          graph,
                          vertex_perimeter,
                          edge_length
                  );
              },
              doc,
              py::arg("tree"),
              py::arg("graph"),
              py::arg("vertex_perimeter"),
              py::arg("edge_length"));
    }
};

struct def_attribute_extrema {
    template<typename T>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_attribute_extrema",
              [](const hg::tree &tree,
                 const pyarray<T> &altitudes) {
                  return hg::attribute_extrema(
                          tree,
                          altitudes
                  );
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"));
    }
};

struct def_attribute_extinction_value {
    template<typename T>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_attribute_extinction_value",
              [](const hg::tree &tree,
                 const pyarray<T> &altitudes,
                 const pyarray<T> &attribute,
                 bool increasing_altitudes) {
                  return hg::attribute_extinction_value(
                          tree,
                          altitudes,
                          attribute,
                          increasing_altitudes
                  );
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"),
              py::arg("attribute"),
              py::arg("increasing_altitudes"));
    }
};

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

    add_type_overloads<def_perimeter_length_component_tree,
            HG_TEMPLATE_FLOAT_TYPES>(m, "");

    add_type_overloads<def_attribute_extrema,
            HG_TEMPLATE_NUMERIC_TYPES>(m, "");

    add_type_overloads<def_attribute_extinction_value,
            HG_TEMPLATE_FLOAT_TYPES>(m, "");
}
