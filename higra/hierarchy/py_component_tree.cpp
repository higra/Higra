/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_component_tree.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

template<typename graph_t>
struct def_min_tree {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_component_tree_min_tree",
              [](const graph_t &graph,
                 const pyarray<value_t> &vertex_weights) {
                  return hg::component_tree_min_tree(graph, vertex_weights);
              },
              doc,
              py::arg("graph"),
              py::arg("vertex_weights"));
    }
};

template<typename graph_t>
struct def_max_tree {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_component_tree_max_tree",
              [](const graph_t &graph,
                 const pyarray<value_t> &vertex_weights) {
                  return hg::component_tree_max_tree(graph, vertex_weights);
              },
              doc,
              py::arg("graph"),
              py::arg("vertex_weights"));
    }
};


void py_init_component_tree(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_min_tree<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_min_tree<hg::regular_grid_graph_1d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_min_tree<hg::regular_grid_graph_2d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_min_tree<hg::regular_grid_graph_3d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_min_tree<hg::regular_grid_graph_4d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");


    add_type_overloads<def_max_tree<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_max_tree<hg::regular_grid_graph_1d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_max_tree<hg::regular_grid_graph_2d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_max_tree<hg::regular_grid_graph_3d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_max_tree<hg::regular_grid_graph_4d >, HG_TEMPLATE_NUMERIC_TYPES>(m, "");

}



