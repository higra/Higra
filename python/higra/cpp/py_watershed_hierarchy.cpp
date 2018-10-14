/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_watershed_hierarchy.hpp"
#include "higra/hierarchy/watershed_hierarchy.hpp"
#include "py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

template<typename graph_t>
struct def_watershed_hierarchy_by_attribute {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_watershed_hierarchy_by_attribute",
              [](const graph_t &graph,
                 const pyarray<value_t> &edge_weights,
                 // FIXME can we do better for return type ?
                 const std::function<pyarray<double>(const hg::tree &, const hg::array_1d<value_t>&)> &attribute_functor) {
                  return hg::watershed_hierarchy_by_attribute(graph, edge_weights, attribute_functor);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"),
              py::arg("attribute_functor"));
    }
};

template<typename graph_t>
struct def_watershed_hierarchy_by_dynamics {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_watershed_hierarchy_by_dynamics",
              [](const graph_t &graph,
                 const pyarray<value_t> &edge_weights) {
                  return hg::watershed_hierarchy_by_dynamics(graph, edge_weights);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"));
    }
};

template<typename graph_t>
struct def_watershed_hierarchy_by_area {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_watershed_hierarchy_by_area",
              [](const graph_t &graph,
                 const pyarray<value_t> &edge_weights,
                 const pyarray<long> & vertex_area) {
                  return hg::watershed_hierarchy_by_area(graph, edge_weights, vertex_area);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"),
              py::arg("vertex_area"));
    }
};

template<typename graph_t>
struct def_watershed_hierarchy_by_volume {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_watershed_hierarchy_by_volume",
              [](const graph_t &graph,
                 const pyarray<value_t> &edge_weights,
                 const pyarray<long> & vertex_area) {
                  return hg::watershed_hierarchy_by_volume(graph, edge_weights, vertex_area);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"),
              py::arg("vertex_area"));
    }
};

void py_init_watershed_hierarchy(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_watershed_hierarchy_by_attribute<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Compute the watershed hierarchy by a user defined attributes.\n"
             "\n"
             ":param graph: input graph\n"
             ":param edge_weights: edge weights of the input graph\n"
             ":attribute_functor: a function that takes two arguments "
             "a binary partition tree and its node altitudes and returns "
             "an array with the node attribute values.\n"
             ":return: a pair (tree, altitudes)");

    add_type_overloads<def_watershed_hierarchy_by_dynamics<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Compute the watershed hierarchy by dynamics.\n"
             "\n"
             ":param graph: input graph\n"
             ":param edge_weights: edge weights of the input graph\n"
             ":return: a pair (tree, altitudes)");

    add_type_overloads<def_watershed_hierarchy_by_area<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Compute the watershed hierarchy by area.\n"
             "\n"
             ":param graph: input graph\n"
             ":param edge_weights: edge weights of the input graph\n"
             ":param vertex_area: vertex area of the input graph\n"
             ":return: a pair (tree, altitudes)");

    add_type_overloads<def_watershed_hierarchy_by_volume<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Compute the watershed hierarchy by volume.\n"
             "\n"
             ":param graph: input graph\n"
             ":param edge_weights: edge weights of the input graph\n"
             ":param vertex_area: vertex area of the input graph\n"
             ":return: a pair (tree, altitudes)");
}



