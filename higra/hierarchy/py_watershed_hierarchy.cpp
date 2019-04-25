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
#include "../py_common.hpp"
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
                 const std::function<pyarray<double>(const hg::tree &,
                                                     const hg::array_1d<value_t> &)> &attribute_functor) {
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
                 const pyarray<double> &vertex_area) {
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
                 const pyarray<double> &vertex_area) {
                  return hg::watershed_hierarchy_by_volume(graph, edge_weights, vertex_area);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"),
              py::arg("vertex_area"));
    }
};

template<typename graph_t>
struct def_watershed_hierarchy_by_minima_ordering {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_watershed_hierarchy_by_minima_ordering",
              [](const graph_t &graph,
                 const pyarray<value_t> &edge_weights,
                 const pyarray<size_t> &minima_ranks,
                 const pyarray<double> &minima_altitudes) {
                  return hg::watershed_hierarchy_by_minima_ordering(graph, edge_weights, minima_ranks,
                                                                    minima_altitudes);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"),
              py::arg("minima_ranks"),
              py::arg("minima_ordering"));
    }
};

void py_init_watershed_hierarchy(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_watershed_hierarchy_by_attribute<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");

    add_type_overloads<def_watershed_hierarchy_by_minima_ordering<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");

    add_type_overloads<def_watershed_hierarchy_by_dynamics<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");

    add_type_overloads<def_watershed_hierarchy_by_area<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");

    add_type_overloads<def_watershed_hierarchy_by_volume<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
}



