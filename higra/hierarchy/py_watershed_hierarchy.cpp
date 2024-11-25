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
namespace py_watershed_hierarchy {
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
                      auto res = hg::watershed_hierarchy_by_attribute(graph, edge_weights, attribute_functor);
                      return py::make_tuple(
                              std::move(res.tree),
                              std::move(res.altitudes),
                              std::move(res.mst_edge_map)
                      );
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("edge_weights"),
                  py::arg("attribute_functor"));
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
                     const pyarray<size_t> &minima_ranks) {
                      auto res = hg::watershed_hierarchy_by_minima_ordering(graph, edge_weights, minima_ranks);
                      return py::make_tuple(
                              std::move(res.tree),
                              std::move(res.altitudes),
                              std::move(res.mst_edge_map)
                      );
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("edge_weights"),
                  py::arg("minima_ranks"));
        }
    };

    void py_init_watershed_hierarchy(pybind11::module &m) {
        //xt::import_numpy();

        add_type_overloads<def_watershed_hierarchy_by_attribute<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");

        add_type_overloads<def_watershed_hierarchy_by_minima_ordering<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    }
}


