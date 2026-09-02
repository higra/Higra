/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Wonder Alexandre Luz Alves                              *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_component_tree_dual_filter.hpp"
#include "higra/hierarchy/component_tree_casf.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py_component_tree_dual_filter {
    template<typename T>
    using pyarray = xt::pyarray<T>;

    namespace py = pybind11;

    template<typename graph_t>
    struct def_connected_alternating_sequential_filter {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_connected_alternating_sequential_filter",
                  [](const graph_t &graph,
                     const pyarray<value_t> &vertex_weights,
                     hg::ComponentTreeCasfAttribute attribute,
                     const std::vector<double> &thresholds) {
                        hg::ComponentTreeCasf<value_t, graph_t> casf(graph, vertex_weights, attribute);
                        return casf.filter(thresholds);
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("vertex_weights").noconvert(),
                  py::arg("attribute"),
                  py::arg("thresholds"));
        }
    };

    void py_init_component_tree_dual_filter(pybind11::module &m) {
        py::enum_<hg::ComponentTreeCasfAttribute>(m, "CasfAttribute")
                .value("area", hg::ComponentTreeCasfAttribute::area)
                .value("bounding_box_width", hg::ComponentTreeCasfAttribute::bounding_box_width)
                .value("bounding_box_height", hg::ComponentTreeCasfAttribute::bounding_box_height)
                .value("bounding_box_diagonal", hg::ComponentTreeCasfAttribute::bounding_box_diagonal);

        add_type_overloads<def_connected_alternating_sequential_filter<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "Connected alternating sequential filter on a vertex weighted graph.");
        add_type_overloads<def_connected_alternating_sequential_filter<hg::regular_grid_graph_2d>, HG_TEMPLATE_NUMERIC_TYPES>(m,"Connected alternating sequential filter on a regular grid graph.");
    }
}
