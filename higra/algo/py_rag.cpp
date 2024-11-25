/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_rag.hpp"
#include "higra/algo/rag.hpp"
#include "higra/algo/alignment.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


namespace py_rag {

// @TODO Remove layout_type when xtensor solves the issue with iterators
    template<typename T>
    using pyarray = xt::pyarray<T, xt::layout_type::row_major>;

    namespace py = pybind11;

    template<typename graph_t>
    struct def_make_rag {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_make_region_adjacency_graph_from_labelisation",
                  [](const graph_t &graph, const pyarray<value_t> &input) {
                      auto res = hg::make_region_adjacency_graph_from_labelisation(graph, input);
                      return py::make_tuple(std::move(res.rag), std::move(res.vertex_map), std::move(res.edge_map));
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("vertex_labels"));
        }
    };

    template<typename graph_t>
    struct def_make_rag_cut {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_make_region_adjacency_graph_from_graph_cut",
                  [](const graph_t &graph, const pyarray<value_t> &input) {
                      auto res = hg::make_region_adjacency_graph_from_graph_cut(graph, input);
                      return py::make_tuple(std::move(res.rag), std::move(res.vertex_map), std::move(res.edge_map));
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("edge_weights"));
        }
    };

    struct def_rag_back_project_weights {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_rag_back_project_weights",
                  [](const xt::pyarray<hg::index_t> &rag_map, const pyarray<value_t> &rag_weights) {
                      return hg::rag_back_project_weights(rag_map, rag_weights);
                  },
                  doc,
                  py::arg("rag_map"),
                  py::arg("rag_weights"));
        }
    };


    void py_init_rag(pybind11::module &m) {
        //xt::import_numpy();

        add_type_overloads<def_make_rag<hg::ugraph>, HG_TEMPLATE_INTEGRAL_TYPES>
                (m,
                 "Create a region adjacency graph of the input graph with regions identified by the provided vertex labels.");

        add_type_overloads<def_make_rag_cut<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>
                (m,
                 "Create a region adjacency graph of the input graph with regions identified by the provided graph cut.");

        add_type_overloads<def_rag_back_project_weights, HG_TEMPLATE_NUMERIC_TYPES>
                (m,
                 "Projects vertex or edge weights defined on a region adjacency graph back to the original graph space.");
    }
}


