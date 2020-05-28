/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_graph_core.hpp"
#include "../py_common.hpp"
#include "higra/algo/graph_core.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

template<typename graph_t>
struct def_graph_cut_2_labelisation {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_graph_cut_2_labelisation", [](const graph_t &graph,
                                              const pyarray<value_t> &edge_weights) {
                  return hg::graph_cut_2_labelisation(graph, edge_weights);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"));
    }
};

template<typename graph_t>
struct def_labelisation_2_graph_cut {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_labelisation_2_graph_cut", [](const graph_t &graph,
                                              const pyarray<value_t> &vertex_labels) {
                  return hg::labelisation_2_graph_cut(graph, vertex_labels);
              },
              doc,
              py::arg("graph"),
              py::arg("vertex_labels"));
    }
};

template<typename graph_t>
struct def_minimum_spanning_tree {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_minimum_spanning_tree", [](const graph_t &graph,
                                           const pyarray<value_t> &edge_weights) {
                  auto res = hg::minimum_spanning_tree(graph, edge_weights);
                  return pybind11::make_tuple(std::move(res.mst), std::move(res.mst_edge_map));
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"));
    }
};

struct def_adjacency_matrix_2_ugraph {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_adjacency_matrix_2_undirected_graph", [](const pyarray<value_t> &adjacency_matrix,
                                                         const double non_edge_value) {
                  auto res = hg::adjacency_matrix_2_undirected_graph(adjacency_matrix, static_cast<value_t>(non_edge_value));
                  return pybind11::make_tuple(std::move(res.first), std::move(res.second));
              },
              doc,
              py::arg("adjacency_matrix"),
              py::arg("non_edge_value") = 0);
    }
};

void py_init_algo_graph_core(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_graph_cut_2_labelisation<hg::ugraph>,
            HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Labelize graph vertices according to the given graph cut. "
                     "Each edge having a non zero value in the given edge_weights "
                     "are assumed to be part of the cut."
            );

    add_type_overloads<def_labelisation_2_graph_cut<hg::ugraph>,
            HG_TEMPLATE_INTEGRAL_TYPES>
            (m,
             "Determine the graph cut that corresponds to a given labeling "
                     "of the graph vertices. "
                     "The result is a weighting of the graph edges where edges with "
                     "a non zero weight are part of the cut."
            );

    add_type_overloads<def_adjacency_matrix_2_ugraph,
            HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             ""
            );

    add_type_overloads<def_minimum_spanning_tree<hg::ugraph>,
            HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             ""
            );
}