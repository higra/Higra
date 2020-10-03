/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_regular_graph.hpp"
#include "py_common_graph.hpp"

namespace py = pybind11;


template<int dim>
void py_init_regular_graph_impl(pybind11::module &m) {

    using embedding_t = hg::embedding_grid<dim>;
    using graph_t = hg::regular_graph<embedding_t>;
    using point_t = typename embedding_t::point_type;


    auto c = py::class_<graph_t>(m,
                                 ("RegularGraph" + std::to_string(dim) + "d").c_str(),
                                 py::dynamic_attr());

    c.def(py::init([](const embedding_t &e, const std::vector<std::vector<hg::index_t>> &pl) {
                       std::vector<point_t> points;

                       for (const auto &v : pl) {
                           hg_assert(v.size() == dim, "Invalid dimension in point list.");
                           point_t p;
                           for (hg::index_t i = 0; i < dim; ++i)
                               p(i) = v[i];
                           points.push_back(p);
                       }
                       return graph_t(e, points);
                   }
          ),
          "Create a regular implicit graph from given embedding and neighbouring.",
          py::arg("embedding"),
          py::arg("neighbour_list"));

    c.def(py::init([](const std::vector<hg::size_t> &shape, const std::vector<std::vector<hg::index_t>> &pl) {
                       std::vector<point_t> points;

                       for (const auto &v : pl) {
                           hg_assert(v.size() == dim, "Invalid dimension in point list.");
                           point_t p;
                           for (hg::index_t i = 0; i < dim; ++i)
                               p(i) = v[i];
                           points.push_back(p);
                       }
                       return graph_t(embedding_t(shape), points);
                   }
          ),
          "Create a regular implicit graph from given shape and neighbouring.",
          py::arg("shape"),
          py::arg("neighbour_list"));

    c.def("as_explicit_graph",
          [](const graph_t &graph) {
              return hg::copy_graph<hg::ugraph>(graph);
          },
          "Converts the current regular graph instance to an equivalent explicit undirected graph.");

    add_edge_accessor_graph_concept<graph_t, decltype(c)>(c);
    add_incidence_graph_concept<graph_t, decltype(c)>(c);
    add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
    add_adjacency_graph_concept<graph_t, decltype(c)>(c);
    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
}


void py_init_regular_graph(pybind11::module &m) {
    xt::import_numpy();
    py_init_regular_graph_impl<1>(m);
    py_init_regular_graph_impl<2>(m);
    py_init_regular_graph_impl<3>(m);
    py_init_regular_graph_impl<4>(m);
    py_init_regular_graph_impl<5>(m);
}