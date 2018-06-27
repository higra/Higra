//
// Created by user on 4/1/18.
//

#include "py_regular_graph.hpp"
#include "py_common_graph.hpp"

namespace py = pybind11;


template<int dim>
void py_init_regular_graph_impl(pybind11::module &m) {

    using embedding_t = hg::embedding_grid<dim>;
    using graph_t = hg::regular_graph<embedding_t>;
    using point_t = typename embedding_t::point_type;


    auto c = py::class_<graph_t>(m, ("RegularGraph" + std::to_string(dim) + "d").c_str());

    c.def(py::init([](const embedding_t &e, const std::vector<std::vector<long>> &pl) {
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

    c.def(py::init([](const std::vector<hg::size_t> &shape, const std::vector<std::vector<long>> &pl) {
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