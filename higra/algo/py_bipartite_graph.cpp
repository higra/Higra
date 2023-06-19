/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_alignement.hpp"
#include "../py_common.hpp"
#include "higra/algo/bipartite_graph.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

template<typename T>
using pytensor_1d = xt::pytensor<T, 1>;

namespace py = pybind11;

using namespace hg;


void py_init_bipartite_graph(pybind11::module &m) {
    xt::import_numpy();

    m.def("_is_bipartite_graph", [](const hg::ugraph &g) {
              return hg::is_bipartite_graph(g);
          },
          "Check if the graph is bipartite.",
          py::arg("graph"));

    m.def("_is_bipartite_graph",
          [](const pytensor_1d<index_t> &sources, const pytensor_1d<index_t> &targets, index_t num_vertices) {
              hg_assert((xt::amin)(sources)() >= 0, "Source vertex index cannot be negative.");
              hg_assert((xt::amin)(targets)() >= 0, "Target vertex index cannot be negative.");
              hg_assert((xt::amax)(sources)() < num_vertices, "Source vertex index must be less than the number of vertices.");
              hg_assert((xt::amax)(targets)() < num_vertices, "Target vertex index must be less than the number of vertices.");
              return hg::is_bipartite_graph(sources, targets, num_vertices);
          },
          "Check if the graph is bipartite.",
          py::arg("sources"),
          py::arg("targets"),
          py::arg("num_vertices"));

    m.def("_bipartite_graph_matching",
          [](
                  const pytensor_1d<index_t> &sources,
                  const pytensor_1d<index_t> &targets,
                  index_t num_vertices,
                  const pytensor_1d<index_t> &weights) {
              hg_assert((xt::amin)(sources)() >= 0, "Source vertex index cannot be negative.");
              hg_assert((xt::amin)(targets)() >= 0, "Target vertex index cannot be negative.");
              hg_assert((xt::amax)(sources)() < num_vertices, "Source vertex index must be less than the number of vertices.");
              hg_assert((xt::amax)(targets)() < num_vertices, "Target vertex index must be less than the number of vertices.");
              hg::graph_algorithms::CSA csa(sources, targets, num_vertices, weights);
              return csa.edge_indices();
          },
          "Compute a minimum weight matching in a bipartite graph.",
          py::arg("sources"),
          py::arg("targets"),
          py::arg("num_vertices"),
          py::arg("weights")
    );
}