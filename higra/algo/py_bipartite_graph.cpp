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

namespace py_bipartite_graph {

    namespace py = pybind11;

    using namespace hg;


    void py_init_bipartite_graph(pybind11::module &m) {

        m.def("_is_bipartite_graph", [](const hg::ugraph &g) {
                  return hg::is_bipartite_graph(g);
              },
              "Check if the graph is bipartite.",
              py::arg("graph"));

        m.def("_is_bipartite_graph",
              [](const hg::array_view<const index_t> &sources, const hg::array_view<const index_t> &targets,
                 index_t num_vertices) {
                  auto xsources = py_args::fixed_rank<1>(sources, "sources");
                  auto xtargets = py_args::fixed_rank<1>(targets, "targets");
                  hg_assert(sources.empty() || hg::min(sources) >= 0, "Source vertex index cannot be negative.");
                  hg_assert(targets.empty() || hg::min(targets) >= 0, "Target vertex index cannot be negative.");
                  hg_assert(sources.empty() || hg::max(sources) < num_vertices,
                            "Source vertex index must be less than the number of vertices.");
                  hg_assert(targets.empty() || hg::max(targets) < num_vertices,
                            "Target vertex index must be less than the number of vertices.");
                  return hg::is_bipartite_graph(xsources, xtargets, num_vertices);
              },
              "Check if the graph is bipartite.",
              py::arg("sources"),
              py::arg("targets"),
              py::arg("num_vertices"));

        m.def("_bipartite_graph_matching",
              [](
                      const hg::array_view<const index_t> &sources,
                      const hg::array_view<const index_t> &targets,
                      index_t num_vertices,
                      const hg::array_view<const index_t> &weights) {
                  auto xsources = py_args::fixed_rank<1>(sources, "sources");
                  auto xtargets = py_args::fixed_rank<1>(targets, "targets");
                  auto xweights = py_args::fixed_rank<1>(weights, "weights");
                  hg_assert(sources.empty() || hg::min(sources) >= 0, "Source vertex index cannot be negative.");
                  hg_assert(targets.empty() || hg::min(targets) >= 0, "Target vertex index cannot be negative.");
                  hg_assert(sources.empty() || hg::max(sources) < num_vertices,
                            "Source vertex index must be less than the number of vertices.");
                  hg_assert(targets.empty() || hg::max(targets) < num_vertices,
                            "Target vertex index must be less than the number of vertices.");
                  hg::graph_algorithms::CSA csa(xsources, xtargets, num_vertices, xweights);
                  return csa.edge_indices();
              },
              "Compute a minimum weight matching in a bipartite graph.",
              py::arg("sources"),
              py::arg("targets"),
              py::arg("num_vertices"),
              py::arg("weights")
        );
    }
}