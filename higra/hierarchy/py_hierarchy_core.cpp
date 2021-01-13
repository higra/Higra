/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_hierarchy_core.hpp"
#include "../py_common.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "pybind11/functional.h"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

template<typename graph_t>
struct def_quasi_flat_zone_hierarchy {
    template<typename value_t, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_quasi_flat_zone_hierarchy", [](const graph_t &graph, const pyarray<value_t> &edge_weights) {
                  auto res = hg::quasi_flat_zone_hierarchy(graph, edge_weights);
                  return py::make_tuple(std::move(res.tree), std::move(res.altitudes));
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights")
        );
    }
};

template<typename M>
void add_simplified_tree(M &m) {
    using class_t = hg::remapped_tree<hg::tree, hg::array_1d<hg::index_t>>;
    auto c = py::class_<class_t>(m,
                                 "SimplifiedTree",
                                 "A simple structure to hold the result of hierarchy simplification algorithms, "
                                 "namely a simplified tree and a node map that gives for each node of"
                                 " the simplified tree the corresponding node index in the original tree.");
    c.def("tree", [](class_t &self) -> hg::tree & { return self.tree; }, "The tree!",
          py::return_value_policy::reference_internal);
    c.def("node_map", [](class_t &self) -> hg::array_1d<hg::index_t> & { return self.node_map; }, "A node map array.",
          py::return_value_policy::reference_internal);

}

void py_init_hierarchy_core(pybind11::module &m) {
    xt::import_numpy();

    m.def("_bpt_canonical", [](const xt::pytensor<hg::index_t, 1> &sources,
                               const xt::pytensor<hg::index_t, 1> &targets,
                               const xt::pytensor<hg::index_t, 1> &sorted_edge_indices,
                               const hg::index_t num_vertices) {
        hg_assert(num_vertices >= 0, "Number of vertices must be a positive number.");
        hg_assert((xt::amin)(sources)() >= 0, "Source vertex index cannot be negative.");
        hg_assert((xt::amin)(targets)() >= 0, "Target vertex index cannot be negative.");
        hg_assert((xt::amin)(sorted_edge_indices)() >= 0, "Edge index cannot be negative.");
        hg_assert((xt::amax)(sources)() < num_vertices,
                  "Source vertex index must be less than the number of vertices.");
        hg_assert((xt::amax)(targets)() < num_vertices,
                  "Target vertex index must be less than the number of vertices.");
        hg_assert((xt::amax)(sorted_edge_indices)() < (hg::index_t) sorted_edge_indices.size(),
                  "Edge index must be smaller than the number of edges in the graph/tree.");
        auto res = hg::hierarchy_core_internal::bpt_canonical_from_sorted_edges(sources, targets, sorted_edge_indices,
                                                                                num_vertices);
        return py::make_tuple(std::move(res.first), std::move(res.second));
    });

    add_simplified_tree(m);
    m.def("_simplify_tree",
          [](const hg::tree &t, pyarray<bool> &criterion, bool process_leaves) {
              return hg::simplify_tree(t, criterion, process_leaves);
          },
          "",
          py::arg("tree"),
          py::arg("deleted_nodes"),
          py::arg("process_leaves"));

    add_type_overloads<def_quasi_flat_zone_hierarchy<hg::ugraph>, HG_TEMPLATE_SNUMERIC_TYPES>
            (m,
             "Compute the quasi flat zones hierarchy of the given weighted graph."
            );

    m.def("_tree_2_binary_tree",
          [](const hg::tree &t) {
              return hg::tree_2_binary_tree(t);
          },
          "",
          py::arg("tree")
    );
}