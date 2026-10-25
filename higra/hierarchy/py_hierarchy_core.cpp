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
namespace py_hierarchy_core {
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

    // Validate the permutation before any indexed scatter or parallel comparison.
    auto boruvka_inverse_order(const xt::pytensor<hg::index_t, 1> &order, size_t m) {
        hg_assert(order.size() == m, "Edge order must contain every edge exactly once.");
        hg::array_1d<hg::index_t> ranks = hg::array_1d<hg::index_t>::from_shape({m});
        ranks.fill(hg::invalid_index);
        for (size_t i = 0; i < m; ++i) {
            auto e = order(i);
            hg_assert(e >= 0 && static_cast<size_t>(e) < m, "Invalid edge index in edge order.");
            hg_assert(ranks(e) == hg::invalid_index, "Edge order must be a permutation.");
            ranks(e) = i;
        }
        return ranks;
    }

    struct def_bpt_boruvka {
        template<typename T, typename C>
        static void def(C &m, const char *doc) {
            m.def("_bpt_canonical_boruvka", [](
                    const xt::pytensor<hg::index_t, 1> &sources,
                    const xt::pytensor<hg::index_t, 1> &targets,
                    hg::index_t n, const xt::pytensor<T, 2> &weights, bool stable) {
                hg_assert(static_cast<size_t>(weights.shape()[0]) == sources.size(), "Incorrect edge weight count.");
                for (const auto &w: weights) {
                    if constexpr (std::is_floating_point<T>::value) {
                        hg_assert(!std::isnan(w), "Edge weights cannot contain NaNs.");
                    }
                }
                auto less = [&weights](hg::index_t a, hg::index_t b) {
                    for (hg::index_t k = 0; k < weights.shape()[1]; ++k) {
                        if (weights(a, k) < weights(b, k)) return true;
                        if (weights(b, k) < weights(a, k)) return false;
                    }
                    return a < b;
                };
                auto res = hg::hierarchy_core_internal::bpt_boruvka_from_edge_order(
                        sources, targets, n, less,
                        stable ? hg::hierarchy_core_internal::bpt_boruvka_numbering::exact
                               : hg::hierarchy_core_internal::bpt_boruvka_numbering::basin);
                return py::make_tuple(std::move(res.first), std::move(res.second));
            }, doc, py::arg("sources"), py::arg("targets"), py::arg("num_vertices"),
               py::arg("weights").noconvert(), py::arg("stable") = true);
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
        c.def("node_map", [](class_t &self) -> hg::array_1d<hg::index_t> & { return self.node_map; },
              "A node map array.",
              py::return_value_policy::reference_internal);

    }

    void py_init_hierarchy_core(pybind11::module &m) {
        //xt::import_numpy();

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
            auto res = hg::hierarchy_core_internal::bpt_canonical_from_sorted_edges(sources, targets,
                                                                                    sorted_edge_indices,
                                                                                    num_vertices);
            return py::make_tuple(std::move(res.first), std::move(res.second));
        });

        add_type_overloads<def_bpt_boruvka, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        m.def("_bpt_canonical_boruvka_from_order", [](
                const xt::pytensor<hg::index_t, 1> &sources,
                const xt::pytensor<hg::index_t, 1> &targets,
                hg::index_t n, const xt::pytensor<hg::index_t, 1> &order, bool stable) {
            auto ranks = boruvka_inverse_order(order, sources.size());
            auto less = [&ranks](hg::index_t a, hg::index_t b) { return ranks(a) < ranks(b); };
            auto res = hg::hierarchy_core_internal::bpt_boruvka_from_edge_order(
                    sources, targets, n, less,
                    stable ? hg::hierarchy_core_internal::bpt_boruvka_numbering::exact
                           : hg::hierarchy_core_internal::bpt_boruvka_numbering::basin);
            return py::make_tuple(std::move(res.first), std::move(res.second));
        }, py::arg("sources"), py::arg("targets"), py::arg("num_vertices"),
           py::arg("sorted_edge_indices"), py::arg("stable") = true);

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
}