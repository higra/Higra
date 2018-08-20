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
#include "py_common.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "pybind11/functional.h"

namespace py = pybind11;

template<typename graph_t>
struct def_bptCanonical {
    template<typename value_t, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_bpt_canonical", [](const graph_t &graph, const xt::pyarray<value_t> &edge_weights) {
                  return hg::bpt_canonical(graph, edge_weights);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights")
        );
    }
};

template<typename graph_t>
struct def_simplify_tree {
    template<typename value_t, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_simplify_tree",
              [](const hg::tree &t, xt::pyarray<value_t> &criterion) {
                  return hg::simplify_tree(t, criterion);
              },
              doc,
              py::arg("tree"),
              py::arg("deleted_nodes")
        );
    }
};

void py_init_hierarchy_core(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_bptCanonical<hg::ugraph>, HG_TEMPLATE_SNUMERIC_TYPES>
            (m,
             "Compute the canonical binary partition tree (binary tree by altitude ordering) of the given weighted graph.\n "
                     "Returns a tuple of 3 elements: (tree, node altitudes, minimum spanning tree)"
            );

    add_type_overloads<def_simplify_tree<hg::ugraph>, HG_TEMPLATE_SNUMERIC_TYPES>
            (m,
             "Creates a copy of the current Tree and deletes the nodes such that the criterion function is true.\n"
                     "Also returns an array that maps any node index i of the new tree, to the index of this node in the original tree\n"
                     "\n"
                     "The criterion is an array that associates true (this node must be deleted) or\n"
                     "false (do not delete this node) to a node index."
            );

}