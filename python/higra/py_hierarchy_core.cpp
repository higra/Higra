//
// Created by user on 4/14/18.
//

#include "py_hierarchy_core.hpp"
#include "py_common.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "pybind11/functional.h"

namespace py = pybind11;

template<typename graph_t, typename value_t>
void def_bptCanonical(pybind11::module &m) {
    m.def("bptCanonical", [](const graph_t &graph, const xt::pyarray<value_t> &edge_weights) {
              return hg::bptCanonical(graph, edge_weights);
          },
          "Compute the canonical binary partition tree (binary tree by altitude ordering) of the given weighted graph. "
                  "Returns a tuple of 3 elements: (tree, node altitudes, minimum spanning tree)",
          py::arg("graph"),
          py::arg("edgeWeights"));

}

template<typename value_t>
void def_simplify_tree(pybind11::module &m) {
    m.def("simplifyTree",
          [](const hg::tree &t, xt::pyarray<value_t> &criterion) {
              return hg::simplify_tree(t, criterion);
          },
          "Creates a copy of the current Tree and deletes the nodes such that the criterion function is true.\n"
                  "Also returns an array that maps any node index i of the new tree, to the index of this node in the original tree\n"
                  "\n"
                  "The criterion is an array that associates true (this node must be deleted) or\n"
                  "false (do not delete this node) to a node index.",
          py::arg("tree"),
          py::arg("deletedNodes")
    );
}

void py_init_hierarchy_core(pybind11::module &m) {
    xt::import_numpy();

#define DEF(rawXKCD, dataXKCD, type) \
        def_bptCanonical<hg::ugraph,type>(m);
    HG_FOREACH(DEF, HG_NUMERIC_TYPES);
#undef DEF

#define DEF(rawXKCD, dataXKCD, type) \
        def_simplify_tree<type>(m);
    HG_FOREACH(DEF, HG_INTEGRAL_TYPES);
#undef DEF
}