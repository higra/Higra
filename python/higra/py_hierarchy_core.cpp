//
// Created by user on 4/14/18.
//

#include "py_hierarchy_core.hpp"
#include "py_common.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "xtensor-python/pyarray.hpp"


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

void py_init_hierarchy_core(pybind11::module &m) {
    xt::import_numpy();

#define DEF(rawXKCD, dataXKCD, type) \
        def_bptCanonical<hg::ugraph,type>(m);
    HG_FOREACH(DEF, HG_NUMERIC_TYPES);
#undef DEF
}