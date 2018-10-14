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

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

template<typename tree_t>
struct def_node_weighted_tree_and_mst {
    template<typename value_t, typename M>
    static
    void def(M &m, const char *doc) {
        using class_t = hg::node_weighted_tree_and_mst<tree_t, hg::array_1d < value_t>, hg::ugraph>;
        auto c = py::class_<class_t>(m,
                                     (std::string("NodeWeightedTreeAndMST_") + typeid(class_t).name()).c_str(),
                                     "A simple structure to hold the result of canonical bpt construction algorithms, "
                                     "namely a tree, its associated node altitude array, and its associated MST.");
        c.def("tree", [](class_t &self) -> tree_t& {return self.tree;}, "The binary partition tree!");
        c.def("node_altitude", [](class_t &self) -> hg::array_1d<value_t>& {return self.node_altitude;}, "An array of tree node altitude.");
        c.def("mst", [](class_t &self) -> hg::ugraph& {return self.mst;}, "A minimum spanning tree associated to the binary partition tree.");
    }
};

template<typename graph_t>
struct def_bptCanonical {
    template<typename value_t, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_bpt_canonical", [](const graph_t &graph, const pyarray<value_t> &edge_weights) {
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
              [](const hg::tree &t, pyarray<value_t> &criterion) {
                  return hg::simplify_tree(t, criterion);
              },
              doc,
              py::arg("tree"),
              py::arg("deleted_nodes")
        );
    }
};

template <typename M>
void add_simplified_tree(M &m){
    using class_t = hg::simplified_tree<hg::tree, hg::array_1d<hg::index_t>>;
    auto c = py::class_<class_t>(m,
                                 "SimplifiedTree",
                                 "A simple structure to hold the result of hierarchy simplification algorithms, "
                                 "namely a simplified tree and a node map that gives for each node of"
                                 " the simplified tree the corresponding node index in the original tree.");
    c.def("tree", [](class_t &self) -> hg::tree& {return self.tree;}, "The tree!");
    c.def("node_map", [](class_t &self) -> hg::array_1d<hg::index_t>& {return self.node_map;}, "A node map array.");

}

void py_init_hierarchy_core(pybind11::module &m) {
    xt::import_numpy();
    add_type_overloads<def_node_weighted_tree_and_mst<hg::tree>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    add_type_overloads<def_bptCanonical<hg::ugraph>, HG_TEMPLATE_SNUMERIC_TYPES>
            (m,
             "Compute the canonical binary partition tree (binary tree by altitude ordering) of the given weighted graph.\n "
             "Returns a tuple of 3 elements: (tree, node altitudes, minimum spanning tree)"
            );
    add_simplified_tree(m);
    add_type_overloads<def_simplify_tree<hg::ugraph>, HG_TEMPLATE_SNUMERIC_TYPES>
            (m,
             "Creates a copy of the current Tree and deletes the nodes such that the criterion function is true.\n"
             "Also returns an array that maps any node index i of the new tree, to the index of this node in the original tree\n"
             "\n"
             "The criterion is an array that associates true (this node must be deleted) or\n"
             "false (do not delete this node) to a node index."
            );

}