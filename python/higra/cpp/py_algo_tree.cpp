//
// Created by perretb on 29/04/18.
//

#include "py_algo_tree.hpp"
#include "py_common.hpp"
#include "higra/graph.hpp"
#include "higra/algo/tree.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

void py_init_algo_tree(pybind11::module &m) {
    xt::import_numpy();

    m.def("test_tree_isomorphism", &hg::testTreeIsomorphism<hg::tree, hg::tree>,
          "Test if 2 trees are isomorph assuming that they share the same leaves.\n"
          "\n"
          "By this definition t1 is isomorph to t2 if there exist a bijection f from vertices(t1) to vertices(t2) such that:\n"
          "  1) for any leaf node n of t1, f(n) = n and n\n"
          "  2) for any node n of t1, f(t1.parent(n)) = t2.parent(f(n))\n"
          "\n"
          "Note that the root r node of a tree t is defined by t.parent(r) = r, thus 2) becomes\n"
          "  for the root node r1 of t1, f(r1) = t2.parent(f(r1)), i.e. f(r1) is the root node of t2",
          py::arg("tree1"),
          py::arg("tree2"));
}