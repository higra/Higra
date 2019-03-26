/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_lca_fast.hpp"
#include "../py_common.hpp"
#include "higra/graph.hpp"
#include "higra/structure/lca_fast.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;
using namespace hg;

template<typename T>
using pyarray = xt::pyarray<T>;

struct def_lca_vertices {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("lca", [](const lca_fast &l,
                        const pyarray<value_t> &vertices1,
                        const pyarray<value_t> &vertices2) {
                  return l.lca(vertices1, vertices2);
              },
              doc,
              pybind11::arg("vertices1"),
              pybind11::arg("vertices2"));
    }
};

void py_init_lca_fast(pybind11::module &m) {
    xt::import_numpy();
    auto c = py::class_<lca_fast>(m, "LCAFast");

    c.def(py::init<tree>(),
          "Preprocess the given tree in order for fast lowest common ancestor (LCA) computation.",
          py::arg("tree"));

    c.def("lca",
          [](const lca_fast &l, index_t v1, index_t v2) { return l.lca(v1, v2); },
          "Get LCA of given two vertices.",
          py::arg("v1"),
          py::arg("v2"));

    c.def("lca",
          [](const lca_fast &l, const ugraph &g) { return l.lca(edge_iterator(g)); },
          "Compute the LCA of every edge of the given graph.",
          py::arg("UndirectedGraph"));

    add_type_overloads<def_lca_vertices, int, unsigned int, long long, unsigned long long>
            (c, "Given two 1d array of graph vertex indices v1 and v2, both containing n elements, "
                "this function returns a 1d array or tree vertex indices of size n such that: \n"
                "for all i in 0..n-1, res(i) = lca(v1(i); v2(i)).");

}
