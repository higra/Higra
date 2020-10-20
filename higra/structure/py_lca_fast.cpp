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
                  hg_assert((xt::amin)(vertices1)() >= 0, "Vertex indices cannot be negative.");
                  hg_assert((index_t) (index_t) (xt::amax)(vertices1)() < (index_t) l.num_vertices(),
                            "Vertex indices must be smaller than the number of vertices in the tree.");
                  hg_assert((xt::amin)(vertices2)() >= 0, "Vertex indices cannot be negative.");
                  hg_assert((index_t) (xt::amax)(vertices2)() < (index_t) l.num_vertices(),
                            "Vertex indices must be smaller than the number of vertices in the tree.");
                  return l.lca(vertices1, vertices2);
              },
              doc,
              pybind11::arg("vertices1"),
              pybind11::arg("vertices2"));
    }
};

void py_init_lca_fast(pybind11::module &m) {
    xt::import_numpy();
    auto c = py::class_<lca_fast>(m, "LCAFast",
                                  "Provides fast :math:`\\mathcal{O}(1)` lowest common ancestor computation in a tree thanks "
                                  "to a linearithmic preprocessing of the tree.",
                                  py::dynamic_attr());

    c.def(py::init<tree>(),
          "Preprocess the given tree in order for fast lowest common ancestor (LCA) computation.\n\n"
          "Consider using the function :func:`~higra.make_lca_fast` instead of calling this constructor to"
          "avoid preprocessing the same tree several times.",
          py::arg("tree"));

    c.def("lca",
          [](const lca_fast &l, index_t v1, index_t v2) {
              hg_assert(v1 >= 0 && v2 >= 0, "Vertex indices cannot be negative.");
              hg_assert(v1 < (index_t) l.num_vertices() && v2 < (index_t) l.num_vertices(),
                        "Vertex indices must be smaller than the number of vertices in the tree.");
              return l.lca(v1, v2);
          },
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

    c.def("_get_state",
          [](const lca_fast &l) {
              auto state = l.get_state();
              return py::make_tuple(
                      state.m_num_vertices,
                      std::move(state.m_Euler),
                      std::move(state.m_Depth),
                      std::move(state.m_Represent),
                      std::move(state.m_Number),
                      std::move(state.m_Minim),
                      state.m_rep);
          },
          "Return an opaque structure representing the internal state of the object");


    c.def_static("_make_from_state",
                 [](size_t &a0,
                    const pyarray<index_t> &a1,
                    const pyarray<index_t> &a2,
                    const pyarray<index_t> &a3,
                    const pyarray<index_t> &a4,
                    const pyarray<index_t> &a5,
                    size_t &a6) {

                     return hg::lca_fast::make_lca_fast(
                             lca_fast::internal_state<pyarray<index_t>, pyarray<index_t>>(
                                     a0, a1, a2, a3, a4, a5, a6));

                 },
                 "Create a new lca_fast object from the saved state (see function get_state)");


}
