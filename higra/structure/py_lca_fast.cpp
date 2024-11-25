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

namespace py_lca_fast {
    namespace py = pybind11;
    using namespace hg;

    template<typename T>
    using pyarray = xt::pyarray<T>;

    template<typename T>
    using pyarray_1d = xt::pytensor<T, 1>;

    template<typename lca_t>
    struct def_lca_vertices {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("lca", [](const lca_t &l,
                            const pyarray<value_t> &vertices1,
                            const pyarray<value_t> &vertices2) {
                      hg_assert((xt::amin)(vertices1)() >= 0, "Vertex indices cannot be negative.");
                      hg_assert((index_t) (index_t) (xt::amax)(vertices1)() < (index_t) l.num_elements(),
                                "Vertex indices must be smaller than the number of vertices in the tree.");
                      hg_assert((xt::amin)(vertices2)() >= 0, "Vertex indices cannot be negative.");
                      hg_assert((index_t) (xt::amax)(vertices2)() < (index_t) l.num_elements(),
                                "Vertex indices must be smaller than the number of vertices in the tree.");
                      return l.lca(vertices1, vertices2);
                  },
                  doc,
                  pybind11::arg("vertices1"),
                  pybind11::arg("vertices2"));
        }
    };

    using namespace range_minimum_query_internal;


    auto get_rmq_state_to_python(const rmq_sparse_table<index_t>::internal_state<array_1d> &state) {
        py::list list;
        for (auto &t: state.sparse_table) {
            list.append(std::move(t));
        }
        return list;
    }


    auto get_rmq_state_to_python(const rmq_sparse_table_block<index_t>::internal_state<array_1d> &state) {
        py::list list;

        list.append(state.data_size);
        list.append(state.block_size);
        list.append(state.num_blocks);
        list.append(std::move(state.block_minimum_prefix));
        list.append(std::move(state.block_minimum_suffix));
        list.append(get_rmq_state_to_python(state.sparse_table));

        return list;
    }

    template<typename rmq_t>
    auto get_rmq_state_from_python(const py::list &list);

    template<>
    auto get_rmq_state_from_python<range_minimum_query_internal::rmq_sparse_table<index_t>>(const py::list &list) {

        std::vector<pyarray_1d<size_t>> sp;
        for (auto &e: list) {
            sp.push_back(e.template cast<pyarray_1d<size_t>>());
        }
        return range_minimum_query_internal::rmq_sparse_table<index_t>::internal_state<pyarray_1d>(sp);
    }

    template<>
    auto
    get_rmq_state_from_python<range_minimum_query_internal::rmq_sparse_table_block<index_t>>(const py::list &list) {
        return range_minimum_query_internal::rmq_sparse_table_block<index_t>::internal_state<pyarray_1d>(
                list[0].template cast<index_t>(),
                list[1].template cast<index_t>(),
                list[2].template cast<index_t>(),
                list[3].template cast<pyarray_1d<index_t>>(),
                list[4].template cast<pyarray_1d<index_t>>(),
                get_rmq_state_from_python<range_minimum_query_internal::rmq_sparse_table<index_t>>(
                        list[5].template cast<py::list>())
        );
    }

    template<typename T>
    auto get_lca_state_to_python(const T &state) {
        py::list list;
        list.append(std::move(state.tree_Euler_tour_map));
        list.append(std::move(state.tree_Euler_tour_depth));
        list.append(std::move(state.first_visit_in_Euler_tour));
        list.append(get_rmq_state_to_python(state.rmq_state));
        return list;
    }

    template<typename lca_t>
    auto get_lca_state_from_python(const py::list &list) {
        using state_type = typename lca_t::template internal_state<pyarray_1d>;
        return state_type(
                list[0].template cast<pyarray_1d<index_t>>(),
                list[1].template cast<pyarray_1d<index_t>>(),
                list[2].template cast<pyarray_1d<index_t>>(),
                get_rmq_state_from_python<typename lca_t::rmq_type>(list[3])
        );
    }

    template<typename lca_t, typename T>
    auto def_lca_t(T &m, const char *name, const char *doc) {
        auto c = py::class_<lca_t>(m, name, doc, py::dynamic_attr());

        c.def(py::init<tree>(),
              "Preprocess the given tree in order for fast lowest common ancestor (LCA) computation.\n\n"
              "Consider using the function :func:`~higra.Tree.lowest_ancestor_preprocess` instead of calling this constructor to"
              "avoid preprocessing the same tree several times.",
              py::arg("tree"));

        c.def("lca",
              [](const lca_t &l, index_t v1, index_t v2) {
                  hg_assert(v1 >= 0 && v2 >= 0, "Vertex indices cannot be negative.");
                  hg_assert(v1 < (index_t) l.num_elements() && v2 < (index_t) l.num_elements(),
                            "Vertex indices must be smaller than the number of vertices in the tree.");
                  return l.lca(v1, v2);
              },
              "Get LCA of given two vertices.",
              py::arg("v1"),
              py::arg("v2"));

        c.def("lca",
              [](const lca_t &l, const ugraph &g) { return l.lca(edge_iterator(g)); },
              "Compute the LCA of every edge of the given graph.",
              py::arg("UndirectedGraph"));

        add_type_overloads<def_lca_vertices<lca_t>, int, unsigned int, long long, unsigned long long>
                (c, "Given two 1d array of graph vertex indices v1 and v2, both containing n elements, "
                    "this function returns a 1d array or tree vertex indices of size n such that: \n"
                    "for all i in 0..n-1, res(i) = lca(v1(i); v2(i)).");

        c.def("_get_state",
              [](const lca_t &l) {
                  return py::make_tuple(get_lca_state_to_python(l.get_state()));
              },
              "Return an opaque structure representing the internal state of the object");

        c.def_static("_make_from_state",
                     [](py::tuple &t) {
                         return lca_t::make_from_state(
                                 get_lca_state_from_python<lca_t>(t[0].template cast<py::list>()));
                     },
                     "Create a new lca_fast object from the saved state (see function get_state)");

        return c;
    }


    void py_init_lca_fast(pybind11::module &m) {
        //xt::import_numpy();

        auto c_lca_sp = def_lca_t<lca_sparse_table>(
                m, "LCA_rmq_sparse_table",
                "Provides fast :math:`\\mathcal{O}(1)` lowest common ancestor computation in a tree thanks "
                "to a linearithmic preprocessing of the tree.");

        auto c_lca_spb = def_lca_t<lca_sparse_table_block>(
                m, "LCA_rmq_sparse_table_block",
                "Provides fast :math:`\\mathcal{O}(1)` lowest common ancestor computation in a tree thanks "
                "to a linear preprocessing of the tree.");

        c_lca_spb.def(py::init([](const tree &t, size_t block_size) {
                          return lca_sparse_table_block(t, block_size);
                      }),
                      "Preprocess the given tree in order for fast lowest common ancestor (LCA) computation.\n\n"
                      "Consider using the function :func:`~higra.Tree.lowest_ancestor_preprocess` instead of calling this constructor to"
                      "avoid preprocessing the same tree several times.",
                      py::arg("tree"),
                      py::arg("block_size"));

        // @TODO export symbol LCAFast python

    }
}
