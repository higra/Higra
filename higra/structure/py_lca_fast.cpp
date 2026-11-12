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
// return converter of the xtensor results of lca queries (until the alias flip)
#include "xtensor-python/pytensor.hpp"

namespace py_lca_fast {
    namespace py = pybind11;
    using namespace hg;

    template<typename lca_t>
    struct def_lca_vertices {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("lca", [](const lca_t &l,
                            const hg::array_view<const value_t> &vertices1,
                            const hg::array_view<const value_t> &vertices2) {
                      hg_assert(vertices1.empty() || hg::min(vertices1) >= 0, "Vertex indices cannot be negative.");
                      hg_assert(vertices1.empty() || (index_t) hg::max(vertices1) < (index_t) l.num_elements(),
                                "Vertex indices must be smaller than the number of vertices in the tree.");
                      hg_assert(vertices2.empty() || hg::min(vertices2) >= 0, "Vertex indices cannot be negative.");
                      hg_assert(vertices2.empty() || (index_t) hg::max(vertices2) < (index_t) l.num_elements(),
                                "Vertex indices must be smaller than the number of vertices in the tree.");
                      return l.lca(hg::bridge::as_xt(vertices1), hg::bridge::as_xt(vertices2));
                  },
                  doc,
                  pybind11::arg("vertices1"),
                  pybind11::arg("vertices2"));
        }
    };

    using namespace range_minimum_query_internal;

    /*
     * Pickling of the internal state.
     *
     * Format (unchanged since the first pickled versions):
     *  - LCA state: [tree_Euler_tour_map (int64), tree_Euler_tour_depth (int64),
     *                first_visit_in_Euler_tour (int64), rmq_state];
     *  - sparse table state: [level_0 (uint64), level_1 (uint64), ...];
     *  - sparse table block state: [data_size (int), block_size (int), num_blocks (int),
     *                block_minimum_prefix (int64), block_minimum_suffix (int64), sparse table state].
     *
     * Arrays are returned as copies; incoming arrays are converted to the expected dtype if needed and copied.
     */

    template<typename T>
    hg::array<T> state_array_to_python(const array_1d<T> &a) {
        return hg::bridge::to_array(a);
    }

    template<typename T>
    array_1d<T> state_array_from_python(const py::handle &h) {
        py::detail::make_caster<hg::array_view<const T>> caster;
        if (!caster.load(h, true)) {
            throw py::cast_error("Invalid LCA state: an element cannot be converted to a 1d array.");
        }
        const hg::array_view<const T> &v = py::detail::cast_op<const hg::array_view<const T> &>(caster);
        return hg::bridge::to_xt<1>(v);
    }

    py::list get_rmq_state_to_python(const rmq_sparse_table<index_t>::internal_state &state) {
        py::list list;
        for (auto &t: state.sparse_table) {
            list.append(state_array_to_python(t));
        }
        return list;
    }


    py::list get_rmq_state_to_python(const rmq_sparse_table_block<index_t>::internal_state &state) {
        py::list list;

        list.append(state.data_size);
        list.append(state.block_size);
        list.append(state.num_blocks);
        list.append(state_array_to_python(state.block_minimum_prefix));
        list.append(state_array_to_python(state.block_minimum_suffix));
        list.append(get_rmq_state_to_python(state.sparse_table));

        return list;
    }

    template<typename rmq_t>
    auto get_rmq_state_from_python(const py::list &list);

    template<>
    auto get_rmq_state_from_python<range_minimum_query_internal::rmq_sparse_table<index_t>>(const py::list &list) {

        std::vector<array_1d<size_t>> sp;
        for (auto &e: list) {
            sp.push_back(state_array_from_python<size_t>(e));
        }
        return range_minimum_query_internal::rmq_sparse_table<index_t>::internal_state(std::move(sp));
    }

    template<>
    auto
    get_rmq_state_from_python<range_minimum_query_internal::rmq_sparse_table_block<index_t>>(const py::list &list) {
        return range_minimum_query_internal::rmq_sparse_table_block<index_t>::internal_state(
                list[0].template cast<index_t>(),
                list[1].template cast<index_t>(),
                list[2].template cast<index_t>(),
                state_array_from_python<index_t>(list[3]),
                state_array_from_python<index_t>(list[4]),
                get_rmq_state_from_python<range_minimum_query_internal::rmq_sparse_table<index_t>>(
                        list[5].template cast<py::list>())
        );
    }

    template<typename T>
    py::list get_lca_state_to_python(const T &state) {
        py::list list;
        list.append(state_array_to_python(state.tree_Euler_tour_map));
        list.append(state_array_to_python(state.tree_Euler_tour_depth));
        list.append(state_array_to_python(state.first_visit_in_Euler_tour));
        list.append(get_rmq_state_to_python(state.rmq_state));
        return list;
    }

    template<typename lca_t>
    auto get_lca_state_from_python(const py::list &list) {
        using state_type = typename lca_t::internal_state;
        return state_type(
                state_array_from_python<index_t>(list[0]),
                state_array_from_python<index_t>(list[1]),
                state_array_from_python<index_t>(list[2]),
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
