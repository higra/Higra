/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_accumulators.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "common.hpp"

// @TODO Remove layout_type when xtensor solves the issue with iterators
template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;

namespace py = pybind11;

using graph_t = hg::tree;
using edge_t = graph_t::edge_descriptor;
using vertex_t = graph_t::vertex_descriptor;


template<typename graph_t>
struct def_accumulate_parallel {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_accumulate_parallel", [](const graph_t &tree, const pyarray<value_t> &input,
                                         hg::accumulators accumulator) {
                  return dispatch_accumulator(
                          [&tree, &input](const auto &acc) {
                              return hg::accumulate_parallel(tree, input, acc);
                          },
                          accumulator);
              },
              doc,
              py::arg("tree"),
              py::arg("input"),
              py::arg("accumulator"));
    }
};

template<typename graph_t>
struct def_accumulate_sequential {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_accumulate_sequential",
              [](const graph_t &tree, const pyarray<value_t> &vertex_data, hg::accumulators accumulator) {
                  return dispatch_accumulator(
                          [&tree, &vertex_data](const auto &acc) {
                              return hg::accumulate_sequential(tree, vertex_data, acc);
                          },
                          accumulator);
              },
              doc,
              py::arg("tree"),
              py::arg("leaf_data"),
              py::arg("accumulator"));
    }
};


struct functorMax {
    template<typename T1, typename T2>
    auto operator()(T1 &&a, T2 &&b) {
        return std::max(std::forward<T1>(a), std::forward<T2>(b));
    }
};

struct functorMin {
    template<typename T1, typename T2>
    auto operator()(T1 &&a, T2 &&b) {
        return std::min(std::forward<T1>(a), std::forward<T2>(b));
    }
};

struct functorPlus {
    template<typename T1, typename T2>
    auto operator()(T1 &&a, T2 &&b) {
        return a + b;
    }
};

struct functorMultiply {
    template<typename T1, typename T2>
    auto operator()(T1 &&a, T2 &&b) {
        return a * b;
    }
};


template<typename graph_t>
struct def_accumulate_and_combine_sequential {
    template<typename value_t, typename C, typename F>
    static
    void def(C &c, const char *doc, const char *name, const F &f) {
        c.def(name,
              [&f](const graph_t &tree, const pyarray<value_t> &input, const pyarray<value_t> &vertex_data,
                   hg::accumulators accumulator) {
                  return dispatch_accumulator(
                          [&tree, &input, &vertex_data, &f](const auto &acc) {
                              return hg::accumulate_and_combine_sequential(tree, input, vertex_data, acc, f);
                          },
                          accumulator);
              },
              doc,
              py::arg("tree"),
              py::arg("input"),
              py::arg("leaf_data"),
              py::arg("accumulator"));
    }
};


template<typename graph_t>
struct def_propagate_sequential {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_propagate_sequential",
              [](const graph_t &tree, const pyarray<value_t> &input,
                 const pyarray<bool> &condition) {
                  return hg::propagate_sequential(tree, input, condition);
              },
              doc,
              py::arg("tree"),
              py::arg("input"),
              py::arg("condition"));
    }
};

template<typename graph_t>
struct def_propagate_parallel {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_propagate_parallel",
              [](const graph_t &tree, const pyarray<value_t> &input,
                 const pyarray<bool> &condition) {
                  if (condition.dimension() == 0) {
                      return hg::propagate_parallel(tree, input);
                  } else {
                      return hg::propagate_parallel(tree, input, condition);
                  }
              },
              doc,
              py::arg("tree"),
              py::arg("input"),
              py::arg("condition") = pyarray<bool>{});
    }
};


template<typename graph_t>
struct def_propagate_sequential_and_accumulate {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_propagate_sequential_and_accumulate",
              [](const graph_t &tree, const pyarray<value_t> &vertex_data, hg::accumulators accumulator) {
                  return dispatch_accumulator(
                          [&tree, &vertex_data](const auto &acc) {
                              return hg::propagate_sequential_and_accumulate(tree, vertex_data, acc);
                          },
                          accumulator);
              },
              doc,
              py::arg("tree"),
              py::arg("vertex_data"),
              py::arg("accumulator"));
    }
};

void py_init_tree_accumulator(pybind11::module &m) {
    xt::import_numpy();
    add_type_overloads<def_accumulate_parallel<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "");

    add_type_overloads<def_accumulate_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "");

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "",
             "_accumulate_and_add_sequential",
             functorPlus());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "",
             "_accumulate_and_multiply_sequential",
             functorMultiply());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "",
             "_accumulate_and_max_sequential",
             functorMax());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "",
             "_accumulate_and_min_sequential",
             functorMin());

    add_type_overloads<def_propagate_parallel<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m, "");

    add_type_overloads<def_propagate_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m, "");

    add_type_overloads<def_propagate_sequential_and_accumulate<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m, "");
}