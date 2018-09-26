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
#include "py_common_graph.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/accumulator/tree_accumulator.hpp"

template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;
template<typename T, std::size_t N>
using pytensor = xt::pytensor<T, N, xt::layout_type::row_major>;

namespace py = pybind11;

using graph_t = hg::tree;
using edge_t = graph_t::edge_descriptor;
using vertex_t = graph_t::vertex_descriptor;


template<typename graph_t>
struct def_accumulate_parallel {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("accumulate_parallel", [](const graph_t &tree, const pyarray<value_t> &input,
                                        hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_min());
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_max());
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_mean());
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_sum());
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_prod());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
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
        c.def("accumulate_sequential",
              [](const graph_t &tree, const pyarray<value_t> &vertex_data, hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_min());
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_max());
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_mean());
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_sum());
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_prod());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
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
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_min(),
                                                                       f);
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_max(),
                                                                       f);
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_mean(),
                                                                       f);
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_counter(),
                                                                       f);
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_sum(),
                                                                       f);
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_prod(),
                                                                       f);
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
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
        c.def("propagate_sequential",
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
        c.def("propagate_parallel",
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

void py_init_tree_accumulator(pybind11::module &m) {
    add_type_overloads<def_accumulate_parallel<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "For each node i of the tree, we accumulate values of the children of i in the input array and put the result "
             "in output. i.e. output(i) = accumulate(input(children(i)))");

    add_type_overloads<def_accumulate_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Performs a sequential accumulation of node values from the leaves to the root. "
             "For each leaf node i, output(i) = leaf_data(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = accumulate(output(children(i)))");

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Performs a sequential accumulation of node values from the leaves to the root and add the result with the input array."
             "For each leaf node i, output(i) = leaf_data(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = input(i) + accumulate(output(children(i)))",
             "accumulate_and_add_sequential",
             functorPlus());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Performs a sequential accumulation of node values from the leaves to the root and multiply the result with the input array."
             "For each leaf node i, output(i) = leaf_data(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = input(i) * accumulate(output(children(i)))",
             "accumulate_and_multiply_sequential",
             functorMultiply());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Performs a sequential accumulation of node values from the leaves to the root and max the result with the input array."
             "For each leaf node i, output(i) = leaf_data(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = max(input(i), accumulate(output(children(i))))",
             "accumulate_and_max_sequential",
             functorMax());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Performs a sequential accumulation of node values from the leaves to the root and min the result with the input array."
             "For each leaf node i, output(i) = leaf_data(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = min(input(i), accumulate(output(children(i))))",
             "accumulate_and_min_sequential",
             functorMin());

    add_type_overloads<def_propagate_parallel<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "\
The conditional parallel propagator defines the new value of a node as its parent value if the condition is true and keeps its value otherwise. \
This process is done in parallel on the whole tree. The default condition (if the user does not provide one) is true for all nodes: each node takes \
the value of its parent. \n\n\
The conditional parallel propagator pseudo-code could be::\n\n\
	# input: a tree t\n\
	# input: an attribute att on the nodes of t\n\
	# input: a condition cond on the nodes of t\n\n\
\
	output = copy(input)\n\n\
\
	for each node n of t:\n\
	if(cond(n)):\n\
	    output[n] = input[t.parent(n)]\n\n\
\
	return output\
");

    add_type_overloads<def_propagate_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Conditionally propagates parent values to children. "
             "For each node i from the root to the leaves, if condition(i) then output(i) = output(tree.parent(i)) otherwise"
             "output(i) = input(i)");
}