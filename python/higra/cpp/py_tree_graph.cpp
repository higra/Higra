//
// Created by user on 4/1/18.
//

#include "py_tree_graph.hpp"
#include "py_common_graph.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/accumulator/tree_accumulator.hpp"

namespace py = pybind11;

using graph_t = hg::tree;
using edge_t = typename boost::graph_traits<graph_t>::edge_descriptor;
using vertex_t = typename boost::graph_traits<graph_t>::vertex_descriptor;

template<typename graph_t>
struct def_tree_ctr {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def(py::init([](const xt::pyarray<type> &parent) { return graph_t(parent); }),
              doc,
              py::arg("parentRelation")
        );
    }
};

template<typename graph_t>
struct def_accumulate_parallel {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("accumulateParallel", [](const graph_t &tree, const xt::pyarray<value_t> &input,
                                       hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_min<value_t>());
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_max<value_t>());
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_mean<value_t>());
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_sum<value_t>());
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_parallel(tree, input, hg::accumulator_prod<value_t>());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
              },
              doc,
              py::arg("inputArray"),
              py::arg("accumulator"));
    }
};

template<typename graph_t>
struct def_accumulate_sequential {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("accumulateSequential",
              [](const graph_t &tree, const xt::pyarray<value_t> &vertex_data, hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_min<value_t>());
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_max<value_t>());
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_mean<value_t>());
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_sum<value_t>());
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_sequential(tree, vertex_data, hg::accumulator_prod<value_t>());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
              },
              doc,
              py::arg("leafDataArray"),
              py::arg("accumulator"));
    }
};


struct functorMax {
    template<typename T1, typename T2>
    auto operator()(T1 &&a, T2 &&b) {
        return xt::maximum(std::forward<T1>(a), std::forward<T2>(b));
    }
};

struct functorMin {
    template<typename T1, typename T2>
    auto operator()(T1 &&a, T2 &&b) {
        return xt::minimum(std::forward<T1>(a), std::forward<T2>(b));
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
              [&f](const graph_t &tree, const xt::pyarray<value_t> &input, const xt::pyarray<value_t> &vertex_data,
                 hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_min<value_t>(),
                                                                       f);
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_max<value_t>(),
                                                                       f);
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_mean<value_t>(),
                                                                       f);
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_counter(),
                                                                       f);
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_sum<value_t>(),
                                                                       f);
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                       hg::accumulator_prod<value_t>(),
                                                                       f);
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
              },
              doc,
              py::arg("inputArray"),
              py::arg("leafDataArray"),
              py::arg("accumulator"));
    }
};





template<typename graph_t>
struct def_propagate_sequential {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("propagateSequential",
              [](const graph_t &tree, const xt::pyarray<value_t> &input,
                 const xt::pyarray<bool> &condition) {
                  return hg::propagate_sequential(tree, input, condition);
              },
              doc,
              py::arg("inputArray"),
              py::arg("condition"));
    }
};

template<typename graph_t>
struct def_propagate_parallel {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("propagateParallel",
              [](const graph_t &tree, const xt::pyarray<value_t> &input,
                 const xt::pyarray<bool> &condition) {
                  return hg::propagate_parallel(tree, input, condition);
              },
              doc,
              py::arg("inputArray"),
              py::arg("condition"));
    }
};

void py_init_tree_graph(pybind11::module &m) {
    xt::import_numpy();
    auto c = py::class_<graph_t>(m, "Tree");

    add_type_overloads<def_tree_ctr<graph_t>, HG_TEMPLATE_INTEGRAL_TYPES>
            (c, "Create a tree from the given parent relation.");

    add_incidence_graph_concept<graph_t, decltype(c)>(c);
    add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
    add_adjacency_graph_concept<graph_t, decltype(c)>(c);
    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_index_graph_concept<graph_t, decltype(c)>(c);

    c.def("root", &graph_t::root, "Get the index of the root node (i.e. self.numVertices() - 1)");
    c.def("numLeaves", &graph_t::num_leaves, "Get the number of leaves nodes.");
    c.def("numChildren", &graph_t::num_children, "Get the number of children nodes of the given node.",
          py::arg("node"));
    c.def("children",
          [](const graph_t &g, vertex_t v) {
              auto it = hg::children(v, g);
              return pybind11::make_iterator(it.first, it.second);
          },
          "Get an iterator on the children of the given node.",
          py::arg("node"));
    c.def("parents", &graph_t::parents, "Get the parents array representing the tree.");
    c.def("parent", [](const graph_t &tree, vertex_t v) { return tree.parent(v); }, "Get the parent of the given node.",
          py::arg("node"));

    c.def("iterateFromLeavesToRoot",
          [](const graph_t &tree, bool includeLeaves, bool includeRoot) {
              auto range = tree.iterate_from_leaves_to_root(
                      (includeLeaves) ? hg::leaves_it::include : hg::leaves_it::exclude,
                      (includeRoot) ? hg::root_it::include : hg::root_it::exclude);
              return py::make_iterator(range.begin(), range.end());
          }, "Returns an iterator on the node indices going from the leaves to the root of the tree.",
          py::arg("includeLeaves") = true,
          py::arg("includeRoot") = true);

    c.def("iterateFromRootToLeaves",
          [](const graph_t &tree, bool includeLeaves, bool includeRoot) {
              auto range = tree.iterate_from_root_to_leaves(
                      (includeLeaves) ? hg::leaves_it::include : hg::leaves_it::exclude,
                      (includeRoot) ? hg::root_it::include : hg::root_it::exclude);
              return py::make_iterator(range.begin(), range.end());
          }, "Returns an iterator on the node indices going from the root to the leaves of the tree.",
          py::arg("includeLeaves") = true,
          py::arg("includeRoot") = true);


    add_type_overloads<def_accumulate_parallel<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "For each node i of the tree, we accumulate values of the children of i in the input array and put the result "
                     "in output. i.e. output(i) = accumulate(input(children(i)))");

    add_type_overloads<def_accumulate_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Performs a sequential accumulation of node values from the leaves to the root. "
                     "For each leaf node i, output(i) = leafData(i)."
                     "For each node i from the leaves (excluded) to the root, output(i) = accumulate(output(children(i)))");

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Performs a sequential accumulation of node values from the leaves to the root and add the result with the input array."
                     "For each leaf node i, output(i) = leafData(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = input(i) + accumulate(output(children(i)))",
             "accumulateAndAddSequential",
             functorPlus());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Performs a sequential accumulation of node values from the leaves to the root and multiply the result with the input array."
             "For each leaf node i, output(i) = leafData(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = input(i) * accumulate(output(children(i)))",
             "accumulateAndMultiplySequential",
             functorMultiply());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Performs a sequential accumulation of node values from the leaves to the root and max the result with the input array."
             "For each leaf node i, output(i) = leafData(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = max(input(i), accumulate(output(children(i))))",
             "accumulateAndMaxSequential",
             functorMax());

    add_type_overloads<def_accumulate_and_combine_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Performs a sequential accumulation of node values from the leaves to the root and min the result with the input array."
             "For each leaf node i, output(i) = leafData(i)."
             "For each node i from the leaves (excluded) to the root, output(i) = min(input(i), accumulate(output(children(i))))",
             "accumulateAndMinSequential",
             functorMin());

    add_type_overloads<def_propagate_parallel<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Conditionally propagates parent values to children. "
                     "For each node i, if condition(i) then output(i) = input(tree.parent(i)) otherwise"
                     "output(i) = input(i)");

    add_type_overloads<def_propagate_sequential<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Conditionally propagates parent values to children. "
                     "For each node i from the root to the leaves, if condition(i) then output(i) = output(tree.parent(i)) otherwise"
                     "output(i) = input(i)");

}

