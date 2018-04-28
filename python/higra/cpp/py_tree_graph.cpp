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


#define TREE_CTR(rawXKCD, dataXKCD, type) \
    c.def(py::init([](const xt::pyarray<type> & parent){return graph_t(parent);}), \
        "Create a tree from the given parent relation of type  " HG_XSTR(type) ".",    \
        py::arg("parentRelation")   \
        );

//HG_ACCUMULATORS (min)(max)(mean)(counter)(sum)(prod)

template<typename value_t, typename pyc>
void add_accumulate_parallel(pyc &c) {
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
          "For each node i of the tree, we accumulate values of the children of i in the input array and put the result "
                  "in output. i.e. output(i) = accumulate(input(children(i)))",
          py::arg("inputArray"),
          py::arg("accumulator"));
}

template<typename value_t, typename pyc>
void add_accumulate_sequential(pyc &c) {
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
          "Performs a sequential accumulation of node values from the leaves to the root. "
                  "For each leaf node i, output(i) = leafData(i)."
                  "For each node i from the leaves (excluded) to the root, output(i) = accumulate(output(children(i)))",
          py::arg("leafDataArray"),
          py::arg("accumulator"));
}

template<typename value_t, typename pyc, typename combination_fun_t>
void add_accumulate_and_combine_sequential(pyc &c, combination_fun_t fun, std::string name) {
    c.def(("accumulateAnd" + name + "Sequential").c_str(),
          [fun](const graph_t &tree, const xt::pyarray<value_t> &input, const xt::pyarray<value_t> &vertex_data,
                hg::accumulators accumulator) {
              switch (accumulator) {
                  case hg::accumulators::min:
                      return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                   hg::accumulator_min<value_t>(), fun);
                      break;
                  case hg::accumulators::max:
                      return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                   hg::accumulator_max<value_t>(), fun);
                      break;
                  case hg::accumulators::mean:
                      return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                   hg::accumulator_mean<value_t>(), fun);
                      break;
                  case hg::accumulators::counter:
                      return hg::accumulate_and_combine_sequential(tree, input, vertex_data, hg::accumulator_counter(),
                                                                   fun);
                      break;
                  case hg::accumulators::sum:
                      return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                   hg::accumulator_sum<value_t>(), fun);
                      break;
                  case hg::accumulators::prod:
                      return hg::accumulate_and_combine_sequential(tree, input, vertex_data,
                                                                   hg::accumulator_prod<value_t>(), fun);
                      break;
              }
              throw std::runtime_error("Unknown accumulator.");
          },
          "Performs a sequential accumulation of node values from the leaves to the root and combine result with the input array."
                  "For each leaf node i, output(i) = leafData(i)."
                  "For each node i from the leaves (excluded) to the root, output(i) = combine(input(i),accumulate(output(children(i))))",
          py::arg("inputArray"),
          py::arg("leafDataArray"),
          py::arg("accumulator"));
}

template<typename value_t, typename pyc>
void def_propagate_sequential(pyc &c) {
    c.def("propagateSequential",
          [](const graph_t &tree, const xt::pyarray<value_t> &input,
             const xt::pyarray<bool> &condition) {
              return hg::propagate_sequential(tree, input, condition);
          },
          "Conditionally propagates parent values to children. "
                  "For each node i from the root to the leaves, if condition(i) then output(i) = output(tree.parent(i)) otherwise"
                  "output(i) = input(i)",
          py::arg("inputArray"),
          py::arg("condition"));
}

template<typename value_t, typename pyc>
void def_propagate_parallel(pyc &c) {
    c.def("propagateParallel",
          [](const graph_t &tree, const xt::pyarray<value_t> &input,
             const xt::pyarray<bool> &condition) {
              return hg::propagate_parallel(tree, input, condition);
          },
          "Conditionally propagates parent values to children. "
                  "For each node i, if condition(i) then output(i) = input(tree.parent(i)) otherwise"
                  "output(i) = input(i)",
          py::arg("inputArray"),
          py::arg("condition"));
}

void py_init_tree_graph(pybind11::module &m) {
    xt::import_numpy();
    auto c = py::class_<graph_t>(m, "Tree");

    HG_FOREACH(TREE_CTR, HG_INTEGRAL_TYPES)

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
              vertex_t start = (includeLeaves) ? 0 : tree.num_leaves();
              vertex_t end = (includeRoot) ? tree.num_vertices() : tree.num_vertices() - 1;
              return py::slice(start, end, 1);
          }, "Returns an iterator on the node indices going from the leaves to the root of the tree.",
          py::arg("includeLeaves"),
          py::arg("includeRoot"));

    c.def("iterateFromRootToLeaves",
          [](const graph_t &tree, bool includeLeaves, bool includeRoot) {
              vertex_t end = (includeLeaves) ? -1 : tree.num_leaves() - 1;
              vertex_t start = (includeRoot) ? tree.num_vertices() - 1 : tree.num_vertices() - 2;
              return py::slice(start, end, -2);
          }, "Returns an iterator on the node indices going from the root to the leaves of the tree.",
          py::arg("includeLeaves"),
          py::arg("includeRoot"));


#define DEF(rawXKCD, dataXKCD, TYPE) \
        add_accumulate_parallel<TYPE, decltype(c)>(c);
    HG_FOREACH(DEF, HG_NUMERIC_TYPES)
#undef DEF

#define DEF(rawXKCD, dataXKCD, TYPE) \
        add_accumulate_sequential<TYPE, decltype(c)>(c);
    HG_FOREACH(DEF, HG_NUMERIC_TYPES)
#undef DEF

#define DEF(rawXKCD, dataXKCD, TYPE) \
        add_accumulate_and_combine_sequential<TYPE, decltype(c)>(c, std::plus<xt::xarray<TYPE>>(), std::string("Add"));
    HG_FOREACH(DEF, HG_NUMERIC_TYPES)
#undef DEF

#define DEF(rawXKCD, dataXKCD, TYPE) \
        def_propagate_sequential<TYPE, decltype(c)>(c);
    HG_FOREACH(DEF, HG_NUMERIC_TYPES)
#undef DEF

#define DEF(rawXKCD, dataXKCD, TYPE) \
        def_propagate_parallel<TYPE, decltype(c)>(c);
    HG_FOREACH(DEF, HG_NUMERIC_TYPES)
#undef DEF
}

#undef TREE_CTR
