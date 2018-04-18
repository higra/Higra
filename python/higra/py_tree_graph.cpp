//
// Created by user on 4/1/18.
//

#include "py_tree_graph.hpp"
#include "py_common_graph.hpp"
#include "xtensor-python/pyarray.hpp"
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
    c.def("accumulateParallel", [](const graph_t &tree, const xt::pyarray<value_t> &input, xt::pyarray<value_t> &output,
                                   hg::accumulators accumulator) {
              switch (accumulator) {
                  case hg::accumulators::min:
                      hg::accumulate_parallel(tree, input, output, hg::accumulator_min<value_t>());
                      break;
                  case hg::accumulators::max:
                      hg::accumulate_parallel(tree, input, output, hg::accumulator_max<value_t>());
                      break;
                  case hg::accumulators::mean:
                      hg::accumulate_parallel(tree, input, output, hg::accumulator_mean<value_t>());
                      break;
                  case hg::accumulators::counter:
                      hg::accumulate_parallel(tree, input, output, hg::accumulator_counter());
                      break;
                  case hg::accumulators::sum:
                      hg::accumulate_parallel(tree, input, output, hg::accumulator_sum<value_t>());
                      break;
                  case hg::accumulators::prod:
                      hg::accumulate_parallel(tree, input, output, hg::accumulator_prod<value_t>());
                      break;
              }
          },
          "For each node i of the tree, we accumulate values of the children of i in the input array and put the result "
                  "in output. i.e. output(i) = accumulate(input(children(i)))",
          py::arg("inputArray"),
          py::arg("outputArray"),
          py::arg("accumulator"));
}

template<typename value_t, typename pyc>
void add_accumulate_sequential(pyc &c) {
    c.def("accumulateSequential", [](const graph_t &tree, xt::pyarray<value_t> &output, hg::accumulators accumulator) {
              switch (accumulator) {
                  case hg::accumulators::min:
                      hg::accumulate_sequential(tree, output, hg::accumulator_min<value_t>());
                      break;
                  case hg::accumulators::max:
                      hg::accumulate_sequential(tree, output, hg::accumulator_max<value_t>());
                      break;
                  case hg::accumulators::mean:
                      hg::accumulate_sequential(tree, output, hg::accumulator_mean<value_t>());
                      break;
                  case hg::accumulators::counter:
                      hg::accumulate_sequential(tree, output, hg::accumulator_counter());
                      break;
                  case hg::accumulators::sum:
                      hg::accumulate_sequential(tree, output, hg::accumulator_sum<value_t>());
                      break;
                  case hg::accumulators::prod:
                      hg::accumulate_sequential(tree, output, hg::accumulator_prod<value_t>());
                      break;
              }
          },
          "Performs a sequential accumulation of node values from the leaves to the root. For each node i from the leaves to the root, output(i) = accumulate(output(children(i)))", \
        py::arg("outputArray"), py::arg("accumulator"));
}

template<typename value_t, typename pyc, typename combination_fun_t>
void add_accumulate_and_combine_sequential(pyc &c, combination_fun_t fun, std::string name) {
    c.def(("accumulateAnd" + name + "Sequential").c_str(),
          [fun](const graph_t &tree, const xt::pyarray<value_t> &input, xt::pyarray<value_t> &output,
                hg::accumulators accumulator) {
              switch (accumulator) {
                  case hg::accumulators::min:
                      hg::accumulate_and_combine_sequential(tree, input, output, hg::accumulator_min<value_t>(), fun);
                      break;
                  case hg::accumulators::max:
                      hg::accumulate_and_combine_sequential(tree, input, output, hg::accumulator_max<value_t>(), fun);
                      break;
                  case hg::accumulators::mean:
                      hg::accumulate_and_combine_sequential(tree, input, output, hg::accumulator_mean<value_t>(), fun);
                      break;
                  case hg::accumulators::counter:
                      hg::accumulate_and_combine_sequential(tree, input, output, hg::accumulator_counter(),
                                                            fun);
                      break;
                  case hg::accumulators::sum:
                      hg::accumulate_and_combine_sequential(tree, input, output, hg::accumulator_sum<value_t>(), fun);
                      break;
                  case hg::accumulators::prod:
                      hg::accumulate_and_combine_sequential(tree, input, output, hg::accumulator_prod<value_t>(), fun);
                      break;
              }
          },
          "Performs a sequential accumulation of node values from the leaves to the root and combine result with the input array."
                  "For each node i from the leaves to the root, output(i) = combine(input(i),accumulate(output(children(i))))",
          py::arg("inputArray"),
          py::arg("outputArray"),
          py::arg("accumulator"));
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
}

#undef TREE_CTR
