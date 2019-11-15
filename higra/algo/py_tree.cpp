/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree.hpp"
#include "../py_common.hpp"
#include "higra/graph.hpp"
#include "higra/algo/tree.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

struct labelisation_horizontal_cut {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_labelisation_horizontal_cut_from_threshold", [](const hg::tree &tree,
                                                 double threshold,
                                                 const pyarray<value_t> &altitudes) {
                  return hg::labelisation_horizontal_cut_from_threshold(tree, altitudes, threshold);
              },
              doc,
              py::arg("tree"),
              py::arg("threshold"),
              py::arg("altitudes"));
    }
};

struct labelisation_hierarchy_supervertices {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_labelisation_hierarchy_supervertices", [](const hg::tree &tree,
                                                          const pyarray<value_t> &altitudes) {
                  return hg::labelisation_hierarchy_supervertices(tree, altitudes);
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"));
    }
};

struct binary_labelisation_from_markers {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_binary_labelisation_from_markers", [](const hg::tree &tree,
                                                      const pyarray<value_t> &object_marker,
                                                      const pyarray<value_t> &background_marker) {
                  return hg::binary_labelisation_from_markers(tree, object_marker, background_marker);
              },
              doc,
              py::arg("tree"),
              py::arg("object_marker"),
              py::arg("background_marker"));
    }
};

struct sort_hierarchy_with_altitudes {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_sort_hierarchy_with_altitudes", [](const hg::tree &tree,
                                                      const pyarray<value_t> &altitudes) {
                  auto res = hg::sort_hierarchy_with_altitudes(tree, altitudes);
                  return pybind11::make_tuple(std::move(res.tree), std::move(res.node_map));
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"));
    }
};

void py_init_algo_tree(pybind11::module &m) {
    xt::import_numpy();

    m.def("test_tree_isomorphism", &hg::test_tree_isomorphism<hg::tree, hg::tree>,
          "Test if 2 trees :math:`t_1` and :math:`t_2` are isomorph assuming that they share the same leaves.\n"
          "\n"
          "By this definition :math:`t_1` is isomorph to :math:`t_2` if there exists a bijection :math:`f` from "
          "the nodes of :math:`t_1` to the nodes of :math:`t_2` such that:\n\n"
          "  1) for any leaf node :math:`n` of :math:`t_1`, :math:`f(n) = n`, and\n"
          "  2) for any node :math:`n` of :math:`t_1`, :math:`f(t_1.parent(n)) = t_2.parent(f(n))`\n"
          "\n"
          "Note that the root  node of :math:`r` a tree :math:`t` is defined by :math:`t.parent(r) = r`, thus 2) becomes \n"
          "for the root node :math:`r_1` of :math:`t_1`, :math:`f(r_1) = t_2.parent(f(r_1))`, i.e. :math:`f(r_1)` is the root node of :math:`t_2`",
          py::arg("tree1"),
          py::arg("tree2"));

    add_type_overloads<labelisation_horizontal_cut, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Labelize tree leaves according to an horizontal cut in the tree. \n"
             "Two leaves are in the same region (ie. have the same label) if "
             "the altitude of their lowest common ancestor is strictly greater "
             "than the specified threshold."
            );
    add_type_overloads<labelisation_hierarchy_supervertices, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Labelize the tree leaves into supervertices.\n"
             "Two leaves are in the same supervertex if they have a common ancestor "
             "of altitude 0.\n"
             "This functions guaranties that the labels are in the range [0, num_supervertices-1]."
            );

    add_type_overloads<binary_labelisation_from_markers, HG_TEMPLATE_INTEGRAL_TYPES>
            (m,
             "Union of tree regions with a non empty intersection with the object marker and an empty "
             "intersection with the background marker."
            );

    add_type_overloads<sort_hierarchy_with_altitudes, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
}