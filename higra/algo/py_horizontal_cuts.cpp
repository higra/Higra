/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_horizontal_cuts.hpp"
#include "../py_common.hpp"
#include "higra/algo/horizontal_cuts.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

using namespace hg;
namespace py = pybind11;


template<typename c_t, typename tree_t>
struct def_reconstruct_leaf_data {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_reconstruct_leaf_data", [](const c_t &c, const tree_t &tree, const xt::pyarray<type> &data) {
                  return c.reconstruct_leaf_data(tree, data);
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"));
    }
};

template<typename tree_t>
void def_horizontal_cut_nodes(pybind11::module &m) {
    using class_t = hg::horizontal_cut_nodes<double>;
    auto c = py::class_<class_t>(m, "HorizontalCutNodes");
    c.def("nodes",
          [](const class_t &c) -> const array_1d<index_t> & { return c.nodes; },
          "Array containing the indices of the nodes of the cut.");
    c.def("altitude",
          [](const class_t &c) { return c.altitude; },
          "Altitude of the cut.");
    c.def("_labelisation_leaves",
          [](const class_t &c, const tree_t &tree) { return c.labelisation_leaves(tree); },
          "Labelize tree leaves according to the horizontal cut. \n"
          "Two leaves are in the same region (ie. have the same label) if "
          "their lowest common ancestor is a subset or equal to one the node of the cut.",
          py::arg("tree"));
    c.def("_graph_cut",
          [](const class_t &c, const tree_t &tree, const hg::ugraph &leaf_graph) {
              return c.graph_cut(tree, leaf_graph);
          },
          "",
          py::arg("tree"),
          py::arg("leaf_graph"));
    add_type_overloads<def_reconstruct_leaf_data<class_t, tree_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Each leaf of the tree takes the altitude of the node of the cut that contains it."
            );
}

template<typename c_t>
struct def_horizontal_cut_explorer_ctr {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def_static("_make_HorizontalCutExplorer",
                     [](const typename c_t::tree_type &tree, const xt::pyarray<type> &altitudes) {
                         return c_t(tree, altitudes);
                     },
                     doc,
                     py::arg("tree"),
                     py::arg("altitudes"));
    }
};

template<typename tree_t>
void def_horizontal_cut_explorer(pybind11::module &m) {
    using class_t = hg::horizontal_cut_explorer<tree_t, double>;
    auto c = py::class_<class_t>(
            m,
            "HorizontalCutExplorer",
            "This class helps to explore and browse the horizontal cuts of a valued hierarchy.\n"
            "Construction of the HorizontalCutExplorer if performed in linear time O(n) w.r.t. the number of nodes in the tree.\n"
            "Each cut of the hierarchy can be accessed through:\n\n"
            "\t- its index (the first single region cut has index 0). This operations runs in O(k), with k the number of regions in the retrieved cut ;\n"
            "\t- the number of regions in the cut (the smallest partition having at least the given number of regions if found). This operations runs in O(k*log(n)), with k the number of regions in the retrieved cut;\n"
            "\t- the altitude of the cut. This operations runs in O(k*log(n)), with k the number of regions in the retrieved cut.\n"
    );
    add_type_overloads<def_horizontal_cut_explorer_ctr<class_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Create an horizontal cut explorer for the provided valued hierarchy."
            );
    c.def("num_cuts", &class_t::num_cuts, "Number of horizontal cuts in the hierarchy.");
    c.def("num_regions_cut",
          &class_t::num_regions_cut,
          "Number of regions in the i-th cut of the hierarchy (cut numbering start at 0 with the cut with a single region).",
          py::arg("i"));
    c.def("num_regions_cuts",
          &class_t::num_regions_cuts,
          "Number of regions in each cut of the hierarchy.");
    c.def("altitude_cut",
          &class_t::altitude_cut,
          "Altitude of the i-th cut of the hierarchy (cut numbering start at 0 with the cut with a single region).",
          py::arg("i"));
    c.def("altitude_cuts",
          &class_t::altitude_cuts,
          "Altitude of each cut of the hierarchy.");
    c.def("horizontal_cut_from_index",
          &class_t::horizontal_cut_from_index,
          "Retrieve the i-th horizontal cut of tree (cut numbering start at 0 with the cut with a single region).",
          py::arg("i"));
    c.def("horizontal_cut_from_altitude",
          &class_t::horizontal_cut_from_altitude,
          "Retrieve the horizontal cut for given threshold level.",
          py::arg("threshold"));
    c.def("horizontal_cut_from_num_regions",
          &class_t::horizontal_cut_from_num_regions,
          "Retrieve the smallest horizontal cut having at least the given number of regions.",
          py::arg("num_regions"));
}

void py_init_horizontal_cuts(pybind11::module &m) {
    xt::import_numpy();

    def_horizontal_cut_nodes<hg::tree>(m);
    def_horizontal_cut_explorer<hg::tree>(m);
}

