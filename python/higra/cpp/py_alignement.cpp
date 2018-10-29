/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_alignement.hpp"
#include "py_common.hpp"
#include "higra/algo/alignment.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

struct def_project_fine_to_coarse_labelisation {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("project_fine_to_coarse_labelisation",
              [](const pyarray<value_t> &labelisation_fine,
                 const pyarray<value_t> &labelisation_coarse,
                 size_t num_regions_fine,
                 size_t num_regions_coarse) {
                  return hg::project_fine_to_coarse_labelisation(
                          labelisation_fine,
                          labelisation_coarse,
                          num_regions_fine,
                          num_regions_coarse);
              },
              doc,
              py::arg("labelisation_fine"),
              py::arg("labelisation_coarse"),
              py::arg("num_regions_fine") = 0,
              py::arg("num_regions_coarse") = 0);
    }
};

struct def_from_cut {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def_static("from_graph_cut", [](const hg::ugraph &g,
                                          const pyarray<value_t> &edge_weights) {
                         return hg::make_hierarchy_aligner_from_graph_cut(g, edge_weights);
                     },
                     doc,
                     py::arg("graph"),
                     py::arg("edge_weights"));
    }
};

struct def_from_labelisation {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def_static("from_labelisation", [](const hg::ugraph &g,
                                             const pyarray<value_t> &vertex_labels) {
                         return hg::make_hierarchy_aligner_from_labelisation(g, vertex_labels);
                     },
                     doc,
                     py::arg("graph"),
                     py::arg("vertex_labels"));
    }
};

struct def_from_hierarchy {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def_static("from_hierarchy", [](const hg::ugraph &g,
                                          const hg::tree &t,
                                          const pyarray<value_t> &altitudes) {
                         return hg::make_hierarchy_aligner_from_hierarchy(g, t, altitudes);
                     },
                     doc,
                     py::arg("graph"),
                     py::arg("tree"),
                     py::arg("altitudes"));
    }
};

struct def_align_hierarchy {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("align_hierarchy", [](
                      const hg::hierarchy_aligner &a,
                      const hg::tree &t,
                      const pyarray<value_t> &altitudes) {
                  return a.align_hierarchy(t, altitudes);
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"));
        c.def("align_hierarchy", [](
                      const hg::hierarchy_aligner &a,
                      const hg::ugraph &graph,
                      const pyarray<value_t> &saliency_map) {
                  return a.align_hierarchy(graph, saliency_map);
              },
              doc,
              py::arg("graph"),
              py::arg("saliency_map"));
        c.def("align_hierarchy", [](
                      const hg::hierarchy_aligner &a,
                      const pyarray<hg::index_t> &super_vertices,
                      const hg::tree &t,
                      const pyarray<value_t> &altitudes) {
                  return a.align_hierarchy(super_vertices, t, altitudes);
              },
              doc,
              py::arg("super_vertices"),
              py::arg("tree"),
              py::arg("altitudes"));
    }
};

void py_init_alignement(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_project_fine_to_coarse_labelisation, HG_TEMPLATE_INTEGRAL_TYPES>
            (m,
             "Given two labelisations, a fine and a coarse one, of a same set of elements.\n"
             "Find for each label (ie. region) of the fine labelisation, the label of the region in the\n"
             "coarse labelisation that maximises the intersection with the \"fine\" region.\n"
             "\n"
             "Pre-condition:\n"
             "\trange(labelisation_fine) = [0..num_regions_fine[\n"
             "\trange(labelisation_coarse) = [0..num_regions_coarse[\n"
             "\n"
             "If num_regions_fine or num_regions_coarse are not provided, they will\n"
             "be determined as max(labelisation_fine) + 1 and max(labelisation_coarse) + 1");


    auto c = py::class_<hg::hierarchy_aligner>(m, "HierarchyAligner");

    add_type_overloads<def_from_cut, HG_TEMPLATE_NUMERIC_TYPES>
            (c, "Create a hierarchy aligner based on a graph cut.");

    add_type_overloads<def_from_labelisation, HG_TEMPLATE_INTEGRAL_TYPES>
            (c, "Create a hierarchy aligner based on a labelisation of graph vertices.");

    add_type_overloads<def_from_hierarchy, HG_TEMPLATE_NUMERIC_TYPES>
            (c, "Create a hierarchy aligner based on the supervertices of a hierarchy.");

    add_type_overloads<def_align_hierarchy, HG_TEMPLATE_NUMERIC_TYPES>
            (c, "Align the given hierarchy given either as a tree or as a saliency map.");

}