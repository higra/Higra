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
#include "../py_common.hpp"
#include "higra/algo/alignment.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py_alignement {

    namespace py = pybind11;

    struct def_project_fine_to_coarse_labelisation {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_project_fine_to_coarse_labelisation",
                  [](const hg::array_view<const value_t> &labelisation_fine,
                     const hg::array_view<const value_t> &labelisation_coarse,
                     size_t num_regions_fine,
                     size_t num_regions_coarse) {
                      return hg::project_fine_to_coarse_labelisation(
                              hg::bridge::as_xt(labelisation_fine),
                              hg::bridge::as_xt(labelisation_coarse),
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
                                              const hg::array_view<const value_t> &edge_weights) {
                             return hg::make_hierarchy_aligner_from_graph_cut(g, hg::bridge::as_xt(edge_weights));
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
                                                 const hg::array_view<const value_t> &vertex_labels) {
                             return hg::make_hierarchy_aligner_from_labelisation(g, hg::bridge::as_xt(vertex_labels));
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
                                              const hg::array_view<const value_t> &altitudes) {
                             return hg::make_hierarchy_aligner_from_hierarchy(g, t, hg::bridge::as_xt(altitudes));
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
                          const hg::array_view<const value_t> &altitudes) {
                      return a.align_hierarchy(t, hg::bridge::as_xt(altitudes));
                  },
                  doc,
                  py::arg("tree"),
                  py::arg("altitudes"));
            c.def("align_hierarchy", [](
                          const hg::hierarchy_aligner &a,
                          const hg::ugraph &graph,
                          const hg::array_view<const value_t> &saliency_map) {
                      return a.align_hierarchy(graph, hg::bridge::as_xt(saliency_map));
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("saliency_map"));
            c.def("align_hierarchy", [](
                          const hg::hierarchy_aligner &a,
                          const hg::array_view<const hg::index_t> &super_vertices,
                          const hg::tree &t,
                          const hg::array_view<const value_t> &altitudes) {
                      return a.align_hierarchy(hg::bridge::as_xt(super_vertices), t, hg::bridge::as_xt(altitudes));
                  },
                  doc,
                  py::arg("super_vertices"),
                  py::arg("tree"),
                  py::arg("altitudes"));
        }
    };

    void py_init_alignement(pybind11::module &m) {
        //xt::import_numpy();

        add_type_overloads<def_project_fine_to_coarse_labelisation, HG_TEMPLATE_INTEGRAL_TYPES>
                (m, "");


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
}