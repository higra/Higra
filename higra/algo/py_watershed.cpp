/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_watershed.hpp"
#include "higra/algo/watershed.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include <optional>
#include <pybind11/stl.h>

namespace py_watershed {

    template<typename T>
    using pyarray = xt::pyarray<T>;

    namespace py = pybind11;


    template<typename graph_t>
    struct def_labelisation_watershed {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_labelisation_watershed", [](const graph_t &graph, const pyarray<value_t> &edge_weights) {
                      return hg::labelisation_watershed(graph, edge_weights);
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("edge_weights"));
        }
    };

    template<typename graph_t>
    struct def_labelisation_seeded_watershed {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_labelisation_seeded_watershed",
                  [](const graph_t &graph,
                     const pyarray<value_t> &edge_weights,
                     const pyarray<hg::index_t> &vertex_seeds,
                     const hg::index_t background_label) {
                      return hg::labelisation_seeded_watershed(graph, edge_weights, vertex_seeds, background_label);
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("edge_weights"),
                  py::arg("vertex_seeds"),
                  py::arg("background_label"));
        }
    };


    void py_init_watershed(pybind11::module &m) {
        //xt::import_numpy();

        add_type_overloads<def_labelisation_watershed<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_labelisation_seeded_watershed<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");

        auto cls = py::class_<hg::incremental_watershed_cut>(m, "IncrementalWatershedCut",
                "Incremental seeded watershed cut based on the binary partition tree.\n\n"
                "Provides efficient computation of seeded watershed cuts in an interactive "
                "segmentation setting, where seeds are added and removed incrementally.");

        #ifdef HG_USE_TBB
        m.attr("_incremental_watershed_tbb_enabled") = true;
        cls.def(
            py::init([](
                const hg::tree &bpt,
                const hg::ugraph &mst,
                std::optional<hg::index_t> MIN_BREADTH,
                std::optional<hg::index_t> PAR_DEPTH,
                std::optional<hg::index_t> number_processors) {
                    return hg::incremental_watershed_cut(
                        bpt,
                        mst,
                        MIN_BREADTH.value_or(165),
                        PAR_DEPTH.value_or(3),
                        number_processors.value_or(
                            oneapi::tbb::info::default_concurrency()));
                }),
            py::arg("bpt"),
            py::arg("mst"),
            py::arg("MIN_BREADTH") = std::nullopt,
            py::arg("PAR_DEPTH") = std::nullopt,
            py::arg("number_processors") = std::nullopt,
            "Create an incremental watershed cut object from a BPT and its MST."
        );
        #else
        m.attr("_incremental_watershed_tbb_enabled") = false;
        cls.def(py::init<const hg::tree &, const hg::ugraph &>(),
            py::arg("bpt"),
            py::arg("mst"),
            "Create an incremental watershed cut object from a binary partition tree and its MST.");
        #endif

        cls.def("_add_seeds",
                [](hg::incremental_watershed_cut &self,
                   const pyarray<hg::index_t> &seed_vertices,
                   const pyarray<hg::index_t> &seed_labels) {
                    self.add_seeds(seed_vertices, seed_labels);
                },
                "Add seeds with given labels.",
                py::arg("seed_vertices"),
                py::arg("seed_labels"));

        cls.def("_remove_seeds",
                [](hg::incremental_watershed_cut &self,
                   const pyarray<hg::index_t> &seed_vertices) {
                    self.remove_seeds(seed_vertices);
                },
                "Remove seeds.",
                py::arg("seed_vertices"));

        cls.def("_get_labeling",
                &hg::incremental_watershed_cut::get_labeling,
                "Compute and return the current vertex labeling.");
    }
}


