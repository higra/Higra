/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "pybind11/pybind11.h"

#include "all.hpp"

#include "xtl/xmeta_utils.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

PYBIND11_MODULE(higram, m) {
    m.doc() = R"pbdoc(
        Higra: Hierarchical Graph Analysis
        -----------------------
        .. currentmodule:: higra
        .. autosummary::
           :toctree: _generate
    )pbdoc";


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
    xt::import_numpy();
    py_init_accumulators(m);
    py_init_algo_graph_core(m);
    py_init_algo_tree(m);
    py_init_alignement(m);
    py_init_assessment_partition(m);
    py_init_at_accumulator(m);
    py_init_attributes(m);
    py_init_binary_partition_tree(m);
    py_init_common_hierarchy(m);
    py_init_component_tree(m);
    py_init_contour_2d(m);
    py_init_embedding(m);
    py_init_graph_accumulator(m);
    py_init_graph_image(m);
    py_init_graph_weights(m);
    py_init_fragmentation_curve(m);
    py_init_hierarchy_core(m);
    py_init_hierarchy_mean_pb(m);
    py_init_horizontal_cuts(m);
    py_init_lca_fast(m);
    py_init_log(m);
    py_init_pink_io(m);
    py_init_rag(m);
    py_init_regular_graph(m);
    py_init_scipy(m);
    py_init_tree_accumulator(m);
    py_init_tree_contour_accumulator(m);
    py_init_tree_energy_optimization(m);
    py_init_tree_fusion(m);
    py_init_tree_graph(m);
    py_init_tree_io(m);
    py_init_tree_of_shapes_image(m);
    py_init_undirected_graph(m);
    py_init_watershed(m);
    py_init_watershed_hierarchy(m);
}