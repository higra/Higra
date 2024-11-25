/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/
#define FORCE_IMPORT_ARRAY
#include "pybind11/pybind11.h"

#include "all.hpp"

#include "xtl/xmeta_utils.hpp"


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
    py_accumulator::py_init_accumulators(m);
    py_graph_core::py_init_algo_graph_core(m);
    py_tree::py_init_algo_tree(m);
    py_alignement::py_init_alignement(m);
    py_partition::py_init_assessment_partition(m);
    py_at_accumulator::py_init_at_accumulator(m);
    py_attributes::py_init_attributes(m);
    py_binary_partition_tree::py_init_binary_partition_tree(m);
    py_bipartite_graph::py_init_bipartite_graph(m);
    py_common_hierarchy::py_init_common_hierarchy(m);
    py_component_tree::py_init_component_tree(m);
    py_contour_2d::py_init_contour_2d(m);
    py_embedding::py_init_embedding(m);
    py_graph_accumulator::py_init_graph_accumulator(m);
    py_graph_image::py_init_graph_image(m);
    py_graph_weights::py_init_graph_weights(m);
    py_fragmentation_curve::py_init_fragmentation_curve(m);
    py_hierarchy_core::py_init_hierarchy_core(m);
    py_hierarchy_mean_pb::py_init_hierarchy_mean_pb(m);
    py_horizontal_cuts::py_init_horizontal_cuts(m);
    py_lca_fast::py_init_lca_fast(m);
    py_log::py_init_log(m);
    py_pink_io::py_init_pink_io(m);
    py_rag::py_init_rag(m);
    py_regular_graph::py_init_regular_graph(m);
    py_scipy::py_init_scipy(m);
    py_sorting::py_init_sorting(m);
    py_tree_accumulator::py_init_tree_accumulator(m);
    py_contour_accumulator::py_init_tree_contour_accumulator(m);
    py_tree_energy_optimization::py_init_tree_energy_optimization(m);
    py_tree_fusion::py_init_tree_fusion(m);
    py_tree_graph::py_init_tree_graph(m);
    py_tree_io::py_init_tree_io(m);
    py_tree_of_shapes::py_init_tree_of_shapes_image(m);
    py_tree_monotonic_regression::py_init_tree_monotonic_regression(m);
    py_undirected_graph::py_init_undirected_graph(m);
    py_watershed::py_init_watershed(m);
    py_watershed_hierarchy::py_init_watershed_hierarchy(m);
}