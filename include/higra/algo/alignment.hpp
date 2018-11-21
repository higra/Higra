/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "rag.hpp"
#include "tree.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "xtensor/xsort.hpp"

namespace hg {

    /**
   * Given two labelisations, a fine and a coarse one, of a same set of elements.
   * Find for each label (ie. region) of the fine labelisation, the label of the region in the
   * coarse labelisation that maximises the intersection with the "fine" region.
   *
   * Pre-condition:
   *  range(xlabelisation_fine) = [0..num_regions_fine[
   *  range(xlabelisation_coarse) = [0..num_regions_coarse[
   *
   * If num_regions_fine or num_regions_coarse are not provided, they will
   * be determined as max(xlabelisation_fine) + 1 and max(xlabelisation_coarse) + 1
   * @tparam T1
   * @tparam T2
   * @param xlabelisation_fine
   * @param num_regions_fine
   * @param xlabelisation_coarse
   * @param num_regions_coarse
   * @return a 1d array of size num_regions_fine
   */
    template<typename T1, typename T2>
    auto project_fine_to_coarse_labelisation
            (const xt::xexpression<T1> &xlabelisation_fine,
             const xt::xexpression<T2> &xlabelisation_coarse,
             size_t num_regions_fine = 0,
             size_t num_regions_coarse = 0) {
        HG_TRACE();
        auto &labelisation_fine = xlabelisation_fine.derived_cast();
        auto &labelisation_coarse = xlabelisation_coarse.derived_cast();

        hg_assert_integral_value_type(labelisation_fine);
        hg_assert_integral_value_type(labelisation_coarse);
        hg_assert_1d_array(labelisation_fine);
        hg_assert_1d_array(labelisation_coarse);
        hg_assert(labelisation_fine.size() == labelisation_coarse.size(),
                  "Labelisations must have the same size.");

        if (num_regions_fine == 0) {
            num_regions_fine = xt::amax(labelisation_fine)(0) + 1;
        }

        if (num_regions_coarse == 0) {
            num_regions_coarse = xt::amax(labelisation_coarse)(0) + 1;
        }

        array_2d<size_t> intersections = xt::zeros<size_t>({num_regions_fine, num_regions_coarse});

        for (index_t i = 0; i < (index_t)labelisation_fine.size(); i++) {
            intersections(labelisation_fine(i), labelisation_coarse(i))++;
        }

        // cast size_t -> index_t
        array_1d<index_t> res = xt::argmax(intersections, 1);
        return res;
    }

    /**
     * Given two region adjacency graphs, a fine and a coarse one, of a same set of elements.
     * Find for each region of the fine rag, the region of the
     * coarse rag that maximises the intersection with the "fine" region.
     *
     * @param fine_rag
     * @param coarse_rag
     * @return a 1d array of size num_vertices(fine_rag.rag)
     */
    inline
    auto project_fine_to_coarse_rag
            (const region_adjacency_graph &fine_rag,
             const region_adjacency_graph &coarse_rag) {
        return project_fine_to_coarse_labelisation(fine_rag.vertex_map,
                                                   coarse_rag.vertex_map,
                                                   num_vertices(fine_rag.rag),
                                                   num_vertices(coarse_rag.rag));
    }

    namespace alignment_internal {

        template<typename rag_t, typename T, typename tree_t, typename T2>
        auto project_hierarchy(const rag_t &rag_fine, const T &coarse_supervertices, const tree_t &tree_coarse,
                               const T2 &tree_coarse_node_altitudes) {
            HG_TRACE();
            hg_assert_node_weights(tree_coarse, tree_coarse_node_altitudes);
            hg_assert_1d_array(tree_coarse_node_altitudes);
            hg_assert_1d_array(coarse_supervertices);
            hg_assert(rag_fine.vertex_map.size() == coarse_supervertices.size(),
                      "Dimensions of the two labelisations do not match.");

            auto &fine_supervertices = rag_fine.vertex_map;
            auto &rag = rag_fine.rag;

            auto fine_to_coarse_map = project_fine_to_coarse_labelisation(fine_supervertices, coarse_supervertices);
            array_1d<typename T2::value_type> coarse_sm_on_fine_rag = xt::empty<typename T2::value_type>(
                    {num_edges(rag_fine.rag)});

            lca_fast lca(tree_coarse);

            for (auto e: edge_iterator(rag)) {
                auto projected_lca = lca.lca(fine_to_coarse_map[source(e, rag)], fine_to_coarse_map[target(e, rag)]);
                coarse_sm_on_fine_rag(index(e, rag)) = tree_coarse_node_altitudes(projected_lca);
            }

            return coarse_sm_on_fine_rag;
        }
    }

    /**
     * This class allows to project hierarchies build from coarse supervertices
     * onto fine supervertices.
     *
     * The class is contructed by providing a fine supervertices decomposition of a graph.
     * Then the functions align_hierarchy allows to project a hierarchy, given as a tree or as a saliency map,
     * onto the fine supervertices.
     *
     * Given:
     *  - a graph g
     *  - a fine labelisation l1 of the vertices of g;
     *  - a tree t on g whose supervertices corresponds to the coarse labelisation l2 of the vertices of g; and
     *  - the altitudes a of the nodes of t.
     * Let us denote:
     *  - given a vertex x of g and a labelisation l, l(x) is the region of l that contains x
     *  - given a region r of l1, s(r, l2) is the region R of l2 that has the largest intersection with r:
     *            s(r, l2) = arg_max(R in l2) |R \cap r|
     * The projection of t onto l1 is a hierarchy given by the saliency map sm on g defined by:
     *     for all {x,y} in edges(g), sm({x,y}) = a(lca_t(s(l1(x), l2), s(l1(y), l2)))
     *
     * See the following helper functions for instanciation
     *  - make_hierarchy_aligner_from_graph_cut
     *  - make_hierarchy_aligner_from_labelisation
     *  - make_hierarchy_aligner_from_hierarchy
     */
    class hierarchy_aligner {
    public:

        hierarchy_aligner(region_adjacency_graph &&rag) : m_fine_rag(std::forward<region_adjacency_graph>(rag)) {

        }

        template<typename T>
        auto align_hierarchy(const hg::tree &tree, const xt::xexpression<T> &xaltitudes) const {
            HG_TRACE();
            auto &altitudes = xaltitudes.derived_cast();
            hg_assert_node_weights(tree, altitudes);
            hg_assert_1d_array(altitudes);
            hg_assert(num_leaves(tree) == m_fine_rag.vertex_map.size(),
                      "Cannot align given hierarchy: incompatible sizes!");
            auto sv_hierarchy = supervertices_hierarchy(tree, altitudes);
            auto altitudes_sv_hierarchy = xt::index_view(altitudes, sv_hierarchy.node_map);
            auto coarse_sm_on_fine_rag =
                    alignment_internal::project_hierarchy(m_fine_rag,
                                                           sv_hierarchy.supervertex_labelisation,
                                                           sv_hierarchy.tree,
                                                           altitudes_sv_hierarchy);
            return rag_back_project_weights(m_fine_rag.edge_map, coarse_sm_on_fine_rag);
        }

        template<typename graph_t, typename T>
        auto align_hierarchy(const graph_t &graph, const xt::xexpression<T> &xsaliency_map) const {
            HG_TRACE();
            auto &saliency_map = xsaliency_map.derived_cast();
            hg_assert_edge_weights(graph, saliency_map);
            hg_assert_1d_array(saliency_map);
            hg_assert(num_vertices(graph) == m_fine_rag.vertex_map.size(),
                      "Cannot align given hierarchy: incompatible sizes!");
            auto coarse_rag = make_region_adjacency_graph_from_graph_cut(graph, saliency_map);
            auto coarse_rag_edge_weights = rag_accumulate(coarse_rag.edge_map, saliency_map, accumulator_first());
            auto bpt_coarse_rag = bpt_canonical(coarse_rag.rag, coarse_rag_edge_weights);

            auto coarse_sm_on_fine_rag =
                    alignment_internal::project_hierarchy(m_fine_rag,
                                                           coarse_rag.vertex_map,
                                                           bpt_coarse_rag.tree,
                                                           bpt_coarse_rag.altitudes);

            return rag_back_project_weights(m_fine_rag.edge_map, coarse_sm_on_fine_rag);
        }

        template<typename T1, typename T2>
        auto align_hierarchy(const xt::xexpression<T1> &xcoarse_supervertices,
                             const hg::tree &tree,
                             const xt::xexpression<T2> &xaltitudes) const {
            HG_TRACE();
            auto &coarse_supervertices = xcoarse_supervertices.derived_cast();
            auto &altitudes = xaltitudes.derived_cast();
            hg_assert_node_weights(tree, altitudes);
            hg_assert_1d_array(altitudes);
            hg_assert_1d_array(coarse_supervertices);
            hg_assert_integral_value_type(coarse_supervertices);
            hg_assert(coarse_supervertices.size() == m_fine_rag.vertex_map.size(),
                      "Cannot align given hierarchy: incompatible sizes!");

            auto coarse_sm_on_fine_rag =
                    alignment_internal::project_hierarchy(m_fine_rag,
                                                           coarse_supervertices,
                                                           tree,
                                                           altitudes);

            return rag_back_project_weights(m_fine_rag.edge_map, coarse_sm_on_fine_rag);
        }

    private:
        region_adjacency_graph m_fine_rag;
    };

    template<typename graph_t, typename T>
    auto make_hierarchy_aligner_from_graph_cut(const graph_t &graph, const xt::xexpression<T> &saliency_map) {
        return hierarchy_aligner(make_region_adjacency_graph_from_graph_cut(graph, saliency_map));
    }

    template<typename graph_t, typename T>
    auto make_hierarchy_aligner_from_labelisation(const graph_t &graph, const xt::xexpression<T> &vertex_labels) {
        return hierarchy_aligner(make_region_adjacency_graph_from_labelisation(graph, vertex_labels));
    }

    template<typename graph_t, typename tree_t, typename T>
    auto make_hierarchy_aligner_from_hierarchy(const graph_t &graph, const tree_t &tree,
                                               const xt::xexpression<T> &altitudes) {
        return hierarchy_aligner(
                make_region_adjacency_graph_from_labelisation(graph, labelisation_hierarchy_supervertices(tree,
                                                                                                          altitudes)));
    }

}
