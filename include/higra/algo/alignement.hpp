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

#include "../graph.hpp"
#include "rag.hpp"
#include "higra/structure/lca_fast.hpp"
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

        auto &labelisation_fine = xlabelisation_fine.derived_cast();
        auto &labelisation_coarse = xlabelisation_coarse.derived_cast();

        hg_assert_integral_value_type(labelisation_fine);
        hg_assert_integral_value_type(labelisation_coarse);
        hg_assert_1d_array(labelisation_fine);
        hg_assert_1d_array(labelisation_coarse);
        hg_assert(labelisation_fine.size() == labelisation_coarse.size(),
                  "Labelisations must have the same size.");

        if(num_regions_fine == 0){
            num_regions_fine = xt::amax(labelisation_fine)(0) + 1;
        }

        if(num_regions_coarse == 0){
            num_regions_coarse = xt::amax(labelisation_coarse)(0) + 1;
        }

        array_2d <size_t> intersections = xt::zeros<size_t>({num_regions_fine, num_regions_coarse});

        for (index_t i = 0; i < labelisation_fine.size(); i++) {
            intersections(labelisation_fine(i), labelisation_coarse(i))++;
        }

        return xt::eval(xt::argmax(intersections, 1));
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

    namespace alignement_internal{

        template <typename rag_t, typename T, typename tree_t, typename T2>
        auto project_hierarchy(const rag_t & rag_fine, const T & coarse_supervertices, const tree_t & tree_coarse, const T2 & tree_coarse_node_altitudes){
            auto & fine_supervertices = rag_fine.vertex_map;
            auto & rag = rag_fine.rag;

            auto fine_to_coarse_map = project_fine_to_coarse_labelisation(fine_supervertices, coarse_supervertices);
            array_1d<typename T2::value_type> coarse_sm_on_fine_rag = xt::empty({num_edges(rag_fine.rag)});

            lca_fast lca(tree_coarse);
            for(auto e: edge_iterator(rag)){
                auto projected_lca = lca.lca(fine_to_coarse_map[source(e, rag)], fine_to_coarse_map[target(e, rag)]);
                coarse_sm_on_fine_rag(index(e, rag)) = tree_coarse_node_altitudes(projected_lca);
            }

            return coarse_sm_on_fine_rag;
        }
    }
/*

    class hierarchy_aligner{
                public:

                    template<typename graph_t, typename T>
                    hierarchy_aligner(const graph_t & graph, const xt::xexpression<T> & saliency_map) :
                    m_fine_rag(make_region_adjacency_graph(graph, saliency_map)){
                    }

                    template<typename graph_t, typename tree_t>
                    hierarchy_aligner(const graph_t & graph, const tree_t & tree) :
                    fine_rag()

                private:
                    region_adjacency_graph m_fine_rag;
                };
*/


}
