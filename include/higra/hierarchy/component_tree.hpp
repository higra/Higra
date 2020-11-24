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

#include "common.hpp"
#include "higra/structure/unionfind.hpp"
#include "higra/graph.hpp"
#include "higra/sorting.hpp"
#include "xtensor/xadapt.hpp"

namespace hg {
    namespace component_tree_internal {

        /**
         * Generic pre-tree construction from ordered vertex values
         *
         * @tparam graph_t
         * @tparam T
         * @tparam E
         * @param graph
         * @param vertex_values
         * @param sorted_vertex_indices
         * @return
         */
        template<typename graph_t, typename E>
        auto pre_tree_construction(const graph_t &graph,
                                   const E &sorted_vertex_indices) {
            auto nbe = num_vertices(graph);
            array_1d<index_t> parent = array_1d<index_t>::from_shape({nbe});
            array_1d<index_t> representing = array_1d<index_t>::from_shape({nbe});
            array_1d<bool> processed({nbe}, false);
            union_find uf(nbe);

            for (index_t i = nbe - 1; i >= 0; i--) {
                auto current_vertex = sorted_vertex_indices[i];
                parent(current_vertex) = current_vertex;
                representing(current_vertex) = current_vertex;
                processed(current_vertex) = true;
                auto current_vertex_reprez = current_vertex;
                for (auto n: adjacent_vertex_iterator(current_vertex, graph)) {
                    if (processed(n)) {
                        auto neighbor_component = uf.find(n);
                        if (neighbor_component != current_vertex_reprez) {
                            parent[representing[neighbor_component]] = current_vertex;
                            current_vertex_reprez = uf.link(neighbor_component, current_vertex_reprez);
                            representing(current_vertex_reprez) = current_vertex;
                        }
                    }
                }
            }
            return parent;
        }

        /**
         * Parent relation "canonization" (path compression) after pre_tree_construction
         *
         * Parent relation is modified in-place!
         *
         * @tparam T1
         * @tparam T2
         * @tparam T3
         * @param parents a pre-parent relation as constructed by pre_tree_construction
         * @param vertex_weights the node levels associated to the pre-parent relation
         * @param sorted_vertex_indices the sorted vertex indices
         * @return void
         */
        template<typename T1, typename T2, typename T3>
        void canonize_tree(T1 &parents, const T2 &vertex_weights, const T3 &sorted_vertex_indices) {
            for (auto e: sorted_vertex_indices) {
                auto par = parents[e];
                if (vertex_weights[parents[par]] == vertex_weights[par]) {
                    parents[e] = parents[par];
                }
            }
        }

        /**
         * Expand a canonized parent relation to a regular parent relation (each node is represented individually)
         * @tparam T1
         * @tparam T2
         * @tparam T3
         * @param parents a canonized parent relation
         * @param vertex_weights the node levels associated to the canonized parent relation
         * @param sorted_vertex_indices the sorted vertex indices
         * @return
         */
        template<typename T1, typename T2, typename T3>
        auto expand_canonized_parent_relation(
                const T1 &parents,
                const T2 &vertex_weights,
                const T3 &sorted_vertex_indices) {
            index_t nbe = parents.size();
            std::vector<typename T2::value_type> altitudes(vertex_weights.begin(), vertex_weights.end());
            std::vector<index_t> new_parents(nbe, invalid_index);

            for (index_t j = nbe - 1; j >= 0; j--) {
                auto i = sorted_vertex_indices[j];
                auto par = (vertex_weights[i] != vertex_weights[parents[i]]) ? i : parents[i];
                if (new_parents[par] == invalid_index) {
                    new_parents.push_back(nbe - 1);
                    new_parents[par] = nbe;
                    nbe++;
                    altitudes.push_back(vertex_weights[par]);
                }
                new_parents[i] = new_parents[par];
            }

            for (index_t j = sorted_vertex_indices.size() - 1; j >= 0; j--) {
                auto i = sorted_vertex_indices[j];
                if (vertex_weights[i] != vertex_weights[parents[i]]) {
                    auto par = i;
                    auto ppar = parents[par];
                    new_parents[new_parents[par]] = new_parents[ppar];
                }
            }
            new_parents[new_parents.size() - 1] = new_parents.size() - 1;
            return std::make_pair(std::move(new_parents), std::move(altitudes));
        }

        template<typename graph_t, typename T1, typename T2>
        auto
        tree_from_sorted_vertices(const graph_t &graph, const T1 &vertex_weights, const T2 &sorted_vertex_indices) {
            auto parents = pre_tree_construction(graph, sorted_vertex_indices);
            canonize_tree(parents, vertex_weights, sorted_vertex_indices);
            auto res = expand_canonized_parent_relation(parents, vertex_weights, sorted_vertex_indices);
            array_1d<typename T1::value_type> altitudes = xt::adapt(res.second, {res.second.size()});
            return make_node_weighted_tree(
                    tree(xt::adapt(res.first, {res.first.size()}), tree_category::component_tree),
                    std::move(altitudes));
        }
    }

    /**
     * Construct the Max Tree of the vertex weighted graph.
     *
     * The Min/Max Tree structure were proposed in [1], [2].
     * The algorithm used in this implementation was first described in [3].
     *
     * [1] Ph. Salembier, A. Oliveras, and L. Garrido, "Anti-extensive connected operators for image
     * and sequence processing," IEEE Trans. Image Process., vol. 7, no. 4, pp. 555-570, Apr. 1998.
     *
     * [2] Ro. Jones, "Connected filtering and segmentation using component trees," Comput. Vis.
     * Image Understand., vol. 75, no. 3, pp. 215-228, Sep. 1999.
     *
     * [3] Ch. Berger, T. Geraud, R. Levillain, N. Widynski, A. Baillard, and E. Bertin, "Effective
     * Component Tree Computation with Application to Pattern Recognition in Astronomical Imaging,"
     * IEEE ICIP 2007.
     *
     * @tparam graph_t
     * @tparam T
     * @param graph input graph
     * @param vertex_weights graph vertex weights
     * @return a node weighted tree
     */
    template<typename graph_t, typename T>
    auto component_tree_max_tree(const graph_t &graph, const xt::xexpression<T> &xvertex_weights) {
        HG_TRACE();
        auto &vertex_weights = xvertex_weights.derived_cast();
        hg_assert_vertex_weights(graph, vertex_weights);
        hg_assert_1d_array(vertex_weights);

        array_1d<index_t> sorted_vertex_indices = stable_arg_sort(vertex_weights);
        return component_tree_internal::tree_from_sorted_vertices(graph, vertex_weights, sorted_vertex_indices);
    }

    /**
    * Construct the Min Tree of the vertex weighted graph.
    *
    * The Min/Max Tree structure were proposed in [1], [2].
    * The algorithm used in this implementation was first described in [3].
    *
    * [1] Ph. Salembier, A. Oliveras, and L. Garrido, "Anti-extensive connected operators for image
    * and sequence processing," IEEE Trans. Image Process., vol. 7, no. 4, pp. 555-570, Apr. 1998.
    *
    * [2] Ro. Jones, "Connected filtering and segmentation using component trees," Comput. Vis.
    * Image Understand., vol. 75, no. 3, pp. 215-228, Sep. 1999.
    *
    * [3] Ch. Berger, T. Geraud, R. Levillain, N. Widynski, A. Baillard, and E. Bertin, "Effective
    * Component Tree Computation with Application to Pattern Recognition in Astronomical Imaging,"
    * IEEE ICIP 2007.
    *
    * @tparam graph_t
    * @tparam T
    * @param graph input graph
    * @param vertex_weights graph vertex weights
    * @return a node weighted tree
    */
    template<typename graph_t, typename T>
    auto component_tree_min_tree(const graph_t &graph, const xt::xexpression<T> &xvertex_weights) {
        HG_TRACE();
        auto &vertex_weights = xvertex_weights.derived_cast();
        hg_assert_vertex_weights(graph, vertex_weights);
        hg_assert_1d_array(vertex_weights);

        array_1d<index_t> sorted_vertex_indices = stable_arg_sort(vertex_weights,
                                                                  std::greater<typename T::value_type>());
        return component_tree_internal::tree_from_sorted_vertices(graph, vertex_weights, sorted_vertex_indices);
    }

}
