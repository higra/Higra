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
            array_1d<typename T1::value_type> altitudes = array_1d<typename T1::value_type>::from_shape(
                    {res.second.size()});
            std::copy(res.second.begin(), res.second.end(), altitudes.begin());
            array_1d<index_t> tree_parents = array_1d<index_t>::from_shape({res.first.size()});
            std::copy(res.first.begin(), res.first.end(), tree_parents.begin());
            return make_node_weighted_tree(
                    tree(std::move(tree_parents), tree_category::component_tree),
                    std::move(altitudes));
        }
    }

    /**
     * Compute the max tree of the given vertex weighted graph.
     *
     * For a threshold \f$k\f$, the upper level set \f$U_k\f$ of the vertex weights \f$w\f$ is the set of the vertices
     * whose weight is greater than or equal to \f$k\f$:
     *
     * \f[
     *     U_k = \{v \in V \mid w(v) \geq k\}
     * \f]
     *
     * where \f$V\f$ is the vertex set of the graph. The max tree is the tree of the connected components of the upper
     * level sets of \f$w\f$, for all thresholds \f$k\f$, ordered by inclusion. Each component appears only once, even
     * if it is a connected component of several upper level sets: its altitude is the largest threshold for which it is
     * a connected component, that is the smallest weight of its vertices:
     *
     * \f[
     *     \mathrm{alt}(C) = \min_{v \in C} w(v)
     * \f]
     *
     * The max tree was proposed in [1] and [2], and the algorithm used in this implementation was first described
     * in [3].
     *
     * The resulting tree is made of:
     *
     * - the leaves, which are the vertices of the graph: the leaf \f$v\f$ has the altitude \f$w(v)\f$, and its parent
     *   is the connected component of \f$U_{w(v)}\f$ containing \f$v\f$, which has the same altitude as \f$v\f$;
     * - the non-leaf nodes, which are the connected components of the upper level sets. The altitudes are decreasing
     *   from the leaves to the root (for any node \f$n\f$, \f$\mathrm{alt}(n) \geq \mathrm{alt}(parent(n))\f$), and a
     *   non-leaf node never has the same altitude as its parent, except the root, which is its own parent. The root is
     *   the whole vertex set, with the smallest vertex weight as altitude.
     *
     * The graph must have at least one vertex (an exception is raised otherwise) and must be connected (the result is
     * undefined otherwise). The vertex weights must be scalars: vertex_weights is a 1d array with one element per
     * vertex.
     *
     * Complexity: The computation takes \f$\mathcal{O}(n\log(n) + m\,\alpha(n))\f$ time, where \f$n\f$ is the number
     * of vertices and \f$m\f$ the number of edges of the graph, and \f$\alpha\f$ is the inverse Ackermann function
     * (sorting of the vertices by weight followed by a union-find over the edges).
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
     * @tparam graph_t graph type
     * @tparam T element type of vertex_weights
     * @param graph input graph, connected and with at least one vertex
     * @param vertex_weights weight of the vertices of the input graph (1d array with one element per vertex)
     * @return a node weighted tree: the tree and its node altitudes (1d array with one element per node of the tree,
     * with the same element type as vertex_weights)
     */
    template<typename graph_t, typename T>
    auto component_tree_max_tree(const graph_t &graph, array_view<const T> vertex_weights) {
        HG_TRACE();
        hg_assert_vertex_weights(graph, vertex_weights);
        hg_assert_1d_array(vertex_weights);
        hg_assert(num_vertices(graph) > 0, "The graph must have at least one vertex.");

        array_1d<index_t> sorted_vertex_indices = stable_arg_sort(vertex_weights);
        return component_tree_internal::tree_from_sorted_vertices(graph, vertex_weights, sorted_vertex_indices);
    }

    /**
     * Compute the min tree of the given vertex weighted graph.
     *
     * For a threshold \f$k\f$, the lower level set \f$L_k\f$ of the vertex weights \f$w\f$ is the set of the vertices
     * whose weight is smaller than or equal to \f$k\f$:
     *
     * \f[
     *     L_k = \{v \in V \mid w(v) \leq k\}
     * \f]
     *
     * where \f$V\f$ is the vertex set of the graph. The min tree is the tree of the connected components of the lower
     * level sets of \f$w\f$, for all thresholds \f$k\f$, ordered by inclusion. Each component appears only once, even
     * if it is a connected component of several lower level sets: its altitude is the smallest threshold for which it
     * is a connected component, that is the largest weight of its vertices:
     *
     * \f[
     *     \mathrm{alt}(C) = \max_{v \in C} w(v)
     * \f]
     *
     * The min tree was proposed in [1] and [2], and the algorithm used in this implementation was first described
     * in [3].
     *
     * The resulting tree is made of:
     *
     * - the leaves, which are the vertices of the graph: the leaf \f$v\f$ has the altitude \f$w(v)\f$, and its parent
     *   is the connected component of \f$L_{w(v)}\f$ containing \f$v\f$, which has the same altitude as \f$v\f$;
     * - the non-leaf nodes, which are the connected components of the lower level sets. The altitudes are increasing
     *   from the leaves to the root (for any node \f$n\f$, \f$\mathrm{alt}(n) \leq \mathrm{alt}(parent(n))\f$), and a
     *   non-leaf node never has the same altitude as its parent, except the root, which is its own parent. The root is
     *   the whole vertex set, with the largest vertex weight as altitude.
     *
     * The graph must have at least one vertex (an exception is raised otherwise) and must be connected (the result is
     * undefined otherwise). The vertex weights must be scalars: vertex_weights is a 1d array with one element per
     * vertex.
     *
     * Complexity: The computation takes \f$\mathcal{O}(n\log(n) + m\,\alpha(n))\f$ time, where \f$n\f$ is the number
     * of vertices and \f$m\f$ the number of edges of the graph, and \f$\alpha\f$ is the inverse Ackermann function
     * (sorting of the vertices by weight followed by a union-find over the edges).
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
     * @tparam graph_t graph type
     * @tparam T element type of vertex_weights
     * @param graph input graph, connected and with at least one vertex
     * @param vertex_weights weight of the vertices of the input graph (1d array with one element per vertex)
     * @return a node weighted tree: the tree and its node altitudes (1d array with one element per node of the tree,
     * with the same element type as vertex_weights)
     */
    template<typename graph_t, typename T>
    auto component_tree_min_tree(const graph_t &graph, array_view<const T> vertex_weights) {
        HG_TRACE();
        hg_assert_vertex_weights(graph, vertex_weights);
        hg_assert_1d_array(vertex_weights);
        hg_assert(num_vertices(graph) > 0, "The graph must have at least one vertex.");

        array_1d<index_t> sorted_vertex_indices = stable_arg_sort(vertex_weights,
                                                                  std::greater<T>());
        return component_tree_internal::tree_from_sorted_vertices(graph, vertex_weights, sorted_vertex_indices);
    }

}
