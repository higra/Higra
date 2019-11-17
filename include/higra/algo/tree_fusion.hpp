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
#include "../attribute/tree_attribute.hpp"
#include <xtensor/xnoalias.hpp>
#include <vector>
#include <stack>

namespace hg {

    namespace tree_fusion_internal {

        using namespace std;

        template<typename tree_iterator>
        auto tree_fusion_depth_map(const tree_iterator first, const tree_iterator last) {

            index_t i, j;
            tree_iterator ti, tj;
            auto ntrees = last - first;
            hg_assert(ntrees > 1, "Fusion requires at least two trees");
            auto nleaves = num_leaves(**first);
            for (tree_iterator t = first; t != last; t++) {
                hg_assert(num_leaves(**t), "All trees must have the same number of leaves.");
            }

            // cheap digraph!
            vector<vector<index_t>> adj_lists;

            // precompute areas and smallest enclosing shapes
            vector<array_1d<index_t>> areas;
            array_2d<array_1d<index_t>> ses = xt::empty<array_1d<index_t>>({ntrees, ntrees});
            for (ti = first, i = 0; ti != last; ti++, i++) {
                areas.push_back(attribute_area(**ti));
                for (tj = first, j = 0; tj != last; tj++, j++) {
                    if (j != i) {
                        ses(i, j) = attribute_smallest_enclosing_shape(**ti, **tj);
                    }
                }
            }

            /* ***************
             * Add nodes to the graph of shapes (GOS)
             */

            // associate each node of each tree to a node of the GOS
            vector<array_1d<index_t>> node_maps;

            // add leaves
            adj_lists.resize(nleaves);

            // add internal nodes (except root) and avoid duplication
            for (ti = first, i = 0; ti != last; ti++, i++) {
                node_maps.emplace_back(array_1d<index_t>::from_shape({num_vertices(**ti)}));
                // create leaves association
                xt::noalias(xt::view(node_maps[i], xt::range(0, nleaves))) = xt::arange<index_t>(nleaves);

                for (index_t n: leaves_to_root_iterator(**ti, leaves_it::exclude, root_it::exclude)) {
                    bool keep = true;
                    for (index_t j = 0; j < i && keep; j++) {
                        auto ses_ij_n = ses(i, j)(n);
                        if (areas[j](ses_ij_n) == areas[i](n)) {
                            keep = false;
                            node_maps[i](n) = node_maps[j](ses_ij_n);
                        }
                    }
                    if (keep) {
                        node_maps[i](n) = adj_lists.size();
                        adj_lists.emplace_back();
                    }
                }
            }

            // add root
            auto rootn = adj_lists.size();
            adj_lists.emplace_back();
            for (ti = first, i = 0; ti != last; ti++, i++) {
                node_maps[i](root(**ti)) = rootn;
            }


            /* ***************
             * Add edges to the graph of shapes (GOS)
             */
            for (ti = first, i = 0; ti != last; ti++, i++) {
                for (index_t n: leaves_to_root_iterator(**ti, leaves_it::include, root_it::exclude)) {
                    auto represent_n = node_maps[i](n);
                    adj_lists[node_maps[i](parent(n, **ti))].push_back(represent_n);
                    for (j = 0; j < (index_t) ntrees; j++) {
                        if (i != j) {
                            auto ses_ij_n = ses(i, j)(n);
                            if (areas[j](ses_ij_n) != areas[i](n)) {
                                adj_lists[node_maps[j](ses_ij_n)].push_back(represent_n);
                            }
                        }

                    }
                }
            }

            /* ***************
            * Transitive reduction of the GOS
            */

            // perhaps one day, doesn't bring anything in terms of complexity

            /* ***************
            * Topological sort of the GOS
            */
            auto nnodes = adj_lists.size();
            array_1d<index_t> sorted_nodes = xt::empty<index_t>({nnodes});
            // marks: 0 = never seen, 1 = being visited (not finalized and sucessors on the stack), 2 = sorted
            array_1d<char> marks = xt::zeros<char>({nnodes});
            stack<index_t> s;

            index_t count = 0;
            s.push(rootn);
            while (!s.empty()) {
                auto n = s.top();
                if (marks(n) > 0) {
                    s.pop();
                    if (marks(n) == 1) {
                        sorted_nodes(count++) = n;
                        marks(n) = 2;
                    }
                } else {
                    marks(n) = 1;
                    for (auto o: adj_lists[n]) {
                        if (marks(o) != 2) {
                            s.push(o);
                        }
                    }
                }
            }

            /* ***************
            * Depth of the nodes of the GOS
            */
            array_1d<index_t> depth = xt::zeros<index_t>({nnodes});
            for (index_t i = nnodes - 1; i >= 0; i--) {
                index_t n = sorted_nodes[i];
                for (auto o: adj_lists[n]) {
                    depth(o) = (std::max)(depth(o), depth(n) + 1);
                }
            }

            /* ***************
            * Depth of the nodes of the GOS
            */
            return xt::eval(xt::view(depth, xt::range(0, nleaves)));
        }

        template<typename range_tree_t>
        auto tree_fusion_depth_map(const range_tree_t &range) {
            return tree_fusion_depth_map(range.begin(), range.end());
        }

    }

    /**
     * Depth map associated to the fusion of the given list of trees.
     *
     * The method is described in:
     *
     * > E. Carlinet. A Tree of shapes for multivariate images. PhD Thesis, Université Paris-Est, 2015.
     *
     * All trees must be defined over the same domain, i.e. have the same number of leaves.
     *
     * Given a set of trees (T_1, T_2, T_3,... T_n) composed of the nodes (N_1, N_2, N_3, ... N_n).
     * We define the fusion graph as the graph induced the inclusion relation on the union of all the tree nodes
     * \bigcup\{N_1, N_2, N_3, ... N_n\}.
     * The result is a directed acyclic graph with a single root (corresponding to the roots of the input trees).
     * The depth of a node in this graph is defined as the length of the longest path from the root this node.
     *
     * This function returns the depth of the leaves of this graph (which are the same as the leaves of the input trees).
     *
     * @tparam tree_iterator Iterator on tree pointers
     * @param first
     * @param last
     * @return an 1d array of size num_leaves(**first)
     */
    template<typename tree_iterator>
    auto tree_fusion_depth_map(const tree_iterator first, const tree_iterator last) {
        static_assert(std::is_same<typename tree_iterator::value_type, tree *>::value,
                      "Iterator value type must 'tree *'.");
        return tree_fusion_internal::tree_fusion_depth_map(first, last);
    }

    template<typename range_tree_t>
    auto tree_fusion_depth_map(const range_tree_t &range) {
        return tree_fusion_internal::tree_fusion_depth_map(range.begin(), range.end());
    }


}