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

#include "xtensor/xgenerator.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xindex_view.hpp"
#include "../structure/tree_graph.hpp"
#include "../accumulator/tree_accumulator.hpp"
#include "rag.hpp"

namespace hg {

    /**
     * Each leaf of the tree takes the weight of its closest non deleted ancestor.
     *
     * @tparam tree_t
     * @tparam T1
     * @tparam T2
     * @param tree
     * @param altitudes
     * @param deleted_nodes
     * @return
     */
    template<typename tree_t,
            typename T1,
            typename T2>
    auto reconstruct_leaf_data(const tree_t &tree,
                               const xt::xexpression<T1> &altitudes,
                               const xt::xexpression<T2> &deleted_nodes) {
        auto reconstruction = propagate_sequential(tree,
                                                   altitudes,
                                                   deleted_nodes);
        return xt::eval(xt::strided_view(reconstruction, {xt::range(0, num_leaves(tree)), xt::ellipsis()}));
    };

    /**
     * Labelize tree leaves according to an horizontal cut in the tree.
     *
     * Two leaves are in the same region (ie. have the same label) if
     * the altitude of their lowest common ancestor is strictly greater
     * than the specified threshold.
     *
     * The label of a leave l is equal to the index of smallest node containing l
     * and whose altitude is strictly greater than the specified threshold.
     *
     * @tparam tree_t
     * @tparam T
     * @tparam value_t
     * @param tree
     * @param xaltitudes
     * @param threshold
     * @return
     */
    template<typename tree_t,
            typename T,
            typename value_t>
    auto labelisation_horizontal_cut(const tree_t &tree,
                                     const xt::xexpression<T> &xaltitudes,
                                     const value_t threshold) {
        auto &altitudes = xaltitudes.derived_cast();
        return reconstruct_leaf_data(tree,
                                     xt::arange(num_vertices(tree)),
                                     xt::index_view(altitudes, tree.parents())
                                     <= static_cast<typename T::value_type>(threshold));
    };

    /**
     * Labelize the tree leaves into supervertices.
     *
     * Two leaves are in the same supervertex if they have a common ancestor of altitude 0.
     *
     * This functions guaranties that the labels are in the range [0, num_supervertices-1].
     *
     * @tparam tree_t
     * @tparam T
     * @param tree
     * @param xaltitudes
     * @return
     */
    template<typename tree_t,
            typename T>
    auto labelisation_hierarchy_supervertices(const tree_t &tree,
                                     const xt::xexpression<T> &xaltitudes) {
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(tree, altitudes);
        
        auto labels = labelisation_horizontal_cut(tree, altitudes, 0);
        // remap labels to 0...num_labels - 1
        array_1d<index_t> map({num_vertices(tree)}, invalid_index);
        index_t current_label = 0;
        for(auto n: leaves_iterator(tree)){
            auto lbl = labels(n);
            if(map(lbl) == invalid_index){
                map(lbl) = current_label;
                current_label++;
            }
            labels(n) = map(lbl);
        }
        return labels;
    };

    /**
     * Test if 2 trees are isomorph assuming that they share the same leaves.
     *
     * By this definition t1 is isomorph to t2 if there exist a bijection f from vertices(t1) to vertices(t2) such that:
     *   1) for any leaf node n of t1, f(n) = n and n
     *   2) for any node n of t1, f(t1.parent(n)) = t2.parent(f(n))
     *
     * Note that the root r node of a tree t is defined by t.parent(r) = r, thus 2) becomes
     *   for the root node r1 of t1, f(r1) = t2.parent(f(r1)), i.e. f(r1) is the root node of t2
     *
     * @tparam tree1_t
     * @tparam tree2_t
     * @param t1
     * @param t2
     * @return
     */
    template<typename tree1_t, typename tree2_t>
    bool testTreeIsomorphism(const tree1_t &t1, const tree2_t &t2) {
        HG_TRACE();
        if (num_vertices(t1) != num_vertices(t2) || num_leaves(t1) != num_leaves(t2))
            return false;

        long num_v = num_vertices(t1);
        long num_l = num_leaves(t1);
        auto not_defined = num_v;

        auto f = xt::eval(not_defined + xt::zeros<long>({num_v}));


        for (auto i: leaves_to_root_iterator(t1)) {
            if (i < num_l)
                f[i] = i; // 1) for any leaf node n of t1, f(n) = n and n

            auto t1pi = parent(i, t1); // t1.parent(n)
            long t2pfi = parent(f(i), t2); //t2.parent(f(n))

            if (f(t1pi) == not_defined) { // f(t1.parent(n)) not defined
                f(t1pi) = t2pfi; // f(t1.parent(n)) = t2.parent(f(n))
            } else if (f(t1pi) != t2pfi) { // f(t1.parent(n)) was already set by a brother of n and is not compatible
                return false;
            }// else ok
        }
        return true;
    }

}
