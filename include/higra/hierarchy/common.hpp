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


namespace hg {

    /**
     * A simple structure to hold the result of hierarchy construction algorithms.
     *
     * See make_node_weighted_tree for construction
     *
     * @tparam tree_t
     * @tparam altitude_t
     */
    template <typename tree_t, typename altitude_t>
    struct node_weighted_tree{
        tree_t tree;
        altitude_t altitudes;
    };

    template <typename tree_t, typename altitude_t>
    decltype(auto) make_node_weighted_tree(tree_t && tree, altitude_t && node_altitude){
        return node_weighted_tree<tree_t, altitude_t>{std::forward<tree_t>(tree), std::forward<altitude_t>(node_altitude)};
    }

    /**
     * A simple structure to hold the result of a remapping operation on the nodes of a tree.
     * When algorithm transforms a tree into a new tree by removing or duplicating or reordering some of its nodes,
     * it is useful to know the relation between the nodes of the new tree and the nodes of the original one.
     *
     * For each node i of the new tree, node_map[i] gives the corresponding node in the original tree.
     *
     * @tparam tree_t
     * @tparam node_map_t
     */
    template<typename tree_t, typename node_map_t>
    struct remapped_tree {
        tree_t tree;
        node_map_t node_map;
    };

    template <typename tree_t, typename node_map_t>
    decltype(auto) make_remapped_tree(tree_t && tree, node_map_t && node_map){
        return remapped_tree<tree_t, node_map_t>{std::forward<tree_t>(tree), std::forward<node_map_t>(node_map)};
    }
}
