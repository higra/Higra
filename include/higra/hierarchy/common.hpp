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
}
