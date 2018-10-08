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
#include "graph_concepts.hpp"

namespace hg {

    /**
     * An edge with a source vertex, a target vertex, and an index.
     * @tparam vertex_descriptor
     * @tparam edge_index_t
     */
    template<typename vertex_descriptor,
            typename edge_index_t>
    class indexed_edge {

    public:
        union{vertex_descriptor source; vertex_descriptor first;};
        union{vertex_descriptor target; vertex_descriptor second;};
        edge_index_t index;

        indexed_edge(vertex_descriptor _source,
                     vertex_descriptor _target,
                     edge_index_t _index) :
                source(_source),
                target(_target),
                index(_index) {
        }

        indexed_edge(const std::pair<vertex_descriptor, vertex_descriptor> & edge,
                     edge_index_t _index) :
                source(edge.first),
                target(edge.second),
                index(_index) {
        }

        operator edge_index_t() const { return index; }

        operator std::pair<vertex_descriptor, vertex_descriptor>() const {
            return std::make_pair(source, target);
        }
    };

    /**
     * Source vertex of an edge
     * @tparam graph_t
     * @param e
     * @return
     */
    template<typename graph_t>
    auto & source(
            const indexed_edge<typename graph::graph_traits<graph_t>::vertex_descriptor,
                    typename graph::graph_traits<graph_t>::edge_index> &e,
            const graph_t &) {
        return e.source;
    }

    /**
     * Target vertex of an edge
     * @tparam graph_t
     * @param e
     * @return
     */
    template<typename graph_t>
    auto & target(
            const indexed_edge<typename graph::graph_traits<graph_t>::vertex_descriptor,
                    typename graph::graph_traits<graph_t>::edge_index> &e,
            const graph_t &) {
        return e.target;
    }

    /**
    * Index of an edge
    * @tparam graph_t
    * @param e
    * @return
    */
    template<typename graph_t>
    auto & index(
            const indexed_edge<typename graph::graph_traits<graph_t>::vertex_descriptor,
                    typename graph::graph_traits<graph_t>::edge_index> &e,
            const graph_t &) {
        return e.index;
    }
}
