//
// Created by user on 3/13/18.
//

#pragma once


namespace hg {

    template<typename iterator_t>
    struct iterator_wrapper {
        iterator_t const first;
        iterator_t const last;

        iterator_wrapper(iterator_t &_first, iterator_t &_last) : first(_first), last(_last) {};

        iterator_wrapper(const std::pair<iterator_t, iterator_t> &p) : first(p.first), last(p.second) {};

        iterator_t begin() {
            return first;
        }

        iterator_t end() {
            return last;
        }
    };

    template<typename graph_t>
    auto graphVertexIterator(graph_t &g) {
        using vertex_iterator_t = typename boost::graph_traits<graph_t>::vertex_iterator;
        return iterator_wrapper<vertex_iterator_t>(boost::vertices(g));
    }

    template<typename graph_t>
    auto graphEdgeIterator(graph_t &g) {
        using edge_iterator_t = typename boost::graph_traits<graph_t>::edge_iterator;
        return iterator_wrapper<edge_iterator_t>(boost::edges(g));
    }

    template<typename graph_t>
    auto graphOutEdgeIterator(typename boost::graph_traits<graph_t>::vertex_descriptor v, graph_t &g) {
        using out_edge_iterator_t = typename boost::graph_traits<graph_t>::out_edge_iterator;
        return iterator_wrapper<out_edge_iterator_t>(boost::out_edges(v, g));
    }

    template<typename graph_t>
    auto graphInEdgeIterator(typename boost::graph_traits<graph_t>::vertex_descriptor v, graph_t &g) {
        using in_edge_iterator_t = typename boost::graph_traits<graph_t>::in_edge_iterator;
        return iterator_wrapper<in_edge_iterator_t>(boost::in_edges(v, g));
    }

    template<typename graph_t>
    auto graphAdjacentVertexIterator(typename boost::graph_traits<graph_t>::vertex_descriptor v, graph_t &g) {
        using adjacency_iterator_t = typename boost::graph_traits<graph_t>::adjacency_iterator;
        return iterator_wrapper<adjacency_iterator_t>(boost::adjacent_vertices(v, g));
    }
}
