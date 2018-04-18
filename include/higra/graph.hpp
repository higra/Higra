//
// Created by user on 3/9/18.
//

#pragma once

#include "utils.hpp"
#include <boost/graph/graph_traits.hpp>
#include "structure/undirected_graph.hpp"
#include "structure/regular_graph.hpp"
#include "structure/tree_graph.hpp"
#include "structure/graph_utils.hpp"
#include "structure/array.hpp"


namespace hg {

    template<typename T>
    auto copy_graph(const T &graph) {
        static_assert(
                std::is_base_of<boost::adjacency_graph_tag, typename boost::graph_traits<T>::traversal_category>::value,
                "Graph must implement adjacency graph concept.");
        static_assert(
                std::is_base_of<boost::vertex_list_graph_tag, typename boost::graph_traits<T>::traversal_category>::value,
                "Graph must implement vertex list graph concept.");

        ugraph g(num_vertices(graph));
        for (const auto v : vertex_iterator(graph)) {
            for (const auto n : adjacent_vertex_iterator(v, graph)) {
                if (n > v)
                    g.add_edge(v, n);
            }
        }
        return g;
    };

    template<>
    inline
    auto copy_graph(const ugraph &graph) {
        ugraph g(num_vertices(graph));
        for (const auto e : edge_iterator(graph)) {
            g.add_edge(source(e, graph), target(e, graph));
        }
        return g;
    };



}