//
// Created by user on 3/9/18.
//

#pragma once

#include <boost/graph/adjacency_list.hpp>

namespace hg{

    /**
     * Undirected graph with in and out edge lists
     */
    using Graph = boost::adjacency_list<
            boost::vecS, // OutEdgeList
            boost::vecS, // VertexList
            boost::undirectedS, // directed
            boost::no_property, // vertex property
            boost::no_property, // edge property
            boost::no_property, // graph property
            boost::listS>; // edge list





    template <typename IteratorType>
    struct IteratorWrapper{
        IteratorType const & first;
        IteratorType const & last;

        IteratorWrapper(IteratorType & _first, IteratorType & _last): first(_first), last(_last){};

        IteratorWrapper(const std::pair<IteratorType, IteratorType> & p): first(p.first), last(p.second){};

        IteratorType begin(){
            return first;
        }

        IteratorType end(){
            return last;
        }
    };

    template <typename GraphType>
    auto graphVertexIterator(GraphType & g){
        using VertexIterator = typename boost::graph_traits<GraphType>::vertex_iterator;
        return IteratorWrapper<VertexIterator>(boost::vertices(g));
    }

    template <typename GraphType>
    auto graphEdgeIterator(GraphType & g){
        using EdgeIterator = typename boost::graph_traits<Graph>::edge_iterator;
        return IteratorWrapper<EdgeIterator>(boost::edges(g));
    }

    template <typename GraphType>
    auto graphOutEdgeIterator(typename boost::graph_traits<Graph>::vertex_descriptor v, GraphType & g){
        using OutEdgeIterator = typename boost::graph_traits<Graph>::out_edge_iterator;
        return IteratorWrapper<OutEdgeIterator>(boost::out_edges(v, g));
    }

    template <typename GraphType>
    auto graphInEdgeIterator(typename boost::graph_traits<Graph>::vertex_descriptor v, GraphType & g){
        using InEdgeIterator = typename boost::graph_traits<Graph>::in_edge_iterator;
        return IteratorWrapper<InEdgeIterator>(boost::in_edges(v, g));
    }

    template <typename GraphType>
    auto graphAdjacentVertexIterator(typename boost::graph_traits<Graph>::vertex_descriptor v, GraphType & g){
        using AdjacencyIterator = typename boost::graph_traits<Graph>::adjacency_iterator;
        return IteratorWrapper<AdjacencyIterator>(boost::adjacent_vertices(v, g));
    }

}