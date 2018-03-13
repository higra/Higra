//
// Created by user on 3/12/18.
//

#pragma once

#include <boost/graph/adjacency_list.hpp>

namespace hg {
/**
  * Undirected graph with in and out edge lists
  */
    using undirected_graph = boost::adjacency_list<
            boost::vecS, // OutEdgeList
            boost::vecS, // VertexList
            boost::undirectedS, // directed
            boost::no_property, // vertex property
            boost::no_property, // edge property
            boost::no_property, // graph property
            boost::listS>; // edge list
}