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

#ifdef HG_USE_BOOST_GRAPH
#include <boost/graph/graph_concepts.hpp>
#endif

namespace hg {
    namespace graph {
#ifdef HG_USE_BOOST_GRAPH

        using directed_tag = boost::directed_tag;
        using undirected_tag = boost::undirected_tag;
        using bidirectional_tag = boost::bidirectional_tag;

        using incidence_graph_tag = boost::incidence_graph_tag;
        using adjacency_graph_tag = boost::adjacency_graph_tag;
        using bidirectional_graph_tag = boost::bidirectional_graph_tag;
        using vertex_list_graph_tag = boost::vertex_list_graph_tag;
        using edge_list_graph_tag = boost::edge_list_graph_tag;
        using adjacency_matrix_tag = boost::adjacency_matrix_tag;

        using allow_parallel_edge_tag = boost::allow_parallel_edge_tag;
        using disallow_parallel_edge_tag = boost::disallow_parallel_edge_tag;

        using graph_traits = boost::graph_traits;

#else

        struct directed_tag {
        };
        struct undirected_tag {
        };
        struct bidirectional_tag : virtual public directed_tag {
        };

        struct incidence_graph_tag {
        };
        struct adjacency_graph_tag {
        };
        struct bidirectional_graph_tag : virtual public incidence_graph_tag {
        };
        struct vertex_list_graph_tag {
        };
        struct edge_list_graph_tag {
        };
        struct adjacency_matrix_tag {
        };

        struct allow_parallel_edge_tag {
        };
        struct disallow_parallel_edge_tag {
        };

        template<typename T>
        struct graph_traits;

#endif
    }
    using graph::graph_traits;
}
