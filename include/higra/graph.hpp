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

#include "utils.hpp"
#include "structure/undirected_graph.hpp"
#include "structure/regular_graph.hpp"
#include "structure/tree_graph.hpp"

namespace hg {

    /**
     * Source vertex of an edge
     * @tparam graph_t
     * @param e
     * @return
     */
    template<typename graph_t>
    auto source(
            const std::pair<typename graph::graph_traits<graph_t>::vertex_descriptor,
                    typename graph::graph_traits<graph_t>::vertex_descriptor> &e,
            const graph_t &) {
        return e.first;
    }

    /**
     * Target vertex of an edge
     * @tparam graph_t
     * @param e
     * @return
     */
    template<typename graph_t>
    auto target(
            const std::pair<typename graph::graph_traits<graph_t>::vertex_descriptor,
                    typename graph::graph_traits<graph_t>::vertex_descriptor> &e,
            const graph_t &) {
        return e.second;
    }

    /**
     * Simple wrapper over two iterators to create a "range" object usable in foreach loops
     * @tparam iterator_t
     */
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

        iterator_t begin() const {
            return first;
        }

        iterator_t end() const {
            return last;
        }
    };

    /**
     * Range over all vertices of the given graph
     * @tparam graph_t
     * @param g
     * @return
     */
    template<typename graph_t>
    auto vertex_iterator(const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::vertex_iterator;
        return iterator_wrapper<it_t>(vertices(g));
    }

    /**
     * Range over all edges of the given graph
     * @tparam graph_t
     * @param g
     * @return
     */
    template<typename graph_t>
    auto edge_iterator(const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::edge_iterator;
        return iterator_wrapper<it_t>(edges(g));
    }

    /**
     * Range over all edges whose source is the given vertex in the given graph
     * @tparam graph_t
     * @param v
     * @param g
     * @return
     */
    template<typename graph_t>
    auto out_edge_iterator(typename graph::graph_traits<graph_t>::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::out_edge_iterator;
        return iterator_wrapper<it_t>(out_edges(v, g));
    }

    /**
     * Range over all edges whose target is the given vertex in the given graph
     * @tparam graph_t
     * @param v
     * @param g
     * @return
     */
    template<typename graph_t>
    auto in_edge_iterator(typename graph::graph_traits<graph_t>::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::in_edge_iterator;
        return iterator_wrapper<it_t>(in_edges(v, g));
    }

    /**
     * Range over all vertices adjacent to the given vertex
     * @tparam graph_t
     * @param v
     * @param g
     * @return
     */
    template<typename graph_t>
    auto adjacent_vertex_iterator(typename graph::graph_traits<graph_t>::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::adjacency_iterator;
        return iterator_wrapper<it_t>(adjacent_vertices(v, g));
    }

    /**
     * Range over the children vertices of the given node in the given tree
     * @tparam graph_t
     * @param v
     * @param g
     * @return
     */
    template<typename graph_t>
    auto children_iterator(typename graph_t::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph_t::children_iterator;
        return iterator_wrapper<it_t>(children(v, g));
    }

    /**
     * Range over the ancestors of v in topological order (starting from v included)
     *
     * @tparam graph_t
     * @param v
     * @param g
     * @return
     */
    template<typename graph_t>
    auto ancestors_iterator(typename graph_t::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph_t::ancestors_iterator;
        return iterator_wrapper<it_t>(ancestors(v, g));
    }

    /**
     * Degrees of all the given vertices in the given graph
     * @tparam T type of indices (must be integral, preferably index_t)
     * @tparam graph_t
     * @param xindex array of vertex indices
     * @param g
     * @return array of the same size as xindex containing the degree of each vertex indicated in xindex
     */
    template<typename T, typename graph_t>
    auto degree(const xt::xexpression<T> &xindex, const graph_t &g) {
        static_assert(std::is_integral<typename T::value_type>::value,
                      "Vertex indices must have integral value type.");
        auto &index = xindex.derived_cast();
        auto f = xt::flatten(index);
        array_nd<size_t> res = array_nd<size_t>::from_shape({index.size()});
        for (index_t i = 0; i < (index_t) res.size(); ++i) {
            res(i) = degree(f(i), g);
        }
        res.reshape(index.shape());
        return res;
    };

    /**
     * In-degrees of all the given vertices in the given graph
     * @tparam T type of indices (must be integral, preferably index_t)
     * @tparam graph_t
     * @param xindex array of vertex indices
     * @param g
     * @return array of the same size as xindex containing the in-degree of each vertex indicated in xindex
     */
    template<typename T, typename graph_t>
    auto in_degree(const xt::xexpression<T> &xindex, const graph_t &g) {
        static_assert(std::is_integral<typename T::value_type>::value,
                      "Vertex indices must have integral value type.");
        auto &index = xindex.derived_cast();
        auto f = xt::flatten(index);
        array_nd<size_t> res = array_nd<size_t>::from_shape({index.size()});
        for (index_t i = 0; i < (index_t) res.size(); ++i) {
            res(i) = in_degree(f(i), g);
        }
        res.reshape(index.shape());
        return res;
    };

    /**
     * Out-degrees of all the given vertices in the given graph
     * @tparam T type of indices (must be integral, preferably index_t)
     * @tparam graph_t
     * @param xindex array of vertex indices
     * @param g
     * @return array of the same size as xindex containing the out-degree of each vertex indicated in xindex
     */
    template<typename T, typename graph_t>
    auto out_degree(const xt::xexpression<T> &xindex, const graph_t &g) {
        static_assert(std::is_integral<typename T::value_type>::value,
                      "Vertex indices must have integral value type.");
        auto &index = xindex.derived_cast();
        auto f = xt::flatten(index);
        array_nd<size_t> res = array_nd<size_t>::from_shape({index.size()});
        for (index_t i = 0; i < (index_t) res.size(); ++i) {
            res(i) = out_degree(f(i), g);
        }
        res.reshape(index.shape());
        return res;
    };

    /**
     * Add all edges given as a pair of arrays (sources, targets) to the graph.
     *
     * @tparam T xexpression type
     * @tparam graph_t Mutable graph type
     * @param xsources Must be a 1d array of integral values
     * @param xtargets Must have the same shape as xsources
     * @param g A mutable graph
     */
    template<typename T, typename graph_t>
    void add_edges(const xt::xexpression<T> &xsources,
                   const xt::xexpression<T> &xtargets,
                   graph_t &g) {
        auto &sources = xsources.derived_cast();
        auto &targets = xtargets.derived_cast();
        hg_assert_1d_array(sources);
        hg_assert_integral_value_type(sources);
        hg_assert_same_shape(sources, targets);

        for (index_t i = 0; i < (index_t)sources.size(); i++)
            g.add_edge(sources(i), targets(i));
    }

    /**
     * Create a new graph as a copy of the given graph
     * @tparam T input graph type
     * @tparam output_graph_type return type (default = ugraph)
     * @param graph
     * @return
     */
    template<typename output_graph_type = ugraph, typename T>
    output_graph_type
    copy_graph(const T &graph) {
        HG_TRACE();
        static_assert(
                std::is_base_of<graph::adjacency_graph_tag, typename graph::graph_traits<T>::traversal_category>::value,
                "Graph must implement adjacency graph concept.");
        static_assert(
                std::is_base_of<graph::vertex_list_graph_tag, typename graph::graph_traits<T>::traversal_category>::value,
                "Graph must implement vertex list graph concept.");

        output_graph_type g(num_vertices(graph));
        auto vertex_it = vertices(graph);
        for (auto vb = vertex_it.first, ve = vertex_it.second; vb != ve; vb++) {
            auto adj_vertex_it = adjacent_vertices(*vb, graph);
            for (auto avb = adj_vertex_it.first, ave = adj_vertex_it.second; avb != ave; avb++) {
                if (*avb > *vb)
                    g.add_edge(*vb, *avb);
            }
        }
        return g;
    };

    /**
     * Create a new graph as a copy of the given graph
     * @tparam output_graph_type return type
     * @param graph
     * @return
     */
    template<typename output_graph_type = ugraph>
    inline
    output_graph_type
    copy_graph(const ugraph &graph) {
        HG_TRACE();
        output_graph_type g(num_vertices(graph));
        auto edge_it = edges(graph);
        for (auto eb = edge_it.first; eb != edge_it.second; eb++) {
            g.add_edge(source(*eb, graph), target(*eb, graph));
        }
        return g;
    };

    /**
     * Given an edge and one of the two extremities of this edge, return the other extremity
     * (if the source is given it returns the target and vice versa).
     * @tparam graph_t
     * @param edge
     * @param vertex
     * @param graph
     * @return
     */
    template<typename graph_t>
    auto other_vertex(const typename graph::graph_traits<graph_t>::edge_descriptor &edge,
                      typename graph::graph_traits<graph_t>::vertex_descriptor vertex,
                      const graph_t &graph) {
        return (source(edge, graph) == vertex) ? target(edge, graph) : source(edge, graph);
    }


    /**
     * Create an adjacency matrix from an undirected edge-weighted graph (the result is thus symmetric).
     *
     * As the given graph is not necessarily complete, non-existing edges will receive the value `non_edge_value` in
     * the adjacency matrix.
     *
     * @tparam undirected_graph
     * @tparam T
     * @tparam value_type
     * @param graph Input undirected graph
     * @param xedge_weights Input edge-weights
     * @param non_edge_value Value used to represent non existing edges
     * @return A 2d square array
     */
    template<typename undirected_graph, typename T, typename value_type = typename T::value_type>
    auto undirected_graph_2_adjacency_matrix(const undirected_graph &graph,
                                             const xt::xexpression<T> &xedge_weights,
                                             const value_type &non_edge_value = 0) {
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        array_2d<value_type> a = array_2d<value_type>::from_shape({num_vertices(graph), num_vertices(graph)});

        a.fill(non_edge_value);

        for (const auto &e: edge_iterator(graph)) {
            a(source(e, graph), target(e, graph)) = a(target(e, graph), source(e, graph)) = edge_weights(
                    index(e, graph));
        }

        return a;
    }

    /**
     * Creates an undirected edge-weighted graph from an adjacency matrix.
     *
     * Adjacency matrix entries which are equal to `non_edge_value` are not considered to be part of the graph.
     *
     * @tparam T
     * @tparam value_type
     * @param xadjacency_matrix Input adjacency matrix
     * @param non_edge_value Value used to represent non existing edges
     * @return a pair of types (ugraph, array_1d) representing the graph and its edge-weights
     */
    template<typename T, typename value_type = typename T::value_type>
    auto adjacency_matrix_2_undirected_graph(const xt::xexpression<T> &xadjacency_matrix,
                                             const value_type &non_edge_value = 0) {
        auto &adjacency_matrix = xadjacency_matrix.derived_cast();
        hg_assert(adjacency_matrix.dimension() == 2, "Adjacency matrix must be a 2d array.");
        hg_assert(adjacency_matrix.shape()[0] == adjacency_matrix.shape()[1], "Adjacency matrix must be square.");

        index_t n_vertices = adjacency_matrix.shape()[0];
        ugraph g(n_vertices);

        index_t n_edges = 0;
        for (index_t i = 0; i < n_vertices; i++) {
            for (index_t j = i; j < n_vertices; j++) {
                if (adjacency_matrix(i, j) != non_edge_value) {
                    n_edges++;
                }
            }
        }
        array_1d<value_type> edge_weights = xt::empty<value_type>({n_edges});
        index_t n = 0;
        for (index_t i = 0; i < n_vertices; i++) {
            for (index_t j = i; j < n_vertices; j++) {
                if (adjacency_matrix(i, j) != non_edge_value) {
                    g.add_edge(i, j);
                    edge_weights(n++) = adjacency_matrix(i, j);
                }
            }
        }
        return std::make_pair(std::move(g), std::move(edge_weights));
    }
};
