//
// Created by user on 3/9/18.
//

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
     * Range over the indices of all edges of the given graph
     * @tparam graph_t
     * @param g
     * @return
     */
    template<typename graph_t>
    auto edge_index_iterator(const graph_t &g) {
        using it_t = typename graph_t::edge_index_iterator;
        return iterator_wrapper<it_t>(hg::edge_indexes(g));
    }

    /**
     * Range over all the edge indices whose source is the given vertex in the given graph
     * @tparam graph_t
     * @param v
     * @param g
     * @return
     */
    template<typename graph_t>
    auto out_edge_index_iterator(typename graph_t::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph_t::out_edge_index_iterator;
        return iterator_wrapper<it_t>(out_edge_indexes(v, g));
    }

    /**
     * Range over all the edge indices whose target is the given vertex in the given graph
     * @tparam graph_t
     * @param v
     * @param g
     * @return
     */
    template<typename graph_t>
    auto in_edge_index_iterator(typename graph_t::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph_t::in_edge_index_iterator;
        return iterator_wrapper<it_t>(in_edge_indexes(v, g));
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
        for (index_t i = 0; i < (index_t)res.size(); ++i) {
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
        for (index_t i = 0; i < (index_t)res.size(); ++i) {
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
        for (index_t i = 0; i < (index_t)res.size(); ++i) {
            res(i) = out_degree(f(i), g);
        }
        res.reshape(index.shape());
        return res;
    };

    /**
     * Create a new undirected graph (ugraph) as a copy of the given graph
     * @tparam T
     * @param graph
     * @return
     */
    template<typename T>
    ugraph make_ugraph(const T &graph) {
        HG_TRACE();
        static_assert(
                std::is_base_of<graph::adjacency_graph_tag, typename graph::graph_traits<T>::traversal_category>::value,
                "Graph must implement adjacency graph concept.");
        static_assert(
                std::is_base_of<graph::vertex_list_graph_tag, typename graph::graph_traits<T>::traversal_category>::value,
                "Graph must implement vertex list graph concept.");

        ugraph g(num_vertices(graph));
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
     * Create a new undirected graph (ugraph) as a copy of the given graph
     * @tparam T
     * @param graph
     * @return
     */
    template<>
    inline
    ugraph make_ugraph(const ugraph &graph) {
        HG_TRACE();
        ugraph g(num_vertices(graph));
        auto edge_it = edges(graph);
        for (auto eb = edge_it.first, ee = edge_it.second; eb != ee; eb++) {
            g.add_edge((*eb).first, (*eb).second);
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
};
