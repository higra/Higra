//
// Created by user on 3/9/18.
//

#pragma once

#include "utils.hpp"
#include "structure/undirected_graph.hpp"
#include "structure/regular_graph.hpp"
#include "structure/tree_graph.hpp"

namespace hg {

    template<typename graph_t>
    auto source(
            const std::pair<typename graph::graph_traits<graph_t>::vertex_descriptor,
                    typename graph::graph_traits<graph_t>::vertex_descriptor> &e,
            const graph_t &) {
        return e.first;
    }

    template<typename graph_t>
    auto target(
            const std::pair<typename graph::graph_traits<graph_t>::vertex_descriptor,
                    typename graph::graph_traits<graph_t>::vertex_descriptor> &e,
            const graph_t &) {
        return e.second;
    }

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

    template<typename graph_t>
    auto vertex_iterator(const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::vertex_iterator;
        return iterator_wrapper<it_t>(vertices(g));
    }

    template<typename graph_t>
    auto edge_iterator(const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::edge_iterator;
        return iterator_wrapper<it_t>(edges(g));
    }

    template<typename graph_t>
    auto out_edge_iterator(typename graph::graph_traits<graph_t>::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::out_edge_iterator;
        return iterator_wrapper<it_t>(out_edges(v, g));
    }

    template<typename graph_t>
    auto in_edge_iterator(typename graph::graph_traits<graph_t>::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::in_edge_iterator;
        return iterator_wrapper<it_t>(in_edges(v, g));
    }

    template<typename graph_t>
    auto adjacent_vertex_iterator(typename graph::graph_traits<graph_t>::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph::graph_traits<graph_t>::adjacency_iterator;
        return iterator_wrapper<it_t>(adjacent_vertices(v, g));
    }

    template<typename graph_t>
    auto edge_index_iterator(const graph_t &g) {
        using it_t = typename graph_t::edge_index_iterator;
        return iterator_wrapper<it_t>(hg::edge_indexes(g));
    }

    template<typename graph_t>
    auto out_edge_index_iterator(typename graph_t::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph_t::out_edge_index_iterator;
        return iterator_wrapper<it_t>(out_edge_indexes(v, g));
    }

    template<typename graph_t>
    auto in_edge_index_iterator(typename graph_t::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph_t::in_edge_index_iterator;
        return iterator_wrapper<it_t>(in_edge_indexes(v, g));
    }


    template<typename graph_t>
    auto children_iterator(typename graph_t::vertex_descriptor v, const graph_t &g) {
        using it_t = typename graph_t::children_iterator;
        return iterator_wrapper<it_t>(children(v, g));
    }

    template<typename T, typename graph_t>
    auto degree(const xt::xexpression<T> &xindex, const graph_t &g) {
        static_assert(std::is_integral<typename T::value_type>::value,
                      "Vertex indices must have integral value type.");
        auto &index = xindex.derived_cast();
        auto f = xt::flatten(index);
        array_nd<std::size_t> res = xt::zeros<std::size_t>({index.size()});
        for (std::size_t i = 0; i < res.size(); ++i) {
            res(i) = degree(f(i), g);
        }
        res.reshape(index.shape());
        return res;
    };

    template<typename T, typename graph_t>
    auto in_degree(const xt::xexpression<T> &xindex, const graph_t &g) {
        static_assert(std::is_integral<typename T::value_type>::value,
                      "Vertex indices must have integral value type.");
        auto &index = xindex.derived_cast();
        auto f = xt::flatten(index);
        array_nd<std::size_t> res = xt::zeros<std::size_t>({index.size()});
        for (std::size_t i = 0; i < res.size(); ++i) {
            res(i) = in_degree(f(i), g);
        }
        res.reshape(index.shape());
        return res;
    };

    template<typename T, typename graph_t>
    auto out_degree(const xt::xexpression<T> &xindex, const graph_t &g) {
        static_assert(std::is_integral<typename T::value_type>::value,
                      "Vertex indices must have integral value type.");
        auto &index = xindex.derived_cast();
        auto f = xt::flatten(index);
        array_nd<std::size_t> res = xt::zeros<std::size_t>({index.size()});
        for (std::size_t i = 0; i < res.size(); ++i) {
            res(i) = out_degree(f(i), g);
        }
        res.reshape(index.shape());
        return res;
    };




    template<typename T>
    ugraph make_ugraph(const T &graph) {
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


    template<>
    inline
    ugraph make_ugraph(const ugraph &graph) {
        ugraph g(num_vertices(graph));
        auto edge_it = edges(graph);
        for (auto eb = edge_it.first, ee = edge_it.second; eb != ee; eb++) {
            g.add_edge((*eb).first, (*eb).second);
        }
        return g;
    };

    template<typename graph_t>
    auto other_vertex(const typename graph::graph_traits<graph_t>::edge_descriptor &edge,
                      typename graph::graph_traits<graph_t>::vertex_descriptor vertex,
                      const graph_t &graph) {
        return (source(edge, graph) == vertex) ? target(edge, graph) : source(edge, graph);
    }
};
