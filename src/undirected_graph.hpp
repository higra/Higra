//
// Created by user on 3/12/18.
//

#pragma once

#include <functional>
#include <boost/graph/graph_concepts.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <vector>
#include <boost/graph/adjacency_list.hpp>

namespace hg {
/**
  * Undirected graph with in and out edge lists
  */
    /*using undirected_graph = boost::adjacency_list<
            boost::vecS, // OutEdgeList
            boost::vecS, // VertexList
            boost::undirectedS, // directed
            boost::no_property, // vertex property
            boost::no_property, // edge property
            boost::no_property, // graph property
            boost::listS>; // edge list*/


    namespace undirected_graph_internal {


        struct undirected_graph_traversal_category :
                virtual public boost::incidence_graph_tag,
                virtual public boost::bidirectional_graph_tag,
                virtual public boost::adjacency_graph_tag,
                virtual public boost::vertex_list_graph_tag {
        };


        // like boost adjacency list
        struct vecS {
        };
        struct listS {
        };

        template<class Selector, class ValueType>
        struct container_gen {
        };

        template<class ValueType>
        struct container_gen<listS, ValueType> {
            typedef std::list<ValueType> type;
        };


        template<class ValueType>
        struct container_gen<vecS, ValueType> {
            typedef std::vector<ValueType> type;
        };


        template<typename edgeS=vecS>
        struct undirected_graph {

            // Graph associated types
            using vertex_descriptor = std::size_t;
            using edge_descriptor = std::pair<vertex_descriptor, vertex_descriptor>;
            using out_edge_t = std::pair<std::size_t, vertex_descriptor>; // (edge_index, adjacent vertex)
            using directed_category = boost::undirected_tag;
            using edge_parallel_category = boost::allow_parallel_edge_tag;
            using traversal_category = undirected_graph_traversal_category;

            // VertexListGraph associated types
            using vertex_iterator = boost::counting_iterator<vertex_descriptor>;
            using vertices_size_type = std::size_t;

            // EdgeListGraph associated types
            using edges_size_type = std::size_t;
            using edge_iterator = typename container_gen<edgeS, edge_descriptor>::type::const_iterator;


            // IncidenceGraph associated types
            using out_iterator_transform_function = std::function<edge_descriptor(out_edge_t)>;
            using out_edge_iterator = boost::transform_iterator<out_iterator_transform_function,
                    typename container_gen<edgeS, out_edge_t>::type::const_iterator>;
            using degree_size_type = std::size_t;

            //BidirectionalGraph associated types
            using in_edge_iterator = out_edge_iterator;

            //AdjacencyGraph associated types
            using adjacent_iterator_transform_function = std::function<vertex_descriptor(out_edge_t)>;
            using adjacency_iterator = boost::transform_iterator<adjacent_iterator_transform_function,
                    typename container_gen<edgeS, out_edge_t>::type::const_iterator>;


            undirected_graph(std::size_t num_vertices = 0) : _num_vertices(num_vertices), out_edges(num_vertices) {};

            vertices_size_type num_vertices() const {
                return _num_vertices;
            }

            edges_size_type num_edges() const {
                return edges.size();
            }

            degree_size_type degree(vertex_descriptor v) const {
                return out_edges[v].size();
            }

            vertex_descriptor add_vertex() {
                auto tmp = _num_vertices;
                _num_vertices++;
                out_edges.push_back({});
                return tmp;
            }

            auto edges_cbegin() const {
                return edges.cbegin();
            }

            auto edges_cend() const {
                return edges.cend();
            }

            auto out_edges_cbegin(vertex_descriptor v) const {
                return out_edges[v].cbegin();
            }

            auto out_edges_cend(vertex_descriptor v) const {
                return out_edges[v].cend();
            }

            edge_descriptor add_edge(vertex_descriptor v1, vertex_descriptor v2) {
                if (v1 > v2) {
                    std::swap(v1, v2);
                }
                edges.push_back(std::make_pair(v1, v2));
                auto index = edges.size() - 1;
                out_edges[v1].push_back(std::make_pair(index, v2));
                out_edges[v2].push_back(std::make_pair(index, v1));

                return std::make_pair(v1, v2);
            }


        private:

            std::size_t _num_vertices;
            typename container_gen<edgeS, edge_descriptor>::type edges;
            std::vector<typename container_gen<edgeS, out_edge_t>::type> out_edges; // same as in_edges...

        };

    }
    template<typename storage_type = undirected_graph_internal::vecS>
    using undirected_graph = undirected_graph_internal::undirected_graph<storage_type>;

    using ugraph = undirected_graph_internal::undirected_graph<>;
}


namespace boost {

    template<typename T>
    struct graph_traits<hg::undirected_graph<T>> {
        using G = hg::undirected_graph<T>;

        using vertex_descriptor = typename G::vertex_descriptor;
        using edge_descriptor = typename G::edge_descriptor;
        using edge_iterator = typename G::edge_iterator;
        using out_edge_iterator = typename G::out_edge_iterator;

        using directed_category = typename G::directed_category;
        using edge_parallel_category = typename G::edge_parallel_category;
        using traversal_category = typename G::traversal_category;

        using degree_size_type = typename G::degree_size_type;

        using in_edge_iterator = typename G::in_edge_iterator;
        using vertex_iterator = typename G::vertex_iterator;
        using vertices_size_type = typename G::vertices_size_type;
        using edges_size_type = typename G::edges_size_type;
        using adjacency_iterator = typename G::adjacency_iterator;
    };

    template<typename T>
    typename hg::undirected_graph<T>::vertices_size_type num_vertices(const hg::undirected_graph<T> &g) {
        return g.num_vertices();
    }

    template<typename T>
    typename hg::undirected_graph<T>::edges_size_type num_edges(const hg::undirected_graph<T> &g) {
        return g.num_edges();
    }

    template<typename T>
    typename hg::undirected_graph<T>::degree_size_type degree(typename hg::undirected_graph<T>::vertex_descriptor v,
                                                              const hg::undirected_graph<T> &g) {
        return g.degree(v);
    }

    template<typename T>
    typename hg::undirected_graph<T>::degree_size_type in_degree(typename hg::undirected_graph<T>::vertex_descriptor v,
                                                                 const hg::undirected_graph<T> &g) {
        return g.degree(v);
    }


    template<typename T>
    typename hg::undirected_graph<T>::degree_size_type out_degree(typename hg::undirected_graph<T>::vertex_descriptor v,
                                                                  const hg::undirected_graph<T> &g) {
        return g.degree(v);
    }

    template<typename T>
    typename hg::undirected_graph<T>::vertex_descriptor add_vertex(hg::undirected_graph<T> &g) {
        return g.add_vertex();
    }

    template<typename T>
    typename hg::undirected_graph<T>::edge_descriptor add_edge(typename hg::undirected_graph<T>::vertex_descriptor v1,
                                                               typename hg::undirected_graph<T>::vertex_descriptor v2,
                                                               hg::undirected_graph<T> &g) {
        return g.add_edge(v1, v2);
    }

    template<typename T>
    std::pair<typename hg::undirected_graph<T>::vertex_iterator, typename hg::undirected_graph<T>::vertex_iterator>
    vertices(const hg::undirected_graph<T> &g) {
        using vertex_iterator = typename hg::undirected_graph<T>::vertex_iterator;
        return std::make_pair(
                vertex_iterator(0),                 // The first iterator position
                vertex_iterator(num_vertices(g))); // The last iterator position
    }

    template<typename T>
    std::pair<typename hg::undirected_graph<T>::edge_iterator, typename hg::undirected_graph<T>::edge_iterator>
    edges(const hg::undirected_graph<T> &g) {
        return std::make_pair(
                g.edges_cbegin(),                 // The first iterator position
                g.edges_cend()); // The last iterator position
    }

    template<typename T>
    typename hg::undirected_graph<T>::vertex_descriptor
    source(
            typename hg::undirected_graph<T>::edge_descriptor &e,
            hg::undirected_graph<T> &) {
        return e.first;
    }

    template<typename T>
    typename hg::undirected_graph<T>::vertex_descriptor
    target(
            typename hg::undirected_graph<T>::edge_descriptor &e,
            hg::undirected_graph<T> &) {
        return e.second;
    }

    template<typename T>
    std::pair<typename hg::undirected_graph<T>::out_edge_iterator, typename hg::undirected_graph<T>::out_edge_iterator>
    out_edges(typename hg::undirected_graph<T>::vertex_descriptor v, hg::undirected_graph<T> &g) {
        auto fun = [v](typename hg::undirected_graph<T>::out_edge_t oe) {
            return std::make_pair(v, oe.second);
        };
        using it = typename hg::undirected_graph<T>::out_edge_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }

    template<typename T>
    std::pair<typename hg::undirected_graph<T>::out_edge_iterator, typename hg::undirected_graph<T>::out_edge_iterator>
    in_edges(typename hg::undirected_graph<T>::vertex_descriptor v, hg::undirected_graph<T> &g) {
        auto fun = [v](typename hg::undirected_graph<T>::out_edge_t oe) {
            return std::make_pair(oe.second, v);
        };
        using it = typename hg::undirected_graph<T>::out_edge_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }

    template<typename T>
    std::pair<typename hg::undirected_graph<T>::adjacency_iterator, typename hg::undirected_graph<T>::adjacency_iterator>
    adjacent_vertices(typename hg::undirected_graph<T>::vertex_descriptor v, hg::undirected_graph<T> &g) {
        auto fun = [v](typename hg::undirected_graph<T>::out_edge_t oe) {
            return oe.second;
        };
        using it = typename hg::undirected_graph<T>::adjacency_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }

/*
    template<typename embedding_t>
    std::pair<typename hg::regular_graph<embedding_t>::out_edge_iterator, typename hg::regular_graph<embedding_t>::out_edge_iterator>
    in_edges(typename hg::regular_graph<embedding_t>::vertex_descriptor u, hg::regular_graph<embedding_t> &g) {
        return std::make_pair(
                hg::regular_graph_out_edge_iterator<embedding_t>(
                        hg::regular_graph_adjacent_vertex_iterator<embedding_t>(
                                u,
                                g.embedding,
                                g.neighbours.begin(),
                                g.neighbours.end()),
                        [u](typename hg::regular_graph<embedding_t>::vertex_descriptor v) {
                            return std::make_pair(v, u);
                        }),
                hg::regular_graph_out_edge_iterator<embedding_t>(
                        hg::regular_graph_adjacent_vertex_iterator<embedding_t>(
                                u,
                                g.embedding,
                                g.neighbours.end(),
                                g.neighbours.end()),
                        [u](typename hg::regular_graph<embedding_t>::vertex_descriptor v) {
                            return std::make_pair(v, u);
                        })
        );
    }









    template<typename embedding_t>
    std::pair<typename hg::regular_graph<embedding_t>::vertex_iterator, typename hg::regular_graph<embedding_t>::vertex_iterator>
    vertices(const hg::regular_graph<embedding_t> &g) {
        using vertex_iterator = typename hg::regular_graph<embedding_t>::vertex_iterator;
        return std::pair<vertex_iterator, vertex_iterator>(
                vertex_iterator(0),                 // The first iterator position
                vertex_iterator(num_vertices(g))); // The last iterator position
    }

    template<typename embedding_t>
    std::pair<typename hg::regular_graph<embedding_t>::adjacency_iterator, typename hg::regular_graph<embedding_t>::adjacency_iterator>
    adjacent_vertices(typename hg::regular_graph<embedding_t>::vertex_descriptor u, hg::regular_graph<embedding_t> &g) {
        return std::make_pair<typename hg::regular_graph<embedding_t>::adjacency_iterator, typename hg::regular_graph<embedding_t>::adjacency_iterator>(
                hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g.embedding, g.neighbours.begin(),
                                                                        g.neighbours.end()),
                hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g.embedding, g.neighbours.end(),
                                                                        g.neighbours.end()));
    };*/
}