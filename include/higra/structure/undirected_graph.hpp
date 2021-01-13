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

#include <functional>
#include "details/graph_concepts.hpp"
#include "details/indexed_edge.hpp"
#include "higra/structure/details/iterators.hpp"
#include <vector>
#include <list>
#include <unordered_set>

namespace hg {
/**
  * Undirected graph with in and out edge lists
  */

    namespace undirected_graph_internal {


        struct undirected_graph_traversal_category :
                virtual public graph::incidence_graph_tag,
                virtual public graph::bidirectional_graph_tag,
                virtual public graph::adjacency_graph_tag,
                virtual public graph::vertex_list_graph_tag,
                virtual public graph::edge_list_graph_tag {
        };

        // like boost adjacency list
        struct vecS {
        };
        struct hash_setS {
        };

        template<typename Selector, typename ValueType>
        struct container_gen {
        };

        template<typename ValueType>
        struct container_gen<vecS, ValueType> {
            typedef std::vector<ValueType> type;
        };

        template<typename ValueType>
        struct container_gen<hash_setS, ValueType> {
            typedef std::unordered_set<ValueType> type;
        };


        template<typename ValueType>
        void remove_from_container(std::vector<ValueType> &c, ValueType v) {
            auto position = std::find(c.begin(), c.end(), v);
            if (position != c.end())
                c.erase(position);
        }

        template<typename ValueType>
        void remove_from_container(std::unordered_set<ValueType> &c, ValueType v) {
            c.erase(v);
        }

        template<typename ValueType>
        void add_to_container(std::vector<ValueType> &c, ValueType v) {
            c.push_back(v);
        }

        template<typename ValueType>
        void add_to_container(std::unordered_set<ValueType> &c, ValueType v) {
            c.insert(v);
        }

        template<typename edgeS=vecS>
        struct undirected_graph {

            // Graph associated types
            using vertex_descriptor = index_t;
            using edge_index_t = index_t;
            using edge_descriptor = indexed_edge<vertex_descriptor, edge_index_t>;
            using out_edge_t = std::pair<vertex_descriptor, vertex_descriptor>; // (edge_index, adjacent vertex)
            using directed_category = graph::undirected_tag;
            using edge_parallel_category = graph::allow_parallel_edge_tag;
            using traversal_category = undirected_graph_traversal_category;

            // VertexListGraph associated types
            using vertex_iterator = counting_iterator<vertex_descriptor>;
            using vertices_size_type = size_t;

            // custom edge index iterators
            using out_edge_container_type = typename container_gen<edgeS, edge_index_t>::type;
            using out_edge_index_iterator = typename out_edge_container_type::const_iterator;
            using in_edge_index_iterator = out_edge_index_iterator;

            // EdgeListGraph associated types
            using edges_size_type = size_t;
            using edge_iterator = std::vector<edge_descriptor>::const_iterator;

            // IncidenceGraph associated types
            using out_iterator_transform_function = std::function<edge_descriptor(edge_index_t)>;
            using out_edge_iterator = transform_forward_iterator<out_iterator_transform_function,
                    out_edge_index_iterator,
                    edge_descriptor>;
            using degree_size_type = size_t;

            //BidirectionalGraph associated types
            using in_edge_iterator = out_edge_iterator;

            //AdjacencyGraph associated types
            using adjacent_iterator_transform_function = std::function<vertex_descriptor(edge_index_t)>;
            using adjacency_iterator = transform_forward_iterator<adjacent_iterator_transform_function,
                    out_edge_index_iterator,
                    vertex_descriptor>;


            undirected_graph(const size_t num_vertices = 0,
                             const size_t reserved_edges = 0,
                             const size_t reserved_edge_per_vertex = 0) :
                    _num_vertices(num_vertices), out_edges(num_vertices) {
                if (reserved_edges > 0) {
                    edges.reserve(reserved_edges);
                }
                if (reserved_edge_per_vertex > 0) {
                    for (index_t i = 0; i < (index_t) num_vertices; ++i) {
                        out_edges[i].reserve(reserved_edge_per_vertex);
                    }
                }
            };

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
                out_edges.emplace_back();
                return tmp;
            }

            void add_vertices(size_t num) {
                if (num > 0) {
                    _num_vertices += num;
                    out_edges.resize(_num_vertices);
                }
            }

            void remove_edge(edge_index_t ei) {
                auto &source = edges[ei].source;
                auto &target = edges[ei].target;
                remove_from_container(out_edges[source], ei);
                if (source != target)
                    remove_from_container(out_edges[target], ei);
                edges[ei].source = invalid_index;
                edges[ei].target = invalid_index;
            }

            void set_edge(edge_index_t ei, vertex_descriptor v1, vertex_descriptor v2) {
                if (v1 > v2) {
                    std::swap(v1, v2);
                }
                // TODO optimize cases !
                remove_edge(ei);

                add_to_container(out_edges[v1], ei);
                if (v1 != v2)
                    add_to_container(out_edges[v2], ei);
                edges[ei].source = v1;
                edges[ei].target = v2;
            }

            const auto &edge_from_index(index_t v) const {
                return edges[v];
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

            const edge_descriptor &add_edge(vertex_descriptor v1, vertex_descriptor v2) {
                if (v1 > v2) {
                    std::swap(v1, v2);
                }

                vertex_descriptor index = edges.size();
                edges.emplace_back(v1, v2, index);

                add_to_container(out_edges[v1], index);
                if (v1 != v2)
                    add_to_container(out_edges[v2], index);

                return edges[index];
            }

            template<typename T>
            edge_descriptor add_edge(const T &e) {
                return add_edge(e.first, e.second);
            }

            auto sources() const {
                return HG_ADAPT_STRUCT_ARRAY(edges.data(), source, num_edges());
            }

            auto targets() const {
                return HG_ADAPT_STRUCT_ARRAY(edges.data(), target, num_edges());
            }

        private:

            size_t _num_vertices;
            std::vector<edge_descriptor> edges;
            std::vector<out_edge_container_type> out_edges; // same as in_edges...

        };

    }

    using vecS = undirected_graph_internal::vecS;
    using hash_setS = undirected_graph_internal::hash_setS;

    template<typename storage_type = vecS>
    using undirected_graph = undirected_graph_internal::undirected_graph<storage_type>;

    using ugraph = undirected_graph_internal::undirected_graph<>;


    namespace graph {
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

            using edge_index = typename G::edge_index_t;
        };
    }

    template<typename T>
    const auto &edge_from_index(const typename undirected_graph<T>::vertex_descriptor v, const undirected_graph<T> &g) {
        return g.edge_from_index(v);
    }

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
    void add_vertices(size_t num, hg::undirected_graph<T> &g) {
        g.add_vertices(num);
    }

    template<typename T>
    typename hg::undirected_graph<T>::edge_descriptor add_edge(typename hg::undirected_graph<T>::vertex_descriptor v1,
                                                               typename hg::undirected_graph<T>::vertex_descriptor v2,
                                                               hg::undirected_graph<T> &g) {
        return g.add_edge(v1, v2);
    }

    template<typename T>
    void remove_edge(typename hg::undirected_graph<T>::edge_index_t ei, hg::undirected_graph<T> &g) {
        g.remove_edge(ei);
    }

    template<typename T>
    void set_edge(typename hg::undirected_graph<T>::edge_index_t ei,
                  typename hg::undirected_graph<T>::vertex_descriptor v1,
                  typename hg::undirected_graph<T>::vertex_descriptor v2,
                  hg::undirected_graph<T> &g) {
        g.set_edge(ei, v1, v2);
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
    std::pair<typename hg::undirected_graph<T>::out_edge_iterator, typename hg::undirected_graph<T>::out_edge_iterator>
    out_edges(typename hg::undirected_graph<T>::vertex_descriptor v, const hg::undirected_graph<T> &g) {
        auto fun = [v, &g](const typename hg::undirected_graph<T>::edge_index_t &oei) {
            const auto &oe = g.edge_from_index(oei);
            return typename hg::undirected_graph<T>::edge_descriptor(v, (v == oe.source) ? oe.target : oe.source,
                                                                     oe.index);
        };
        using it = typename hg::undirected_graph<T>::out_edge_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }

    template<typename T>
    std::pair<typename hg::undirected_graph<T>::out_edge_iterator, typename hg::undirected_graph<T>::out_edge_iterator>
    in_edges(typename hg::undirected_graph<T>::vertex_descriptor v, const hg::undirected_graph<T> &g) {
        auto fun = [v, &g](const typename hg::undirected_graph<T>::edge_index_t &oei) {
            const auto &oe = g.edge_from_index(oei);
            return typename hg::undirected_graph<T>::edge_descriptor((v == oe.source) ? oe.target : oe.source, v,
                                                                     oe.index);
        };
        using it = typename hg::undirected_graph<T>::out_edge_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }

    template<typename T>
    std::pair<typename hg::undirected_graph<T>::adjacency_iterator, typename hg::undirected_graph<T>::adjacency_iterator>
    adjacent_vertices(typename hg::undirected_graph<T>::vertex_descriptor v, const hg::undirected_graph<T> &g) {
        auto fun = [v, &g](const typename hg::undirected_graph<T>::edge_index_t &oei) {
            const auto &oe = g.edge_from_index(oei);
            return (v == oe.source) ? oe.target : oe.source;
        };
        using it = typename hg::undirected_graph<T>::adjacency_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }

}

#ifdef HG_USE_BOOST_GRAPH
namespace boost {

    using hg::graph_traits;
    using hg::out_edges;
    using hg::in_edges;
    using hg::in_degree;
    using hg::out_degree;
    using hg::degree;
    using hg::vertices;
    using hg::edges;
    using hg::add_vertex;
    using hg::add_edge;
    using hg::num_vertices;
    using hg::num_edges;
    using hg::adjacent_vertices;
}
#endif


