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

#include "details/graph_concepts.hpp"
#include "higra/structure/details/iterators.hpp"
#include <functional>
#include <vector>
#include <utility>

#include "xtensor/xarray.hpp"
#include "embedding.hpp"

namespace hg {


    namespace regular_graph_internal {

        template<typename value_t, int dim>
        using point_list_t = std::vector<point<value_t, dim>>;

        template<typename value_t, int dim>
        using point_list_iterator_t = typename point_list_t<value_t, dim>::const_iterator;

        //forward declaration
        template<typename embedding_t>
        struct regular_graph_adjacent_vertex_iterator;

        struct regular_graph_traversal_category :
                virtual public graph::incidence_graph_tag,
                virtual public graph::bidirectional_graph_tag,
                virtual public graph::adjacency_graph_tag,
                virtual public graph::vertex_list_graph_tag {
        };

        template<typename embedding_t>
        class regular_graph {

        public:
            using self_type = regular_graph<embedding_t>;
            // Graph associated types
            using vertex_descriptor = index_t;
            using directed_category = graph::undirected_tag;
            using edge_parallel_category = graph::disallow_parallel_edge_tag;
            using traversal_category = regular_graph_traversal_category;

            // VertexListGraph associated types
            using vertex_iterator = counting_iterator<vertex_descriptor>;
            using vertices_size_type = size_t;

            //AdjacencyGraph associated types
            using adjacency_iterator = regular_graph_adjacent_vertex_iterator<embedding_t>;

            // IncidenceGraph associated types
            using edge_descriptor = std::pair<vertex_descriptor, vertex_descriptor>;
            using iterator_transform_function = std::function<edge_descriptor(vertex_descriptor)>;

            using out_edge_iterator = transform_forward_iterator<iterator_transform_function,
                    adjacency_iterator,
                    edge_descriptor>;
            using degree_size_type = size_t;

            //BidirectionalGraph associated types
            using in_edge_iterator = out_edge_iterator;

            using point_type = typename embedding_t::point_type;

            vertices_size_type num_vertices() const {
                return m_embedding.size();
            }

            const auto &embedding() const {
                return m_embedding;
            }

            const auto &neighbours() const {
                return m_neighbours;
            }

            regular_graph(embedding_t _embedding = {}, point_list_t<index_t, embedding_t::_dim> _neighbours = {})
                    : m_embedding(_embedding), m_neighbours(_neighbours) {
                init_safe_are();
            }

            ~regular_graph() = default;

            regular_graph(const self_type &other) = default;

            regular_graph(self_type &&other) = default;

            self_type &operator=(const self_type &) = default;

            self_type &operator=(self_type &&) = default;

            degree_size_type out_degree(const vertex_descriptor v) const;

        private:

            void init_safe_are() {
                // determine the largest sub domain such that every neighbours of a vertex is present in the graph domain
                // for a vertex in this domain, we can just use the relative linear index to find its neighbours
                m_safe_lower_bound.fill(std::numeric_limits<index_t>::max());
                m_safe_upper_bound.fill(std::numeric_limits<index_t>::lowest());
                for (const auto &n: m_neighbours) {
                    m_safe_lower_bound = xt::minimum(m_safe_lower_bound, n);
                    m_safe_upper_bound = xt::maximum(m_safe_upper_bound, n);

                }
                bool safe_area_non_empty = true;
                for (index_t i = 0; i < embedding_t::_dim; ++i) {
                    m_safe_lower_bound(i) = (std::max)(-m_safe_lower_bound(i), (index_t) 0);
                    m_safe_upper_bound(i) = (std::min)(m_embedding.shape()(i) - 1 - m_safe_upper_bound(i),
                                                       m_embedding.shape()(i) - 1);
                    if (m_safe_lower_bound(i) > m_safe_upper_bound(i)) {
                        safe_area_non_empty = false;
                    }
                }

                if (safe_area_non_empty) {
                    m_relative_neighbours.reserve(m_neighbours.size());
                    point<index_t, embedding_t::_dim> ref = m_safe_lower_bound;
                    index_t ref_index = m_embedding.grid2lin(ref);
                    for (const auto &n: m_neighbours) {
                        index_t n_index = m_embedding.grid2lin(xt::eval(ref + n));
                        m_relative_neighbours.push_back(n_index - ref_index);
                    }
                }
            }

            bool is_in_safe_area(const point<index_t, embedding_t::_dim> &point) const {
                for (index_t i = 0; i < embedding_t::_dim; ++i) {
                    if (point(i) < m_safe_lower_bound(i) || point(i) > m_safe_upper_bound(i)) {
                        return false;
                    }
                }
                return true;
            }

            embedding_t m_embedding;
            point_list_t<index_t, embedding_t::_dim> m_neighbours;
            point<index_t, embedding_t::_dim> m_safe_lower_bound;
            point<index_t, embedding_t::_dim> m_safe_upper_bound;
            std::vector<index_t> m_relative_neighbours;

            friend class regular_graph_adjacent_vertex_iterator<embedding_t>;
        };

        // Iterator
        template<typename embedding_t>
        struct regular_graph_adjacent_vertex_iterator :
                public forward_iterator_facade<regular_graph_adjacent_vertex_iterator<embedding_t>,
                        typename regular_graph<embedding_t>::vertex_descriptor,
                        typename regular_graph<embedding_t>::vertex_descriptor> {
        public:
            using self_type = regular_graph_adjacent_vertex_iterator<embedding_t>;
            using graph_t = regular_graph<embedding_t>;
            using graph_vertex_t = typename graph_t::vertex_descriptor;


            regular_graph_adjacent_vertex_iterator() {}

            regular_graph_adjacent_vertex_iterator(graph_vertex_t _source,
                                                   const regular_graph<embedding_t> &_graph,
                                                   bool end = false
            ) : source(_source), graph(_graph) {

                source_coordinates = graph.m_embedding.lin2grid(source);
                safe_area = graph.is_in_safe_area(source_coordinates);
                num_elem = graph.m_neighbours.size();
                current_element = (end) ? graph.m_neighbours.size() : 0;
                if (current_element != num_elem) {
                    if (safe_area) {
                        neighbour = source + graph.m_relative_neighbours[current_element];
                    } else {
                        point_type neighbourc = graph.m_neighbours[current_element] + source_coordinates;

                        if (!graph.m_embedding.contains(neighbourc)) {
                            increment();
                        } else {
                            neighbour = graph.m_embedding.grid2lin(neighbourc);
                        }
                    }
                }
            }


            using point_type = typename embedding_t::point_type;

            void increment() {
                if (safe_area) {
                    ++current_element;
                    neighbour = source + graph.m_relative_neighbours[current_element];
                } else {
                    bool flag;
                    point_type neighbourc;
                    do {
                        ++current_element;
                        if (current_element != num_elem) {
                            neighbourc = graph.m_neighbours[current_element] + source_coordinates;
                            flag = graph.m_embedding.contains(neighbourc);
                        } else {
                            flag = true;
                        }
                    } while (!flag);
                    if (current_element != num_elem) {
                        neighbour = graph.m_embedding.grid2lin(neighbourc);
                    }
                }

            }

            bool equal(regular_graph_adjacent_vertex_iterator const &other) const {
                return current_element == other.current_element;
            }


            graph_vertex_t dereference() const {
                return neighbour;
            }

        private:
            graph_vertex_t source;
            graph_vertex_t neighbour;
            point_type source_coordinates;
            const regular_graph<embedding_t> &graph;
            int current_element;
            int num_elem;
            bool safe_area;
        };

    }

    template<typename embedding_t>
    using regular_graph = regular_graph_internal::regular_graph<embedding_t>;

    using regular_grid_graph_1d = regular_graph<hg::embedding_grid_1d>;
    using regular_grid_graph_2d = regular_graph<hg::embedding_grid_2d>;
    using regular_grid_graph_3d = regular_graph<hg::embedding_grid_3d>;
    using regular_grid_graph_4d = regular_graph<hg::embedding_grid_4d>;

    namespace graph {
        template<typename embedding_t>
        struct graph_traits<hg::regular_graph<embedding_t>> {
            using G = hg::regular_graph<embedding_t>;

            using vertex_descriptor = typename G::vertex_descriptor;
            using edge_descriptor = typename G::edge_descriptor;
            using out_edge_iterator = typename G::out_edge_iterator;

            using directed_category = typename G::directed_category;
            using edge_parallel_category = typename G::edge_parallel_category;
            using traversal_category = typename G::traversal_category;

            using degree_size_type = typename G::degree_size_type;

            using in_edge_iterator = typename G::in_edge_iterator;
            using vertex_iterator = typename G::vertex_iterator;
            using vertices_size_type = typename G::vertices_size_type;
            using adjacency_iterator = typename G::adjacency_iterator;
        };
    }

    template<typename embedding_t>
    using regular_graph_out_edge_iterator = typename regular_graph_internal::regular_graph<embedding_t>::out_edge_iterator;

    template<typename embedding_t>
    using regular_graph_adjacent_vertex_iterator = typename regular_graph_internal::regular_graph<embedding_t>::adjacency_iterator;

    template<typename embedding_t>
    std::pair<typename hg::regular_graph<embedding_t>::out_edge_iterator, typename hg::regular_graph<embedding_t>::out_edge_iterator>
    out_edges(typename hg::regular_graph<embedding_t>::vertex_descriptor u, const hg::regular_graph<embedding_t> &g) {
        return std::make_pair(
                hg::regular_graph_out_edge_iterator<embedding_t>(
                        hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g),
                        [u](typename hg::regular_graph<embedding_t>::vertex_descriptor v) {
                            return std::make_pair(u, v);
                        }),
                hg::regular_graph_out_edge_iterator<embedding_t>(
                        hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g, true),
                        [u](typename hg::regular_graph<embedding_t>::vertex_descriptor v) {
                            return std::make_pair(u, v);
                        })
        );
    }

    template<typename embedding_t>
    std::pair<typename hg::regular_graph<embedding_t>::out_edge_iterator, typename hg::regular_graph<embedding_t>::out_edge_iterator>
    in_edges(typename hg::regular_graph<embedding_t>::vertex_descriptor u, const hg::regular_graph<embedding_t> &g) {
        return std::make_pair(
                hg::regular_graph_out_edge_iterator<embedding_t>(
                        hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g),
                        [u](typename hg::regular_graph<embedding_t>::vertex_descriptor v) {
                            return std::make_pair(v, u);
                        }),
                hg::regular_graph_out_edge_iterator<embedding_t>(
                        hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g, true),
                        [u](typename hg::regular_graph<embedding_t>::vertex_descriptor v) {
                            return std::make_pair(v, u);
                        })
        );
    }


    template<typename embedding_t>
    typename hg::regular_graph<embedding_t>::vertices_size_type
    num_vertices(const hg::regular_graph<embedding_t> &g) {
        return g.num_vertices();
    };

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
    adjacent_vertices(typename hg::regular_graph<embedding_t>::vertex_descriptor u,
                      const hg::regular_graph<embedding_t> &g) {
        return std::make_pair<typename hg::regular_graph<embedding_t>::adjacency_iterator, typename hg::regular_graph<embedding_t>::adjacency_iterator>(
                hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g),
                hg::regular_graph_adjacent_vertex_iterator<embedding_t>(u, g, true));
    };

    template<typename embedding_t>
    typename hg::regular_graph_internal::regular_graph<embedding_t>::degree_size_type
    hg::regular_graph_internal::regular_graph<embedding_t>::out_degree(const vertex_descriptor v) const {
        if (is_in_safe_area(m_embedding.lin2grid(v))) {
            return m_neighbours.size();
        } else {
            degree_size_type count = 0;
            auto its = adjacent_vertices(v, *this);
            auto & it1 = its.first;
            auto & it2 = its.second;
            for (;it1 != it2; ++it1) {
                count++;
            }
            return count;
        }
    }

    template<typename embedding_t>
    typename hg::regular_graph<embedding_t>::degree_size_type
    out_degree(
            const typename hg::regular_graph<embedding_t>::vertex_descriptor v,
            const hg::regular_graph<embedding_t> &g) {
        return g.out_degree(v);
    }

    template<typename embedding_t>
    typename hg::regular_graph<embedding_t>::degree_size_type
    in_degree(
            const typename hg::regular_graph<embedding_t>::vertex_descriptor v,
            const hg::regular_graph<embedding_t> &g) {
        return out_degree(v, g);
    }

    template<typename embedding_t>
    typename hg::regular_graph<embedding_t>::degree_size_type
    degree(
            const typename hg::regular_graph<embedding_t>::vertex_descriptor v,
            const hg::regular_graph<embedding_t> &g) {
        return out_degree(v, g);
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
    using hg::num_vertices;
    using hg::adjacent_vertices;
}
#endif

