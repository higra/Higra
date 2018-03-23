//
// Created by user on 3/14/18.
//

#pragma once

#include <functional>
#include <boost/graph/graph_concepts.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <vector>
#include <utility>
#include "debug.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

namespace hg {

    namespace tree_internal {

        // forward declaration
        struct tree_graph_adjacent_vertex_iterator;
        
        struct tree_graph_traversal_category :
                virtual public boost::incidence_graph_tag,
                virtual public boost::bidirectional_graph_tag,
                virtual public boost::adjacency_graph_tag,
                virtual public boost::vertex_list_graph_tag {
        };

        struct tree {



            // Graph associated types
            using vertex_descriptor = std::size_t;
            using children_list_t = std::vector<vertex_descriptor>;
            using edge_descriptor = std::pair<vertex_descriptor, vertex_descriptor>;
            using directed_category = boost::undirected_tag;
            using edge_parallel_category = boost::disallow_parallel_edge_tag;
            using traversal_category = tree_graph_traversal_category;

            // VertexListGraph associated types

            using vertex_iterator = boost::counting_iterator<vertex_descriptor>;
            using vertices_size_type = std::size_t;

            //AdjacencyGraph associated types

            using adjacency_iterator = tree_graph_adjacent_vertex_iterator;

            // custom edge index iterators
            using edge_index_t = std::size_t;
            //using edge_index_iterator = boost::counting_iterator<edge_index_t>;
            //using out_edge_index_iterator = adjacency_iterator;
            //using in_edge_index_iterator = adjacency_iterator;

            // EdgeListGraph associated types
            using edges_size_type = std::size_t;
            using _edge_iterator_transform_function = std::function<edge_descriptor(edge_index_t)>;
            using edge_iterator = boost::transform_iterator<_edge_iterator_transform_function,
                    boost::counting_iterator<vertex_descriptor>>;


            // IncidenceGraph associated types
            //using out_iterator_transform_function = std::function<edge_descriptor(out_edge_t)>;
            //using out_edge_iterator = boost::transform_iterator<out_iterator_transform_function,
            //        typename container_gen<edgeS, out_edge_t>::type::const_iterator>;
            using degree_size_type = std::size_t;

            //BidirectionalGraph associated types
            //using in_edge_iterator = out_edge_iterator;





            tree(xt::xarray<vertex_descriptor> parents = {0}) : _parents(parents), _children(parents.size()) {

                hg_assert(_parents.shape().size() == 1, "parents must be a linear (1d) array");
                _num_vertices = _parents.size();
                _root = _num_vertices - 1;
                hg_assert(_parents(_root) == _root, "nodes are not in a topological order (last node is not a root)");

                for (vertex_descriptor v = 0; v < _root; ++v) {
                    vertex_descriptor parent_v = _parents(v);
                    hg_assert(parent_v != v, "several root nodes detected");
                    hg_assert(parent_v > v, "nodes are not in a topological order");
                    _children[parent_v].push_back(v);
                }

                _num_leaves = 0;

                for (vertex_descriptor v = 0; v <= _root; ++v) {
                    if (_children[v].size() == 0) {
                        hg_assert(_num_leaves == v, "leaves nodes are not before internal nodes");
                        _num_leaves++;
                    }
                }
            };

            vertices_size_type num_vertices() const {
                return _num_vertices;
            }

            vertices_size_type num_leaves() const {
                return _num_leaves;
            }

            edges_size_type num_edges() const {
                return (_num_vertices == 0) ? 0 : _num_vertices - 1;
            }

            vertex_descriptor root() const {
                return _root;
            }

            degree_size_type degree(vertex_descriptor v) const {
                return _children[v].size() + ((v != _root) ? 1 : 0);
            }

            auto children_cbegin(vertex_descriptor v) const {
                return _children[v].cbegin();
            }

            auto children_cend(vertex_descriptor v) const {
                return _children[v].cend();
            }

            template<typename... Args>
            const auto parent(Args &&... args) const {
                return _parents(std::forward<Args>(args)...);
            }

            const auto &parents() const {
                return _parents;
            }

        private:

            vertex_descriptor _root;
            std::size_t _num_vertices;
            std::size_t _num_leaves;
            xt::xarray<vertex_descriptor> _parents;
            std::vector<children_list_t> _children;
        };



        // Iterator

        struct tree_graph_adjacent_vertex_iterator :
                public boost::iterator_facade<tree_graph_adjacent_vertex_iterator,
                        tree::vertex_descriptor,
                        boost::forward_traversal_tag,
                        tree::vertex_descriptor> {
        public:
            using graph_t = tree;
            using graph_vertex_t = graph_t::vertex_descriptor;
            using point_list_iterator_t = graph_t::children_list_t::const_iterator;

            tree_graph_adjacent_vertex_iterator() {}

            tree_graph_adjacent_vertex_iterator(graph_vertex_t source,
                                                graph_vertex_t parent,
                                                point_list_iterator_t child_iterator
            )
                    : _parent(parent), _child_iterator(child_iterator) {

                if (parent == source) {
                    _iterating_on_children = true;
                }
            }

        private:


            friend class boost::iterator_core_access;

            void increment() {
                if (_iterating_on_children) {
                    _child_iterator++;
                } else {
                    _iterating_on_children = true;
                }
            }

            bool equal(const tree_graph_adjacent_vertex_iterator &that) const {
                return this->_iterating_on_children && that._iterating_on_children &&
                       (this->_child_iterator == that._child_iterator);
            }


            graph_vertex_t dereference() const {
                if (_iterating_on_children) {
                    return *_child_iterator;
                } else {
                    return _parent;
                }
            }

            graph_vertex_t _parent;

            bool _iterating_on_children = false;

            point_list_iterator_t _child_iterator;
        };

    }


    using tree = tree_internal::tree;

}


namespace boost {

    template<>
    struct graph_traits<hg::tree> {
        using G = hg::tree;

        using vertex_descriptor = typename G::vertex_descriptor;
        using edge_descriptor = typename G::edge_descriptor;
        using edge_iterator = typename G::edge_iterator;
        //using out_edge_iterator = typename G::out_edge_iterator;

        using directed_category = typename G::directed_category;
        using edge_parallel_category = typename G::edge_parallel_category;
        using traversal_category = typename G::traversal_category;

        using degree_size_type = typename G::degree_size_type;

        //using in_edge_iterator = typename G::in_edge_iterator;
        using vertex_iterator = typename G::vertex_iterator;
        using vertices_size_type = typename G::vertices_size_type;
        using edges_size_type = typename G::edges_size_type;
        using adjacency_iterator = typename G::adjacency_iterator;
    };

    inline hg::tree::vertices_size_type num_vertices(const hg::tree &g) {
        return g.num_vertices();
    }

    inline hg::tree::edges_size_type num_edges(const hg::tree &g) {
        return g.num_edges();
    }

    inline hg::tree::degree_size_type degree(typename hg::tree::vertex_descriptor v,
                                             const hg::tree &g) {
        return g.degree(v);
    }

    inline hg::tree::degree_size_type in_degree(typename hg::tree::vertex_descriptor v,
                                                const hg::tree &g) {
        return g.degree(v);
    }


    inline hg::tree::degree_size_type out_degree(typename hg::tree::vertex_descriptor v,
                                                 const hg::tree &g) {
        return g.degree(v);
    }


    inline
    std::pair<typename hg::tree::vertex_iterator, typename hg::tree::vertex_iterator>
    vertices(const hg::tree &g) {
        using vertex_iterator = typename hg::tree::vertex_iterator;
        return std::make_pair(
                vertex_iterator(0),                 // The first iterator position
                vertex_iterator(num_vertices(g))); // The last iterator position
    }

    inline
    std::pair<typename hg::tree::edge_iterator, typename hg::tree::edge_iterator>
    edges(const hg::tree &g) {
        using it = hg::tree::edge_iterator;
        auto fun = [&g](hg::tree::edge_index_t i) {
            return std::make_pair(i, g.parent(i));
        };
        return std::make_pair(
                it(boost::counting_iterator<hg::tree::vertex_descriptor>(0),
                   fun),                 // The first iterator position
                it(boost::counting_iterator<hg::tree::vertex_descriptor>(g.num_edges()),
                   fun)); // The last iterator position
    }

    inline
    typename hg::tree::vertex_descriptor
    source(
            typename hg::tree::edge_descriptor &e,
            hg::tree &) {
        return e.first;
    }

    inline
    typename hg::tree::vertex_descriptor
    target(
            typename hg::tree::edge_descriptor &e,
            hg::tree &) {
        return e.second;
    }

    inline
    std::pair<typename hg::tree::adjacency_iterator, typename hg::tree::adjacency_iterator>
    adjacent_vertices(typename hg::tree::vertex_descriptor v, const hg::tree &g) {
        using it = typename hg::tree::adjacency_iterator;
        auto par = g.parent(v);
        return std::make_pair(
                it(v, par, g.children_cbegin(v)),
                it(par, par, g.children_cend(v)));
    }


/*
    template<typename T>
    std::pair<typename hg::tree::out_edge_iterator, typename hg::tree::out_edge_iterator>
    out_edges(typename hg::tree::vertex_descriptor v, hg::tree &g) {
        auto fun = [v](const typename hg::tree::out_edge_t &oe) {
            return std::make_pair(v, oe.second);
        };
        using it = typename hg::tree::out_edge_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }

    template<typename T>
    std::pair<typename hg::tree::out_edge_iterator, typename hg::tree::out_edge_iterator>
    in_edges(typename hg::tree::vertex_descriptor v, const hg::tree &g) {
        auto fun = [v](const typename hg::tree::out_edge_t &oe) {
            return std::make_pair(oe.second, v);
        };
        using it = typename hg::tree::out_edge_iterator;
        return std::make_pair(
                it(g.out_edges_cbegin(v), fun),
                it(g.out_edges_cend(v), fun));
    }


*/
}