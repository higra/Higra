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

        struct tree_graph_traversal_category :
                virtual public boost::incidence_graph_tag,
                virtual public boost::bidirectional_graph_tag,
                virtual public boost::adjacency_graph_tag,
                virtual public boost::vertex_list_graph_tag {
        };

        template<typename T=std::size_t>
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

            // EdgeListGraph associated types
            using edges_size_type = std::size_t;
            //using edge_iterator = typename container_gen<edgeS, edge_descriptor>::type::const_iterator;


            // IncidenceGraph associated types
            //using out_iterator_transform_function = std::function<edge_descriptor(out_edge_t)>;
            //using out_edge_iterator = boost::transform_iterator<out_iterator_transform_function,
            //        typename container_gen<edgeS, out_edge_t>::type::const_iterator>;
            using degree_size_type = std::size_t;

            //BidirectionalGraph associated types
            //using in_edge_iterator = out_edge_iterator;

            //AdjacencyGraph associated types
            //using adjacent_iterator_transform_function = std::function<vertex_descriptor(out_edge_t)>;
            //using adjacency_iterator = boost::transform_iterator<adjacent_iterator_transform_function,
            //        typename container_gen<edgeS, out_edge_t>::type::const_iterator>;


            // custom edge index iterators
            using edge_index_t = std::size_t;
            using edge_index_iterator = boost::counting_iterator<edge_index_t>;
            //using out_edge_index_iterator = adjacency_iterator;
            //using in_edge_index_iterator = adjacency_iterator;

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
                std:
                size_t max_leave_nb = 0;

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
                return _children[v].size() + (v != _root) ? 1 : 0;
            }


        private:

            vertex_descriptor _root;
            std::size_t _num_vertices;
            std::size_t _num_leaves;
            xt::xarray<vertex_descriptor> _parents;
            std::vector<children_list_t> _children;
        };
    }

    using tree = tree_internal::tree<>;

}