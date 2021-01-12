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
#include "details/indexed_edge.hpp"
#include "details/graph_concepts.hpp"
#include "higra/structure/details/iterators.hpp"
#include <vector>
#include <utility>
#include "../utils.hpp"
#include "array.hpp"

namespace hg {


    enum class tree_category {
        component_tree,
        partition_tree
    };

    /**
     * Enum used in tree node iterator (leaves_to_root_iterator and root_to_leaves_iterator)
     * to include or exclude leaves from the iterator.
     */
    enum class leaves_it {
        include,
        exclude
    };

    /**
     * Enum used in tree node iterator (leaves_to_root_iterator and root_to_leaves_iterator)
     * to include or exclude the root from the iterator.
     */
    enum class root_it {
        include,
        exclude
    };

    namespace tree_internal {

        // forward declaration
        template<bool edge_index_iterator>
        struct tree_graph_adjacent_vertex_iterator;

        // forward declaration
        struct tree_graph_node_to_root_iterator;

        struct tree_graph_traversal_category :
                virtual public graph::incidence_graph_tag,
                virtual public graph::bidirectional_graph_tag,
                virtual public graph::adjacency_graph_tag,
                virtual public graph::vertex_list_graph_tag {
        };

        struct tree {

            // Graph associated types
            using vertex_descriptor = index_t;
            using edge_index_t = index_t;
            using children_list_t = std::vector<vertex_descriptor>;
            using children_iterator = children_list_t::const_iterator;
            using ancestors_iterator = tree_graph_node_to_root_iterator;
            using edge_descriptor = indexed_edge<vertex_descriptor, edge_index_t>;
            using directed_category = graph::undirected_tag;
            using edge_parallel_category = graph::disallow_parallel_edge_tag;
            using traversal_category = tree_graph_traversal_category;

            // VertexListGraph associated types
            using vertex_iterator = counting_iterator<vertex_descriptor>;
            using vertices_size_type = size_t;

            //AdjacencyGraph associated types
            using adjacency_iterator = tree_graph_adjacent_vertex_iterator<false>;

            // custom edge index iterators

            using edge_index_iterator = counting_iterator<edge_index_t>;
            //using out_edge_index_iterator = tree_graph_adjacent_vertex_iterator<true>;
            //using in_edge_index_iterator = out_edge_index_iterator;

            // EdgeListGraph associated types
            using edges_size_type = size_t;
            using _edge_iterator_transform_function = std::function<edge_descriptor(edge_index_t)>;
            using edge_iterator = transform_forward_iterator <_edge_iterator_transform_function,
            counting_iterator<vertex_descriptor>, edge_descriptor>;


            // IncidenceGraph associated types
            using out_iterator_transform_function = std::function<edge_descriptor(vertex_descriptor)>;
            using out_edge_iterator = transform_forward_iterator<out_iterator_transform_function,
                    tree_graph_adjacent_vertex_iterator<false>,
                    edge_descriptor>;
            using degree_size_type = size_t;

            //BidirectionalGraph associated types
            using in_edge_iterator = out_edge_iterator;

            tree() : _root(invalid_index), _num_vertices(0), _num_leaves(0) {

            }

            template<typename T>
            tree(const xt::xexpression<T> &parents = xt::xarray<vertex_descriptor>({0}),
                 tree_category category = tree_category::partition_tree) :
                    _parents(parents),
                    _children_computed(false),
                    _children(0),
                    _category(category) {
                HG_TRACE();
                _init();
            };

            template<typename T>
            tree(xt::xexpression<T> &&parents = xt::xarray<vertex_descriptor>({0}),
                 tree_category category = tree_category::partition_tree) :
                    _parents(std::move(parents.derived_cast())),
                    _children_computed(false),
                    _children(0),
                    _category(category) {
                HG_TRACE();
                _init();
            };

            const auto &category() const {
                return _category;
            }

            vertices_size_type num_vertices() const {
                return _num_vertices;
            }

            vertices_size_type num_leaves() const {
                return _num_leaves;
            }

            edges_size_type num_edges() const {
                return (_num_vertices == 0) ? 0 : _num_vertices - 1;
            }

            const auto &children(vertex_descriptor v) const {
                if (v < _num_leaves) {
                    return _empty_children;
                }
                return _children[v - _num_leaves];
            }

            size_t num_children(const vertex_descriptor v) const {
                if (v < _num_leaves) {
                    return 0;
                }
                return _children[v - _num_leaves].size();
            }

            vertex_descriptor root() const {
                return _root;
            }

            degree_size_type degree(vertex_descriptor v) const {
                return num_children(v) + ((v != _root) ? 1 : 0);
            }

            auto children_cbegin(vertex_descriptor v) const {
                return children(v).cbegin();
            }

            auto children_cend(vertex_descriptor v) const {
                return children(v).cend();
            }

            auto child(index_t i, vertex_descriptor v) const {
                return _children[v - _num_leaves][i];
            }

            template<typename... Args>
            auto parent(Args &&... args) const {
                return _parents(std::forward<Args>(args)...);
            }

            const auto &parents() const {
                return _parents;
            }

            auto leaves_iterator() const {
                return irange<index_t>(0, _num_leaves);
            }

            auto leaves_to_root_iterator(leaves_it leaves_opt = leaves_it::include,
                                         root_it root_opt = root_it::include) const {
                vertex_descriptor start = (leaves_opt == leaves_it::include) ? 0 : num_leaves();
                vertex_descriptor end = (root_opt == root_it::include) ? _num_vertices : _num_vertices - 1;
                return irange<index_t>(start, end);
            }

            auto root_to_leaves_iterator(leaves_it leaves_opt = leaves_it::include,
                                         root_it root_opt = root_it::include) const {
                vertex_descriptor end = (leaves_opt == leaves_it::include) ? -1 : num_leaves() - 1;
                vertex_descriptor start = (root_opt == root_it::include) ? _num_vertices - 1 : _num_vertices - 2;
                return irange<index_t>(start, end, -1);
            }

            auto edge_from_index(edge_index_t ei) const {
                return edge_descriptor(ei, parent(ei), ei);
            }

            auto is_leaf(vertex_descriptor v) const {
                return v < _num_leaves;
            }

            template<typename T>
            auto find_region(vertex_descriptor v, const typename T::value_type &lambda, const T &altitudes) const {
                while (parent(v) != v && altitudes[parent(v)] < lambda) {
                    v = parent(v);
                }
                return v;
            }

            void compute_children() const {
                if (!_children_computed) {
                    _children.resize(_num_vertices - _num_leaves);
                    for (vertex_descriptor v = 0; v < _root; ++v) {
                        vertex_descriptor parent_v = _parents(v);
                        _children[parent_v - _num_leaves].push_back(v);
                    }
                    _children_computed = true;
                }
            }

            void clear_children() const {
                _children.clear();
                _children_computed = false;
            }

            bool children_computed() const {
                return _children_computed;
            }

            auto sources() const{
                return xt::arange<index_t>(0, _num_vertices - 1);
            }

            auto targets() const{
                return xt::strided_view(_parents, {xt::range(0, _num_vertices - 1)});
            }


        private:

            void _init() {
                hg_assert(_parents.shape().size() == 1, "parents must be a linear (1d) array");
                _num_vertices = _parents.size();
                _root = _num_vertices - 1;
                hg_assert(_parents(_root) == _root, "nodes are not in a topological order (last node is not a root)");

                array_1d<index_t> num_children = xt::zeros_like(_parents);
                for (vertex_descriptor v = 0; v < _root; ++v) {
                    vertex_descriptor parent_v = _parents(v);
                    hg_assert(parent_v != v, "several root nodes detected");
                    hg_assert(parent_v > v, "nodes are not in a topological order");
                    num_children(parent_v)++;
                }

                index_t num_leaves = 0;

                for (vertex_descriptor v = 0; v <= _root; ++v) {
                    if (num_children(v) == 0) {
                        hg_assert(num_leaves == v, "leaves nodes are not before internal nodes");
                        num_leaves++;
                    }
                }
                _num_leaves = (size_t) num_leaves;
            }


            vertex_descriptor _root;
            size_t _num_vertices;
            index_t _num_leaves;
            array_1d <vertex_descriptor> _parents;
            mutable bool _children_computed;
            mutable std::vector<children_list_t> _children;
            tree_category _category;

            // for leaf nodes
            children_list_t _empty_children;
        };


        // Some meta-programming to use the following iterator for adjacency and edge indices
        template<bool b>
        auto _deference_parent(const tree::vertex_descriptor source, const tree::vertex_descriptor parent);


        template<>
        inline
        auto _deference_parent<true>(const tree::vertex_descriptor source, const tree::vertex_descriptor) {
            return source;
        }

        template<>
        inline
        auto _deference_parent<false>(const tree::vertex_descriptor, const tree::vertex_descriptor parent) {
            return parent;
        }

        // Iterator
        template<bool edge_index_iterator = false>
        struct tree_graph_adjacent_vertex_iterator :
                public forward_iterator_facade<tree_graph_adjacent_vertex_iterator<edge_index_iterator>,
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
                    : _source(source), _parent(parent), _child_iterator(child_iterator) {

                if (parent == source) {
                    _iterating_on_children = true;
                }
            }

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
                    return _deference_parent<edge_index_iterator>(_source, _parent);
                }
            }

        private:
            graph_vertex_t _source;
            graph_vertex_t _parent;

            bool _iterating_on_children = false;

            point_list_iterator_t _child_iterator;
        };


        struct tree_graph_node_to_root_iterator :
                public forward_iterator_facade<tree_graph_node_to_root_iterator, tree::vertex_descriptor> {
        public:
            using graph_t = tree;
            using graph_vertex_t = graph_t::vertex_descriptor;

            tree_graph_node_to_root_iterator(const graph_t &tree, const graph_vertex_t &node)
                    : m_position(node), m_tree(tree) {
            }

            void increment() {
                auto par = m_tree.parent(m_position);
                if (par != m_position) {
                    m_position = par;
                } else {
                    m_position = invalid_index;
                }
            }

            bool equal(const tree_graph_node_to_root_iterator &that) const {
                return this->m_position == that.m_position;
            }

            graph_vertex_t dereference() const {
                return m_position;
            }

        private:
            graph_vertex_t m_position;
            const graph_t &m_tree;
        };

    }

    using tree = tree_internal::tree;

    namespace graph {
        template<>
        struct graph_traits<hg::tree> {
            using G = hg::tree;

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

    inline
    auto
    num_leaves(const tree &t) {
        return t.num_leaves();
    }

    inline
    auto
    num_children(tree::vertex_descriptor v, const tree &t) {
        return t.num_children(v);
    }

    template<typename T>
    auto
    num_children(const xt::xexpression<T> &xvertices, const tree &t) {
        auto &vertices = xvertices.derived_cast();
        hg_assert_1d_array(vertices);
        hg_assert_integral_value_type(vertices);

        array_1d <size_t> result = xt::empty<index_t>({vertices.size()});
        for (index_t j = 0; j < (index_t) vertices.size(); j++) {
            result(j) = num_children(vertices(j), t);
        }
        return result;
    }

    inline
    const auto &
    category(const tree &t) {
        return t.category();
    }

    inline
    auto
    root(const tree &t) {
        return t.root();
    }

    inline
    auto
    parent(tree::vertex_descriptor v, const tree &t) {
        return t.parent(v);
    }

    template<typename T>
    auto
    parent(const xt::xexpression<T> &xvertices, const tree &t) {
        auto &vertices = xvertices.derived_cast();
        hg_assert_1d_array(vertices);
        hg_assert_integral_value_type(vertices);

        array_1d <index_t> result = xt::empty<index_t>({vertices.size()});
        for (index_t j = 0; j < (index_t) vertices.size(); j++) {
            result(j) = parent(vertices(j), t);
        }
        return result;
    }

    inline
    auto
    is_leaf(tree::vertex_descriptor v, const tree &t) {
        return t.is_leaf(v);
    }

    template<typename T>
    auto
    is_leaf(const xt::xexpression<T> &xvertices, const tree &t) {
        auto &vertices = xvertices.derived_cast();
        hg_assert_1d_array(vertices);
        hg_assert_integral_value_type(vertices);

        array_1d<bool> result = xt::empty<bool>({vertices.size()});
        for (index_t j = 0; j < (index_t) vertices.size(); j++) {
            result(j) = is_leaf(vertices(j), t);
        }
        return result;
    }

    inline
    const auto &
    parents(const tree &t) {
        return t.parents();
    }

    inline
    auto
    leaves_to_root_iterator(const tree &t,
                            leaves_it leaves_opt = leaves_it::include,
                            root_it root_opt = root_it::include) {
        return t.leaves_to_root_iterator(leaves_opt, root_opt);
    }

    inline
    auto
    root_to_leaves_iterator(const tree &t,
                            leaves_it leaves_opt = leaves_it::include,
                            root_it root_opt = root_it::include) {
        return t.root_to_leaves_iterator(leaves_opt, root_opt);
    }

    inline
    auto
    leaves_iterator(const tree &t) {
        return t.leaves_iterator();
    }

    inline
    auto
    ancestors(tree::vertex_descriptor v, const tree &t) {
        using it_t = tree_internal::tree_graph_node_to_root_iterator;
        return std::make_pair(it_t(t, v), it_t(t, invalid_index));
    }

    inline
    tree::edge_descriptor
    edge_from_index(tree::edge_index_t ei, const tree &g) {
        return g.edge_from_index(ei);
    }

    inline
    std::pair<tree::children_iterator, tree::children_iterator>
    children(const tree::vertex_descriptor v, const tree &g) {
        auto &c = g.children(v);
        return std::make_pair(c.cbegin(), c.cend());
    }

    inline
    tree::vertex_descriptor
    child(index_t i, tree::vertex_descriptor v, const tree &t) {
        return t.child(i, v);
    }

    template<typename T>
    auto
    child(index_t i, const xt::xexpression<T> &xvertices, const tree &t) {
        auto &vertices = xvertices.derived_cast();
        hg_assert_1d_array(vertices);
        hg_assert_integral_value_type(vertices);

        array_1d <index_t> result = xt::empty<index_t>({vertices.size()});
        for (index_t j = 0; j < (index_t) vertices.size(); j++) {
            result(j) = t.child(i, vertices(j));
        }
        return result;
    }

    inline
    hg::tree::vertices_size_type
    num_vertices(const hg::tree &g) {
        return g.num_vertices();
    }

    inline
    hg::tree::edges_size_type
    num_edges(const hg::tree &g) {
        return g.num_edges();
    }

    inline
    hg::tree::degree_size_type
    degree(const typename hg::tree::vertex_descriptor v,
           const hg::tree &g) {
        return g.degree(v);
    }

    inline
    hg::tree::degree_size_type
    in_degree(const typename hg::tree::vertex_descriptor v,
              const hg::tree &g) {
        return g.degree(v);
    }

    inline
    hg::tree::degree_size_type
    out_degree(const typename hg::tree::vertex_descriptor v,
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
            return g.edge_from_index(i);
        };
        return std::make_pair(
                it(counting_iterator<hg::tree::vertex_descriptor>(0),
                   fun),                 // The first iterator position
                it(counting_iterator<hg::tree::vertex_descriptor>(g.num_edges()),
                   fun)); // The last iterator position
    }

    inline
    std::pair<typename hg::tree::adjacency_iterator, typename hg::tree::adjacency_iterator>
    adjacent_vertices(typename hg::tree::vertex_descriptor v, const hg::tree &g) {
        using it = typename hg::tree::adjacency_iterator;
        auto &c = g.children(v);
        auto par = g.parent(v);
        return std::make_pair(
                it(v, par, c.cbegin()),
                it(par, par, c.cend()));
    }


    inline
    std::pair<hg::tree::out_edge_iterator, hg::tree::out_edge_iterator>
    out_edges(hg::tree::vertex_descriptor v, const hg::tree &g) {
        auto fun = [v](const typename hg::tree::vertex_descriptor t) {
            return hg::tree::edge_descriptor(v, t, (std::min)(v, t));
        };
        auto &c = g.children(v);
        using it = typename hg::tree::out_edge_iterator;
        using ita = typename hg::tree::adjacency_iterator;
        auto par = g.parent(v);
        return std::make_pair(
                it(ita(v, par, c.cbegin()), fun),
                it(ita(par, par, c.cend()), fun));
    }

    inline
    std::pair<hg::tree::out_edge_iterator, hg::tree::out_edge_iterator>
    in_edges(hg::tree::vertex_descriptor v, const hg::tree &g) {
        auto fun = [v](const typename hg::tree::vertex_descriptor t) {
            return hg::tree::edge_descriptor(t, v, (std::min)(v, t));
        };
        auto &c = g.children(v);
        using it = typename hg::tree::out_edge_iterator;
        using ita = typename hg::tree::adjacency_iterator;
        auto par = g.parent(v);
        return std::make_pair(
                it(ita(v, par, c.cbegin()), fun),
                it(ita(par, par, c.cend()), fun));
    }

    template<typename T>
    auto find_region(tree::vertex_descriptor v,
                     const typename T::value_type &lambda,
                     const T &altitudes,
                     const tree &tree) {
        return tree.find_region(v, lambda, altitudes);
    }

    template<typename T1, typename T2, typename T3>
    auto find_region(
            const xt::xexpression<T1> &xvertices,
            const xt::xexpression<T2> &xlambdas,
            const xt::xexpression<T3> &xaltitudes,
            const tree &t) {
        HG_TRACE();
        auto &vertices = xvertices.derived_cast();
        auto &lambdas = xlambdas.derived_cast();
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(t, altitudes);
        hg_assert_1d_array(altitudes);
        hg_assert_1d_array(vertices);
        hg_assert_integral_value_type(vertices);
        hg_assert_1d_array(lambdas);

        array_1d <index_t> result = array_1d<index_t>::from_shape({vertices.size()});

        for (index_t i = 0; i < (index_t) vertices.size(); i++) {
            result(i) = t.find_region(vertices(i), lambdas(i), altitudes);
        }
        return result;
    }

    inline
    auto lowest_common_ancestor(tree::vertex_descriptor v1,
                                tree::vertex_descriptor v2,
                                const tree &t) {
        while (v1 != v2) {
            if (v1 < v2) {
                v1 = parent(v1, t);
            } else {
                v2 = parent(v2, t);
            }
        }
        return v1;
    }

    template<typename T>
    auto lowest_common_ancestor(const xt::xexpression<T> &xvertices_1,
                                const xt::xexpression<T> &xvertices_2,
                                const tree &t) {
        auto &vertices_1 = xvertices_1.derived_cast();
        auto &vertices_2 = xvertices_2.derived_cast();
        hg_assert_1d_array(vertices_1);
        hg_assert_same_shape(vertices_1, vertices_2);
        hg_assert_integral_value_type(vertices_1);

        array_1d <index_t> lcas = array_1d<index_t>::from_shape({vertices_1.size()});

        for (index_t i = 0; i < (index_t) vertices_1.size(); i++) {
            lcas(i) = lowest_common_ancestor(vertices_1(i), vertices_2(i), t);
        }

        return lcas;
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
    using hg::num_vertices;
    using hg::num_edges;
    using hg::adjacent_vertices;
}
#endif






