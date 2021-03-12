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

#include "../graph.hpp"
#include "details/range_minimum_query.hpp"
#include <stack>

namespace hg {
    namespace lca_internal {

        /**
         * Lowest common ancestor solver based on the range minimum query
         * @tparam tree_t
         * @tparam rmq_t range minimum query solver type
         */
        template<typename tree_t, typename rmq_t>
        struct lca_rmq {

            using rmq_type = rmq_t;

        public:
            using self_type = lca_rmq<tree_t, rmq_t>;

            /**
             * Extra arguments are forwarded to the range minimum query solver constructor
             * @tparam Args
             * @param tree
             * @param args
             */
            template<typename ...Args>
            lca_rmq(const tree_t &tree, Args &&... args) {
                HG_TRACE();
                auto num_nodes = hg::num_vertices(tree);
                auto num_elements_Euler_tour = 2 * num_nodes - 1;

                m_first_visit_in_Euler_tour.resize({num_nodes});
                m_tree_Euler_tour_map.resize({num_elements_Euler_tour});
                m_tree_Euler_tour_depth.resize({num_elements_Euler_tour});

                compute_Euler_tour(tree);
                m_rmq_solver = rmq_t(m_tree_Euler_tour_depth, std::forward<Args>(args)...);
            }

            /**
             * Return the lowest common ancestor of two nodes
             * @param n1
             * @param n2
             * @return
             */
            index_t lca(index_t n1, index_t n2) const {
                if (n1 == n2)
                    return n1;

                index_t ii = m_first_visit_in_Euler_tour(n1);
                index_t jj = m_first_visit_in_Euler_tour(n2);

                if (ii > jj) {
                    std::swap(ii, jj);
                }

                return m_tree_Euler_tour_map(m_rmq_solver.query(ii, jj));
            }

            /**
             * Return the lowest common ancestors of a range of pairs of nodes
             * @tparam T
             * @param range
             * @return
             */
            template<typename T>
            auto lca(const T &range) const {
                HG_TRACE();
                size_t size = range.end() - range.begin();
                auto result = array_1d<index_t>::from_shape({size});

                auto it = range.begin();
                parfor(0, size, [&result, &it, this](index_t i) {
                    auto e = it[i];
                    result(i) = this->lca(e.first, e.second);
                });
                return result;
            }

            /**
             * Given two 1d array of graph vertex indices v1 and v2, both containing n elements,
             * this function returns a 1d array or tree vertex indices of size n such that
             * for all i in 0..n-1, res(i) = lca(v1(i); v2(i))
             *
             * @tparam T
             * @param xvertices1 first array of graph vertices
             * @param xvertices2 second array of graph vertices
             * @return array of lowest common ancestors
             */
            template<typename T>
            auto lca(const xt::xexpression<T> &xvertices1, const xt::xexpression<T> &xvertices2) const {
                HG_TRACE();
                auto &vertices1 = xvertices1.derived_cast();
                auto &vertices2 = xvertices2.derived_cast();
                hg_assert_1d_array(vertices1);
                hg_assert_integral_value_type(vertices1);
                hg_assert_same_shape(vertices1, vertices2);

                auto size = vertices1.size();
                auto result = array_1d<index_t>::from_shape({size});

                parfor(0, size, [&vertices1, &vertices2, &result, this](index_t i) {
                    result(i) = this->lca(vertices1(i), vertices2(i));
                });
                return result;
            }

            template<template<typename> typename container_t>
            struct internal_state {
                using type = self_type;
                using rmq_state_type = typename rmq_t::template internal_state<container_t>;
                container_t<index_t> tree_Euler_tour_map;
                container_t<index_t> tree_Euler_tour_depth;
                container_t<index_t> first_visit_in_Euler_tour;
                rmq_state_type rmq_state;

                internal_state(const container_t<index_t> &_tree_Euler_tour_map,
                               const container_t<index_t> &_tree_Euler_tour_depth,
                               const container_t<index_t> &_first_visit_in_Euler_tour,
                               const rmq_state_type &_rmq_state) :
                        tree_Euler_tour_map(_tree_Euler_tour_map),
                        tree_Euler_tour_depth(_tree_Euler_tour_depth),
                        first_visit_in_Euler_tour(_first_visit_in_Euler_tour),
                        rmq_state(_rmq_state) {}

                internal_state(container_t<index_t> &&_tree_Euler_tour_map,
                               container_t<index_t> &&_tree_Euler_tour_depth,
                               container_t<index_t> &&_first_visit_in_Euler_tour,
                               rmq_state_type &&_rmq_state) :
                        tree_Euler_tour_map(std::move(_tree_Euler_tour_map)),
                        tree_Euler_tour_depth(std::move(_tree_Euler_tour_depth)),
                        first_visit_in_Euler_tour(std::move(_first_visit_in_Euler_tour)),
                        rmq_state(std::move(_rmq_state)) {}
            };

            auto get_state() const {
                return internal_state<array_1d>(m_tree_Euler_tour_map,
                                                m_tree_Euler_tour_depth,
                                                m_first_visit_in_Euler_tour,
                                                m_rmq_solver.get_state());
            }

            template<template<typename> typename container_t>
            static auto make_from_state(internal_state<container_t> &&state) {
                using lca_t = typename internal_state<container_t>::type;
                lca_t lca;
                lca.set_state(std::move(state));
                return lca;
            }

            template<template<typename> typename container_t>
            static auto make_from_state(const internal_state<container_t> &state) {
                using lca_t = typename internal_state<container_t>::type;
                lca_t lca;
                lca.set_state(state);
                return lca;
            }

            index_t num_elements() const{
                return m_first_visit_in_Euler_tour.size();
            }

        private:

            lca_rmq(){};

            template<template<typename> typename container_t>
            void set_state(internal_state<container_t> &&state) {
                m_tree_Euler_tour_map = std::move(state.tree_Euler_tour_map);
                m_tree_Euler_tour_depth = std::move(state.tree_Euler_tour_depth);
                m_first_visit_in_Euler_tour = std::move(state.first_visit_in_Euler_tour);
                m_rmq_solver = rmq_t::make_from_state(std::move(state.rmq_state), m_tree_Euler_tour_depth);
            }

            template<template<typename> typename container_t>
            void set_state(const internal_state<container_t> &state) {
                m_tree_Euler_tour_map = state.tree_Euler_tour_map;
                m_tree_Euler_tour_depth = state.tree_Euler_tour_depth;
                m_first_visit_in_Euler_tour = state.first_visit_in_Euler_tour;
                m_rmq_solver = rmq_t::make_from_state(state.rmq_state, m_tree_Euler_tour_depth);
            }

            // tree node index visited at each step of the Euler tour
            array_1d<index_t> m_tree_Euler_tour_map;
            // depth of the tree node visited at each step of the Euler tour
            array_1d<index_t> m_tree_Euler_tour_depth;
            // index of the first time a node of the tree is visited in the Euler tour
            array_1d<index_t> m_first_visit_in_Euler_tour;

            // rmq solver
            rmq_t m_rmq_solver;

            void compute_Euler_tour(const tree_t &tree) {
                tree.compute_children();

                struct se {
                    index_t node;
                    bool first_visit;
                };

                index_t nbr = -1;

                std::stack<se> stack;
                stack.push({tree.root(), true});

                while (!stack.empty()) {
                    auto e = stack.top();
                    stack.pop();
                    nbr++;
                    m_tree_Euler_tour_map[nbr] = e.node;
                    if (e.first_visit) {
                        m_first_visit_in_Euler_tour[e.node] = nbr;
                        for (auto son: children_iterator(e.node, tree)) {
                            stack.push({e.node, false});
                            stack.push({son, true});
                        }
                    }
                    if (e.node == tree.root()) {
                        m_tree_Euler_tour_depth[nbr] = 0;
                    } else {
                        m_tree_Euler_tour_depth[nbr] =
                                m_tree_Euler_tour_depth[m_first_visit_in_Euler_tour[parent(e.node, tree)]] + 1;
                    }
                }
            }

        };

    }

    using lca_sparse_table_block = lca_internal::lca_rmq<tree, range_minimum_query_internal::rmq_sparse_table_block<index_t>>;
    using lca_sparse_table = lca_internal::lca_rmq<tree, range_minimum_query_internal::rmq_sparse_table<index_t>>;

    using lca_fast = lca_sparse_table_block;
}