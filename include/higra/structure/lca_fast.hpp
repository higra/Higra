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
#include <stack>

namespace hg {
    namespace lca_internal {

        /**
         * n x log(n) pre-processing of a tree to obtain a constant query time for lowest common ancestors of two nodes
         * @tparam tree_t
         */
        template<typename tree_t>
        struct lca_fast {
        public:
            template<typename T1, typename T2>
            struct internal_state {

                size_t m_num_vertices;
                T1 m_Euler;
                T1 m_Depth;
                T1 m_Represent;
                T1 m_Number;
                T2 m_Minim;
                size_t m_rep;

                internal_state(size_t num_vertices,
                               const T1 &Euler,
                               const T1 &Depth,
                               const T1 &Represent,
                               const T1 &Number,
                               const T2 &Minim,
                               size_t rep) :
                        m_num_vertices(num_vertices),
                        m_Euler(Euler),
                        m_Depth(Depth),
                        m_Represent(Represent),
                        m_Number(Number),
                        m_Minim(Minim),
                        m_rep(rep) {}

                internal_state(size_t num_vertices,
                               T1 &&Euler,
                               T1 &&Depth,
                               T1 &&Represent,
                               T1 &&Number,
                               T2 &&Minim,
                               size_t rep) :
                        m_num_vertices(num_vertices),
                        m_Euler(std::forward<T1>(Euler)),
                        m_Depth(std::forward<T1>(Depth)),
                        m_Represent(std::forward<T1>(Represent)),
                        m_Number(std::forward<T1>(Number)),
                        m_Minim(std::forward<T2>(Minim)),
                        m_rep(rep) {}
            };

        private:

            using array = xt::xtensor<std::size_t, 1>;
            using array2d = xt::xtensor<std::size_t, 2>;
            using vertex_t = typename tree_t::vertex_descriptor;

            size_t m_num_vertices;

            array Euler;
            array Depth;
            array Represent;
            array Number;
            array2d Minim;

            size_t rep = 0;

            void computeDepth(const tree_t &tree) {
                Depth[root(tree)] = 0;
                for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
                    Depth[i] = Depth[parent(i, tree)] + 1;
                }
            }

            void LCApreprocessEuler(const tree_t &tree) {
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
                    Euler[nbr] = e.node;
                    if (e.first_visit) {
                        Number[e.node] = nbr;
                        Represent[nbr] = e.node;
                        for (auto son: children_iterator(e.node, tree)) {
                            stack.push({e.node, false});
                            stack.push({son, true});
                        }
                    }
                }
            }


            void LCApreprocess(const tree_t &tree) {
                //O(n.log(n)) preprocessing

                computeDepth(tree);
                LCApreprocessEuler(tree);
                index_t nbNodes = m_num_vertices;
                index_t nbRepresent = 2 * nbNodes - 1;

                int logn = (int) (ceil(log((double) (nbRepresent)) / log(2.0)));

                Minim.resize({(size_t) logn, (size_t) nbRepresent});

                parfor(0, nbRepresent - 1, [this](index_t i) {
                    if (this->Depth[this->Euler[i]] < this->Depth[this->Euler[i + 1]]) {
                        this->Minim(0, i) = i;
                    } else {
                        this->Minim(0, i) = i + 1;
                    }
                });
                Minim(0, nbRepresent - 1) = nbRepresent - 1;

                for (int j = 1; j < logn; j++) {
                    index_t k1 = (index_t) (1 << (j - 1));
                    index_t k2 = k1 << 1;
                    parfor(0, nbRepresent, [this, k1, k2, j, nbRepresent](index_t i) {
                        if ((i + k2) >= nbRepresent) {
                            Minim(j, i) = nbRepresent - 1;
                        } else {
                            if (this->Depth[this->Euler[this->Minim(j - 1, i)]] <=
                                this->Depth[this->Euler[this->Minim(j - 1, i + k1)]]) {
                                this->Minim(j, i) = this->Minim(j - 1, i);
                            } else {
                                this->Minim(j, i) = this->Minim(j - 1, i + k1);
                            }
                        }
                    });
                }
            }

            lca_fast() {
                HG_TRACE();
                m_num_vertices = 0;
            }

            template<typename T1, typename T2>
            void set_state(const internal_state<T1, T2> &state) {
                m_num_vertices = state.m_num_vertices;
                Euler = state.m_Euler;
                Depth = state.m_Depth;
                Represent = state.m_Represent;
                Number = state.m_Number;
                Minim = state.m_Minim;
                rep = state.m_rep;
            }

            template<typename T1, typename T2>
            void set_state(internal_state<T1, T2> &&state) {
                m_num_vertices = state.m_num_vertices;
                Euler = std::move(state.m_Euler);
                Depth = std::move(state.m_Depth);
                Represent = std::move(state.m_Represent);
                Number = std::move(state.m_Number);
                Minim = std::move(state.m_Minim);
                rep = state.m_rep;
            }

        public:

            lca_fast(const tree_t &tree) {
                HG_TRACE();
                auto nbNodes = hg::num_vertices(tree);
                m_num_vertices = nbNodes;
                Depth.resize({nbNodes});
                Number.resize({nbNodes});

                Euler.resize({2 * nbNodes - 1});
                Represent.resize({2 * nbNodes - 1});

                LCApreprocess(tree);
            }

            /**
             * Return the lowest common ancestor of two nodes
             * @param n1
             * @param n2
             * @return
             */
            vertex_t lca(vertex_t n1, vertex_t n2) const {
                index_t ii, jj, kk;
                int k;
                ii = Number[n1];
                jj = Number[n2];
                if (ii == jj)
                    return Represent[ii];

                if (ii > jj) {
                    kk = jj;
                    jj = ii;
                    ii = kk;
                }

                k = (int) (log((double) (jj - ii)) / log(2.));

                if (Depth[Euler[Minim(k, ii)]] < Depth[Euler[Minim(k, jj - (index_t) (1 << (k)))]]) {
                    return Represent[Number[Euler[Minim(k, ii)]]];
                } else {
                    return Represent[Number[Euler[Minim(k, jj - (index_t) (1 << k))]]];
                }
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
                auto result = array_1d<vertex_t>::from_shape({size});

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
                auto result = array_1d<vertex_t>::from_shape({size});

                parfor(0, size, [&vertices1, &vertices2, &result, this](index_t i) {
                    result(i) = this->lca(vertices1(i), vertices2(i));
                });
                return result;
            }

            auto num_vertices() const {
                return m_num_vertices;
            }


            /**
             * Returns a copy of the internal state of the object (can be saved/loaded/transmitted...)
             * @return
             */
            auto get_state() const {
                return internal_state<array, array2d>(m_num_vertices,
                                                      Euler,
                                                      Depth,
                                                      Represent,
                                                      Number,
                                                      Minim,
                                                      rep);
            }

            template<typename T1, typename T2>
            static lca_fast make_lca_fast(const internal_state<T1, T2> &state) {
                auto lcaf = lca_fast();
                lcaf.set_state(state);
                return lcaf;
            }

            template<typename T1, typename T2>
            static lca_fast make_lca_fast(internal_state<T1, T2> &&state) {
                auto lcaf = lca_fast();
                lcaf.set_state(std::forward<internal_state<T1, T2>>(state));
                return lcaf;
            }

        };
    }

    using lca_fast = lca_internal::lca_fast<tree>;
}