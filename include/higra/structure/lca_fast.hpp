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
        private:

            using array = xt::xtensor<std::size_t, 1>;
            using array2d = xt::xtensor<std::size_t, 2>;
            using vertex_t = typename tree_t::vertex_descriptor;

            array Euler;
            array Depth;
            array Represent;
            array Number;
            array2d Minim;

            size_t rep = 0;

            size_t nbR;

            void computeDepth(const tree_t &tree) {
                Depth[root(tree)] = 0;
                for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
                    Depth[i] = Depth[parent(i, tree)] + 1;
                }
            }

            void LCApreprocessEuler(const tree_t &tree) {
                struct se {
                    index_t node;
                    bool first_visit;
                };

                long nbr = -1;

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

            /* Recursive version
            long LCApreprocessDepthFirst(const tree_t &tree, std::size_t node, std::size_t depth, long *nbr,
                                         std::size_t *rep) {
                (*nbr)++;
                Euler[*nbr] = node;
                Number[node] = *nbr;
                Depth[node] = depth;
                Represent[*nbr] = node;
                (*rep)++;
                for (auto son: children_iterator(node, tree)) {
                    LCApreprocessDepthFirst(tree, son, depth + 1, nbr, rep);
                    Euler[++(*nbr)] = node;
                }
                return *nbr;
            }
            */

            void LCApreprocess(const tree_t &tree) {
                //O(n.log(n)) preprocessing
                /* Recursive version
                long nbr = -1;
                std::size_t rep = 0;

                nbr = LCApreprocessDepthFirst(tree, tree.root(), 0, &nbr, &rep);

                // Check that the number of nodes in the tree was correct
                std::size_t nbNodes = rep;
                std::size_t nbRepresent = 2 * nbNodes - 1;
                hg_assert((std::size_t) (nbr + 1) == nbRepresent, "oops incorrect number of nodes!");
                */
                computeDepth(tree);
                LCApreprocessEuler(tree);
                index_t nbNodes = (index_t)num_vertices(tree);
                index_t nbRepresent = 2 * nbNodes - 1;

                index_t logn = (index_t) (ceil(log((double) (nbRepresent)) / log(2.0)));

                Minim.resize({(size_t)logn, (size_t)nbRepresent});

                for (index_t i = 0; i < nbRepresent - 1; i++) {
                    if (Depth[Euler[i]] < Depth[Euler[i + 1]]) {
                        Minim(0, i) = i;
                    } else {
                        Minim(0, i) = i + 1;
                    }
                }
                Minim(0, nbRepresent - 1) = nbRepresent - 1;

                for (index_t j = 1; j < logn; j++) {
                    index_t k1 = 1 << (j - 1);
                    index_t k2 = k1 << 1;
                    for (index_t i = 0; i < nbRepresent; i++) {
                        if ((i + k2) >= nbRepresent) {
                            Minim(j, i) = nbRepresent - 1;
                        } else {
                            if (Depth[Euler[Minim(j - 1, i)]] <= Depth[Euler[Minim(j - 1, i + k1)]]) {
                                Minim(j, i) = Minim(j - 1, i);
                            } else {
                                Minim(j, i) = Minim(j - 1, i + k1);
                            }
                        }
                    }
                }
            }


        public:
            lca_fast(const tree_t &tree) {
                HG_TRACE();
                auto nbNodes = num_vertices(tree);

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
                index_t ii, jj, kk, k;

                ii = Number[n1];
                jj = Number[n2];
                if (ii == jj)
                    return Represent[ii];

                if (ii > jj) {
                    kk = jj;
                    jj = ii;
                    ii = kk;
                }

                k = (index_t) (log((double) (jj - ii)) / log(2.));

                if (Depth[Euler[Minim(k, ii)]] < Depth[Euler[Minim(k, jj - (1 << (k)))]]) {
                    return Represent[Number[Euler[Minim(k, ii)]]];
                } else {
                    return Represent[Number[Euler[Minim(k, jj - (1 << k))]]];
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

                auto it = result.begin();
                for (const auto e: range) {
                    *it = lca(e.first, e.second);
                    it++;
                }
                return result;
            }


        };
    }

    using lca_fast = lca_internal::lca_fast<tree>;
}