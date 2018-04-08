//
// Created by user on 4/6/18.
//

#pragma once

#include "graph.hpp"

namespace hg {
    namespace lca_internal {
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

            std::size_t rep = 0;

            std::size_t nbR;

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


            void LCApreprocess(const tree_t &tree) {
                //O(n.log(n)) preprocessing
                long nbr = -1;
                std::size_t rep = 0;

                nbr = LCApreprocessDepthFirst(tree, tree.root(), 0, &nbr, &rep);

                // Check that the number of nodes in the tree was correct
                std::size_t nbNodes = rep;
                std::size_t nbRepresent = 2 * nbNodes - 1;
                hg_assert((std::size_t) (nbr + 1) == nbRepresent, "oops incorrect number of nodes!");

                std::size_t logn = (long) (ceil(log((double) (nbRepresent)) / log(2.0)));

                Minim.resize({logn, nbRepresent});

                for (std::size_t i = 0; i < nbRepresent - 1; i++) {
                    if (Depth[Euler[i]] < Depth[Euler[i + 1]]) {
                        Minim(0, i) = i;
                    } else {
                        Minim(0, i) = i + 1;
                    }
                }
                Minim(0, nbRepresent - 1) = nbRepresent - 1;

                for (std::size_t j = 1; j < logn; j++) {
                    std::size_t k1 = 1 << (j - 1);
                    std::size_t k2 = k1 << 1;
                    for (std::size_t i = 0; i < nbRepresent; i++) {
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

                auto nbNodes = num_vertices(tree);

                Depth.resize({nbNodes});
                Number.resize({nbNodes});

                Euler.resize({2 * nbNodes - 1});
                Represent.resize({2 * nbNodes - 1});

                LCApreprocess(tree);
            }

            vertex_t lca(vertex_t n1, vertex_t n2) {
                long ii, jj, kk, k;

                ii = Number[n1];
                jj = Number[n2];
                if (ii == jj)
                    return Represent[ii];

                if (ii > jj) {
                    kk = jj;
                    jj = ii;
                    ii = kk;
                }

                k = (long) (log((double) (jj - ii)) / log(2.));

                if (Depth[Euler[Minim(k, ii)]] < Depth[Euler[Minim(k, jj - (1 << (k)))]]) {
                    return Represent[Number[Euler[Minim(k, ii)]]];
                } else {
                    return Represent[Number[Euler[Minim(k, jj - (1 << k))]]];
                }
            }

            template<typename T>
            auto lca(const T &range) {
                std::size_t size = range.end() - range.begin();
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