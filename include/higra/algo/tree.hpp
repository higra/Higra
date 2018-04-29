//
// Created by perretb on 29/04/18.
//

#pragma once

#include "xtensor/xgenerator.hpp"
#include "xtensor/xeval.hpp"
#include "../structure/tree_graph.hpp"

namespace hg {

/**
 * Test if 2 trees are isomorph assuming that they share the same leaves.
 *
 * By this definition t1 is isomorph to t2 if there exist a bijection f from vertices(t1) to vertices(t2) such that:
 *   1) for any leaf node n of t1, f(n) = n and n
 *   2) for any node n of t1, f(t1.parent(n)) = t2.parent(f(n))
 *
 * Note that the root r node of a tree t is defined by t.parent(r) = r, thus 2) becomes
 *   for the root node r1 of t1, f(r1) = t2.parent(f(r1)), i.e. f(r1) is the root node of t2
 *
 * @tparam tree1_t
 * @tparam tree2_t
 * @param t1
 * @param t2
 * @return
 */
    template<typename tree1_t, typename tree2_t>
    bool testTreeIsomorphism(const tree1_t &t1, const tree2_t &t2) {
        if (t1.num_vertices() != t2.num_vertices() || t1.num_leaves() != t2.num_leaves())
            return false;

        long num_v = t1.num_vertices();
        long num_l = t1.num_leaves();
        auto not_defined = num_v;

        auto f = xt::eval(not_defined + xt::zeros<long>({num_v}));


        for (auto i: t1.iterate_from_leaves_to_root()) {
            if (i < num_l)
                f[i] = i; // 1) for any leaf node n of t1, f(n) = n and n

            auto t1pi = t1.parent(i); // t1.parent(n)
            long t2pfi = t2.parent(f(i)); //t2.parent(f(n))

            if (f(t1pi) == not_defined) { // f(t1.parent(n)) not defined
                f(t1pi) = t2pfi; // f(t1.parent(n)) = t2.parent(f(n))
            } else if (f(t1pi) != t2pfi) { // f(t1.parent(n)) was already set by a brother of n and is not compatible
                return false;
            }// else ok
        }
        return true;
    }

}
