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

#include <vector>
#include "../utils.hpp"

namespace hg {

    namespace union_find_internal {

        template<typename idx_t=index_t>
        struct union_find {

        public:

            union_find(size_t size = 0) : parent(size), rank(size) {
                for (index_t i = 0; i < (index_t)parent.size(); ++i) {
                    parent[i] = i;
                }
            }

            idx_t make_set() {
                idx_t i = parent.size();
                parent.push_back(i);
                rank.push_back(0);
                return i;
            }


            idx_t find(idx_t element) {
                idx_t i = element;
                // find canonical node i
                while (parent[i] != i)
                    i = parent[i];
                // path compression
                while (parent[element] != i) {
                    idx_t tmp = element;
                    element = parent[element];
                    parent[tmp] = i;
                }
                return i;
            }

            /**
             * Union by rank
             * @param i index of canonical node
             * @param j index of canonical node
             * @return index of the canonical node representing the union of i and j (either i or j)
             */
            idx_t link(idx_t i, idx_t j) {
                if (rank[i] > rank[j])
                    std::swap(i, j);
                else if (rank[i] == rank[j]) {
                    rank[j] += 1;
                }
                parent[i] = j;
                return j;
            }

        private:
            using container_t = std::vector<idx_t>;

            container_t parent;
            container_t rank;
        };
    }

    using union_find = union_find_internal::union_find<>;

}
