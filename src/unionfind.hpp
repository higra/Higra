//
// Created by user on 4/5/18.
//

#pragma once

#include <vector>

namespace hg {

    namespace union_find_internal {

        template<typename idx_t=std::size_t>
        struct union_find {

        public:

            union_find(std::size_t size = 0) : parent(size), rank(size) {
                for (std::size_t i = 0; i < parent.size(); ++i) {
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
