/***************************************************************************
* Copyright ESIEE Paris (2021)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "utils.h"
#include <cassert>

using namespace hg;

bool is_power_of_two(std::size_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

hg::tree get_complete_binary_tree(std::size_t num_leaves) {
    assert(is_power_of_two(num_leaves));
    array_1d<std::size_t> parent = array_1d<std::size_t>::from_shape({num_leaves * 2 - 1});
    for (std::size_t i = 0, j = num_leaves; i < (std::size_t) parent.size() - 1; j++) {
        parent(i++) = j;
        parent(i++) = j;
    }
    parent(parent.size() - 1) = parent.size() - 1;
    return tree(std::move(parent));
}
namespace bench_utils {

    std::mt19937_64 &random_engine() {
        static std::mt19937_64 engine(42);
        return engine;
    }

    std::pair<array_1d<index_t>, array_1d<index_t>> grid_4adj_edges(index_t size) {
        const index_t num_h = size * (size - 1);
        array_1d<index_t> sources = array_1d<index_t>::from_shape({2 * num_h});
        array_1d<index_t> targets = array_1d<index_t>::from_shape({2 * num_h});
        index_t k = 0;
        for (index_t i = 0; i < size; i++) {
            for (index_t j = 0; j < size - 1; j++, k++) {
                sources(k) = i * size + j;
                targets(k) = i * size + j + 1;
            }
        }
        for (index_t i = 0; i < size - 1; i++) {
            for (index_t j = 0; j < size; j++, k++) {
                sources(k) = i * size + j;
                targets(k) = (i + 1) * size + j;
            }
        }
        return {std::move(sources), std::move(targets)};
    }
}
