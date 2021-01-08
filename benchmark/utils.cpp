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

using namespace hg;

bool is_power_of_two(std::size_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

hg::tree get_complete_binary_tree(std::size_t num_leaves) {
    assert(is_power_of_two(num_leaves));
    array_1d<std::size_t> parent = array_1d<std::size_t>::from_shape({num_leaves * 2 - 1});
    for (std::size_t i = 0, j = num_leaves; i < parent.size() - 1; j++) {
        parent(i++) = j;
        parent(i++) = j;
    }
    parent(parent.size() - 1) = parent.size() - 1;
    return tree(std::move(parent));
}