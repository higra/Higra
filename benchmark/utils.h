/***************************************************************************
* Copyright ESIEE Paris (2021)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once
#include "higra/graph.hpp"
#include <cstdint>
#include <random>
#include <utility>

hg::tree get_complete_binary_tree(std::size_t num_leaves);

namespace bench_utils {

    /**
     * Random engine shared by the benchmarks; reseed with random_seed.
     */
    std::mt19937_64 &random_engine();

    inline void random_seed(std::uint64_t seed) {
        random_engine().seed(seed);
    }

    /**
     * Array of the given shape filled with uniform values in [0, 1).
     */
    template<typename T>
    hg::array<T> random_uniform(const hg::array_shape &shape) {
        std::uniform_real_distribution<T> d(T(0), T(1));
        hg::array<T> a(shape);
        for (auto &v: a) {
            v = d(random_engine());
        }
        return a;
    }

    /**
     * Array of the given shape filled with standard normal values.
     */
    template<typename T>
    hg::array<T> random_normal(const hg::array_shape &shape) {
        std::normal_distribution<T> d(T(0), T(1));
        hg::array<T> a(shape);
        for (auto &v: a) {
            v = d(random_engine());
        }
        return a;
    }

    /**
     * Sources and targets of the edges of the 4-adjacency graph of a size x size grid: horizontal edges
     * first, then vertical edges, each in row-major order.
     */
    std::pair<hg::array<hg::index_t>, hg::array<hg::index_t>> grid_4adj_edges(hg::index_t size);
}
