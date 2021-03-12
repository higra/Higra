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

#include "higra/structure/array.hpp"
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse64)
#endif

namespace hg {

    namespace range_minimum_query_internal {
        /*
         * The 2 following classes rmq_sparse_table and rmq_sparse_table_block are freely adapted from
         * https://github.com/wx-csy/librmq (release 2.0 on Jul 28, 2019, commit 32bac30f1a1e1debf482a0477098bd0db203d849)
         * published under the MIT license reproduced hereafter:
         *
         * MIT License
         *
         * Copyright (c) 2019 Shaoyuan CHEN
         *
         * Permission is hereby granted, free of charge, to any person obtaining a copy
         * of this software and associated documentation files (the "Software"), to deal
         * in the Software without restriction, including without limitation the rights
         * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
         * copies of the Software, and to permit persons to whom the Software is
         * furnished to do so, subject to the following conditions:
         *
         * The above copyright notice and this permission notice shall be included in all
         * copies or substantial portions of the Software.
         *
         * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
         * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
         * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
         * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
         * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
         * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
         * SOFTWARE.
         *
         */


        /**
         * RMQ based on sparse table :
         * - O(n log(n)) preprocessing
         * - O(1) query
         * @tparam data_t
         */
        template<typename data_t>
        struct rmq_sparse_table {
            using self_type = rmq_sparse_table<data_t>;

            rmq_sparse_table() {

            }

            template<typename T>
            rmq_sparse_table(const T &values) : m_data(values.data()) {
                index_t size = values.size();
                reserve_sparse_table(size);
                m_sparse_table.push_back(xt::arange<size_t>(0, size));
                init_sparse_table();
            }

            template<typename T>
            rmq_sparse_table(const T &values, array_1d <size_t> &&element_map) : m_data(values.data()) {
                reserve_sparse_table(element_map.size());
                m_sparse_table.push_back(std::move(element_map));
                init_sparse_table();
            }

            template<typename T>
            rmq_sparse_table(const T &values, const array_1d <size_t> &element_map) : m_data(values.data()) {
                reserve_sparse_table(element_map.size());
                m_sparse_table.push_back(element_map);
                init_sparse_table();
            }

            /**
             * Precondition l < r
             * @param l
             * @param r
             * @return
             */
            index_t query(index_t l, index_t r) const {
                auto level = fast_log2(r - l);
                auto p1 = m_sparse_table[level](l);
                auto p2 = m_sparse_table[level](r - (1 << level));
                return m_data[p1] < m_data[p2] ? p1 : p2;
            }

            template<template<typename> typename container_t>
            struct internal_state {
                using type = self_type;
                using array_t = container_t<size_t>;
                using sp_t = std::vector<array_t>;
                sp_t sparse_table;

                internal_state(const sp_t &_sparse_table) :
                        sparse_table(_sparse_table) {}

                internal_state(sp_t && _sparse_table) :
                        sparse_table(std::move(_sparse_table)) {}
            };

            auto get_state() const {
                return internal_state<array_1d>(m_sparse_table);
            }

            template<template<typename> typename container_t, typename T>
            static auto make_from_state(internal_state<container_t> &&state, const T &data) {
                rmq_sparse_table<typename T::value_type> rmq;
                rmq.set_state(std::move(state), data);
                return rmq;
            }

            template<template<typename> typename container_t, typename T>
            static auto make_from_state(const internal_state<container_t> &state, const T &data) {
                rmq_sparse_table<typename T::value_type> rmq;
                rmq.set_state(state, data);
                return rmq;
            }

        private:

            template<template<typename> typename container_t, typename T>
            void set_state(internal_state<container_t> &&state, const T &data) {
                m_sparse_table.clear();
                for(auto & e: state.sparse_table){
                    m_sparse_table.push_back(std::move(e));
                }
                m_data = data.begin();
            }

            template<template<typename> typename container_t, typename T>
            void set_state(const internal_state<container_t> &state, const T &data) {
                m_sparse_table.clear();
                for(auto & e: state.sparse_table){
                    m_sparse_table.push_back(e);
                }
                m_data = data.begin();
            }



            /**
             * Precondition: length > 0
             * @param length
             * @return
             */
            static inline size_t fast_log2(size_t length) {
#ifdef _MSC_VER
                unsigned long most_significant_bit_index = 0;
                _BitScanReverse64(&most_significant_bit_index, length);
                return most_significant_bit_index;
#else
                return sizeof(size_t) * 8 - 1 - __builtin_clzll(length);
#endif
            }



            void reserve_sparse_table(size_t size) {
                m_sparse_table.reserve((size_t) ceil(log((double) (size)) / log(2.0)));
            }

            void init_sparse_table() {
                index_t size = m_sparse_table[0].size();

                for (index_t lvl = 0; (2 << lvl) <= size; lvl++) {
                    index_t size_lvlp1 = size - (2 << lvl) + 1;
                    m_sparse_table.push_back(array_1d<index_t>::from_shape({(size_t) size_lvlp1}));
                    parfor(0, size_lvlp1, [this, lvl](index_t i) {
                        auto p1 = m_sparse_table[lvl](i);
                        auto p2 = m_sparse_table[lvl](i + (1 << lvl));
                        m_sparse_table[lvl + 1](i) = m_data[p1] < m_data[p2] ? p1 : p2;
                    });
                }
            }

            std::vector<array_1d < size_t>> m_sparse_table;
            const data_t *m_data;
        };

        /**
         * RMQ based on sparse table on blocks,
         * - O(n) preprocessing (if block size is in  O(log(n)))
         * - average O(1) query (for uniformly distributed queries)
         * @tparam data_t
         */
        template<typename data_t>
        struct rmq_sparse_table_block {

            using self_type = rmq_sparse_table_block<data_t>;

            rmq_sparse_table_block() {

            }

            template<typename T>
            rmq_sparse_table_block(const T &values, size_t block_size = 1024) : m_data(values.data()),
                                                                               m_block_size(block_size) {
                hg_assert(block_size > 0, "Block size must be strictly positive");
                m_data_size = values.size();
                m_num_blocks = (m_data_size + m_block_size - 1) / m_block_size;
                m_block_minimum_prefix.resize({(size_t) (m_num_blocks * m_block_size)});
                m_block_minimum_suffix.resize({(size_t) (m_num_blocks * m_block_size)});
                init(values);
            }

            /**
             * Precondition l < r
             * @param l
             * @param r
             * @return
             */
            index_t query(index_t l, index_t r) const {


                index_t lb = l / m_block_size;
                index_t rb = r / m_block_size;
                if (lb != rb) { // speculative policy
                    index_t ret = m_sparse_table.query(lb, std::min(m_num_blocks, rb + 1));
                    if (ret >= l && ret < r) return ret;
                } else {
                    if (m_block_minimum_prefix(r) >= l)
                        return m_block_minimum_prefix(r);
                    if (m_block_minimum_suffix(l) <= r)
                        return m_block_minimum_suffix(l);
                    return std::min_element(m_data + l, m_data + r) - m_data;
                }
                //index_t lbase = lb * m_block_size;
                index_t rbase = rb * m_block_size;
                index_t v = m_block_minimum_suffix(l);
                if (r != rbase) {
                    index_t v2 = m_block_minimum_prefix(r - 1);
                    if (m_data[v2] < m_data[v]) v = v2;
                }
                if (lb + 1 == rb) return v;
                index_t vv = m_sparse_table.query(lb + 1, rb);
                return m_data[vv] < m_data[v] ? vv : v;
            }

            template<template<typename> typename container_t>
            struct internal_state {
                using type = self_type;
                using sp_state_type = typename rmq_sparse_table<data_t>::template internal_state<container_t>;
                index_t data_size;
                index_t block_size;
                index_t num_blocks;
                container_t<index_t> block_minimum_prefix;
                container_t<index_t> block_minimum_suffix;
                sp_state_type sparse_table;

                internal_state(index_t _data_size,
                               index_t _block_size,
                               index_t _num_blocks,
                               container_t<index_t> &&_block_minimum_prefix,
                               container_t<index_t> &&_block_minimum_suffix,
                               sp_state_type &&_sp_state) :
                        data_size(_data_size),
                        block_size(_block_size),
                        num_blocks(_num_blocks),
                        block_minimum_prefix(std::move(_block_minimum_prefix)),
                        block_minimum_suffix(std::move(_block_minimum_suffix)),
                        sparse_table(std::move(_sp_state)) {}

                internal_state(index_t _data_size,
                               index_t _block_size,
                               index_t _num_blocks,
                               const container_t<index_t> &_block_minimum_prefix,
                               const container_t<index_t> &_block_minimum_suffix,
                               const sp_state_type &_sp_state) :
                        data_size(_data_size),
                        block_size(_block_size),
                        num_blocks(_num_blocks),
                        block_minimum_prefix(_block_minimum_prefix),
                        block_minimum_suffix(_block_minimum_suffix),
                        sparse_table(_sp_state) {}
            };

            auto get_state() const {
                return internal_state<array_1d>(m_data_size,
                        m_block_size,
                        m_num_blocks,
                        m_block_minimum_prefix,
                        m_block_minimum_suffix,
                        m_sparse_table.get_state());
            }

            template<template<typename> typename container_t, typename T>
            static auto make_from_state(internal_state<container_t> &&state, const T &data) {
                rmq_sparse_table_block<typename T::value_type> rmq;
                rmq.set_state(std::move(state), data);
                return rmq;
            }

            template<template<typename> typename container_t, typename T>
            static auto make_from_state(const internal_state<container_t> &state, const T &data) {
                rmq_sparse_table_block<typename T::value_type> rmq;
                rmq.set_state(state, data);
                return rmq;
            }

        private:

            template<template<typename> typename container_t, typename T>
            void set_state(internal_state<container_t> &&state, const T &data) {
                m_data_size = state.data_size;
                m_block_size = state.block_size;
                m_num_blocks = state.num_blocks;
                m_block_minimum_prefix = std::move(state.block_minimum_prefix);
                m_block_minimum_suffix = std::move(state.block_minimum_suffix);
                m_sparse_table = rmq_sparse_table<typename T::value_type>::make_from_state(
                        std::move(state.sparse_table), data);
                m_data = data.begin();
            }

            template<template<typename> typename container_t, typename T>
            void set_state(const internal_state<container_t> &state, const T &data) {
                m_data_size = state.data_size;
                m_block_size = state.block_size;
                m_num_blocks = state.num_blocks;
                m_block_minimum_prefix = state.block_minimum_prefix;
                m_block_minimum_suffix = state.block_minimum_suffix;
                m_sparse_table = rmq_sparse_table<typename T::value_type>::make_from_state(state.sparse_table, data);
                m_data = data.begin();
            }

            template<typename T>
            void init(const T &values) {

                /*
                 * Blocks preprocessing
                 */
                array_1d<index_t> element_map = array_1d<index_t>::from_shape({(size_t) m_num_blocks});
                parfor(0, m_num_blocks, [&element_map, this](index_t i) {
                    index_t block_start = i * m_block_size;
                    index_t block_end = std::min(block_start + m_block_size, m_data_size);

                    // smallest element position in the i-th block
                    element_map(i) = std::min_element(m_data + block_start, m_data + block_end) - m_data;

                    // minimum prefix block
                    index_t current_minimum_index = block_start;
                    index_t current_minimum = m_data[current_minimum_index];

                    m_block_minimum_prefix(block_start) = current_minimum_index;
                    for (index_t j = block_start + 1; j < block_start + m_block_size; ++j) {
                        if (j < block_end && m_data[j] < current_minimum) {
                            current_minimum = m_data[j];
                            current_minimum_index = j;
                        }
                        m_block_minimum_prefix(j) = current_minimum_index;
                    }

                    // minimum suffix block
                    current_minimum_index = block_start + m_block_size - 1;
                    current_minimum = (current_minimum_index < block_end) ? m_data[current_minimum_index] : -1;

                    m_block_minimum_suffix(block_start + m_block_size - 1) = current_minimum_index;
                    for (index_t j = block_start + m_block_size - 2; j >= block_start; --j) {
                        if (j < block_end && m_data[j] < current_minimum) {
                            current_minimum = m_data[j];
                            current_minimum_index = j;
                        }
                        m_block_minimum_suffix(j) = current_minimum_index;
                    }
                });

                m_sparse_table = rmq_sparse_table<index_t>(values, std::move(element_map));
            }

            const data_t *m_data;
            index_t m_data_size;
            index_t m_block_size;
            index_t m_num_blocks;
            array_1d <index_t> m_block_minimum_prefix;
            array_1d <index_t> m_block_minimum_suffix;
            rmq_sparse_table<index_t> m_sparse_table;

        };
    }
}
