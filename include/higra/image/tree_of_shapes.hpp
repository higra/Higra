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

#include "higra/graph.hpp"
#include <map>
#include <vector>

namespace hg {

    namespace tree_of_shapes_internal {

        /**
         * A simple multi-level priority queue with fixed number of integer levels in [min_level, nax_level].
         *
         * All operations are done in constant time, except:
         * - constructor, and
         * - find_closest_non_empty_level
         * which both run in O(num_levels = max_level - min_level + 1).
         *
         * @paramt value_t type of sored values
         */
        template<typename level_t, typename value_t>
        struct integer_level_multi_queue {
            using value_type = value_t;
            using level_type = level_t;

            /**
             * Create a queue with the given number of levels
             * @param num_levels queue will have integer levels in [0, num_levels[
             */
            integer_level_multi_queue(index_t min_level, index_t max_level) :
                    m_min_level(min_level),
                    m_max_level(max_level),
                    m_num_levels(max_level - min_level + 1),
                    m_data(m_num_levels) {
            }

            auto min_level() const{
                return m_min_level;
            }

            auto max_level() const{
                return m_max_level;
            }

            /**
             *
             * @return number of levels in the queue
             */
            auto num_levels() const {
                return m_num_levels;
            }

            /**
             *
             * @return number of elements in the queue
             */
            auto size() const {
                return m_size;
            }

            /**
             *
             * @return true if the queue is empty
             */
            auto empty() const {
                return m_size == 0;
            }

            /**
             *
             * @param level in [min_level, max_level]
             * @return true if the given level of the queue is empty
             */
            auto level_empty(level_type level) const {
                return m_data[level - min_level()].size() == 0;
            }

            /**
             * Add a new element to the given level of the queue
             * @param level in [min_level, max_level]
             * @param v new element
             */
            void push(level_type level, value_type v) {
                m_data[level - min_level()].push_back(v);
                m_size++;
            }

            /**
             * Return a reference to the last element of the given queue level
             * @param level in [min_level, max_level]
             * @return a reference to a value_type element
             */
            auto &top(level_type level) {
                return m_data[level - min_level()].back();
            }

            /**
            * Return a const reference to the last element of the given queue level
            * @param level in [min_level, max_level]
            * @return a const reference to a value_type element
            */
            const auto &top(level_type level) const {
                return m_data[level - min_level()].back();
            }

            /**
             * Removes the last element of the given queue level
             * @param level in [min_level, max_level]
             */
            void pop(level_type level) {
                m_data[level - min_level()].pop_back();
                m_size--;
            }

            /**
             * Given a queue level, find the closest non empty level in the queue.
             * In case of equality the smallest level is returned.
             *
             * @param level in [min_level, max_level]
             * @return a queue level or hg::invalid_index if the queue is empty
             */
            auto find_closest_non_empty_level(level_type level) const {
                if (!level_empty(level)) {
                    return level;
                }

                auto level_low = level - 1;
                auto level_high = level + 1;

                while (level_low >= m_min_level || level_high < m_max_level) {
                    if (level_low >= m_min_level) {
                        if (!level_empty(level_low)) {
                            return level_low;
                        }
                        level_low--;
                    }
                    if (level_high < m_max_level) {
                        if (!level_empty(level_high)) {
                            return level_high;
                        }
                        level_high++;
                    }
                }
                throw std::runtime_error("Empty queue!");
            }

        private:
            level_t m_min_level;
            level_t m_max_level;
            index_t m_num_levels;
            std::vector<std::vector<value_type>> m_data;
            index_t m_size = 0;
        };
        
    }

}
