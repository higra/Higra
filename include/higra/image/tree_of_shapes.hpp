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
#include "xtensor/xview.hpp"
#include "xtensor/xnoalias.hpp"
#include <map>
#include <deque>

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

            auto min_level() const {
                return m_min_level;
            }

            auto max_level() const {
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
                return m_data[level - min_level()].front();
            }

            /**
            * Return a const reference to the last element of the given queue level
            * @param level in [min_level, max_level]
            * @return a const reference to a value_type element
            */
            const auto &top(level_type level) const {
                return m_data[level - min_level()].front();
            }

            /**
             * Removes the last element of the given queue level
             * @param level in [min_level, max_level]
             */
            void pop(level_type level) {
                m_data[level - min_level()].pop_front();
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

                level_type level_low = level;
                level_type level_high = level;
                bool flag_low = true;
                bool flag_high = true;

                while (flag_low || flag_high) {
                    if (flag_low) {
                        if (!level_empty(level_low)) {
                            return level_low;
                        }
                        if (level_low == m_min_level) {
                            flag_low = false;
                        } else {
                            level_low--;
                        }
                    }
                    if (flag_high) {
                        if (!level_empty(level_high)) {
                            return level_high;
                        }
                        if (level_high == m_max_level) {
                            flag_high = false;
                        } else {
                            level_high++;
                        }
                    }
                }
                throw std::runtime_error("Empty queue!");
                return (level_type) 0;
            }

        private:
            level_t m_min_level;
            level_t m_max_level;
            index_t m_num_levels;
            std::vector<std::deque<value_type>> m_data;
            index_t m_size = 0;
        };

        template<typename T, typename value_type=typename T::value_type>
        auto interpolate_plain_map_khalimsky_2d(const xt::xexpression<T> &ximage, const embedding_grid_2d &embedding) {
            auto &image = ximage.derived_cast();
            size_t h = embedding.shape()[0];
            size_t w = embedding.shape()[1];
            size_t h2 = h * 2 - 1;
            size_t w2 = w * 2 - 1;


            array_2d<value_type> plain_map = array_2d<value_type>::from_shape({h2 * w2, 2});
            const auto image2d = xt::reshape_view(image, {h, w});
            auto plain_map2d = xt::reshape_view(plain_map, {h2, w2, (size_t) 2});

            // 2 faces
            xt::noalias(xt::view(plain_map2d, xt::range(0, h2, 2), xt::range(0, w2, 2), 0)) = image2d;
            xt::noalias(xt::view(plain_map2d, xt::range(0, h2, 2), xt::range(0, w2, 2), 1)) = image2d;

            // horizontal 1 face
            xt::noalias(xt::view(plain_map2d, xt::range(0, h2, 2), xt::range(1, w2, 2), 0)) =
                    xt::minimum(xt::view(image2d, xt::all(), xt::range(0, w - 1)),
                                xt::view(image2d, xt::all(), xt::range(1, w)));
            xt::noalias(xt::view(plain_map2d, xt::range(0, h2, 2), xt::range(1, w2, 2), 1)) =
                    xt::maximum(xt::view(image2d, xt::all(), xt::range(0, w - 1)),
                                xt::view(image2d, xt::all(), xt::range(1, w)));

            // vertical 1 face
            xt::noalias(xt::view(plain_map2d, xt::range(1, h2, 2), xt::range(0, w2, 2), 0)) =
                    xt::minimum(xt::view(image2d, xt::range(0, h - 1), xt::all()),
                                xt::view(image2d, xt::range(1, h), xt::all()));
            xt::noalias(xt::view(plain_map2d, xt::range(1, h2, 2), xt::range(0, w2, 2), 1)) =
                    xt::maximum(xt::view(image2d, xt::range(0, h - 1), xt::all()),
                                xt::view(image2d, xt::range(1, h), xt::all()));

            // 0 face
            xt::noalias(xt::view(plain_map2d, xt::range(1, h2, 2), xt::range(1, w2, 2), 0)) =
                    xt::minimum(
                            xt::minimum(xt::view(image2d, xt::range(0, h - 1), xt::range(0, w - 1)),
                                        xt::view(image2d, xt::range(0, h - 1), xt::range(1, w))),
                            xt::minimum(xt::view(image2d, xt::range(1, h), xt::range(0, w - 1)),
                                        xt::view(image2d, xt::range(1, h), xt::range(1, w))));
            xt::noalias(xt::view(plain_map2d, xt::range(1, h2, 2), xt::range(1, w2, 2), 1)) =
                    xt::maximum(
                            xt::maximum(xt::view(image2d, xt::range(0, h - 1), xt::range(0, w - 1)),
                                        xt::view(image2d, xt::range(0, h - 1), xt::range(1, w))),
                            xt::maximum(xt::view(image2d, xt::range(1, h), xt::range(0, w - 1)),
                                        xt::view(image2d, xt::range(1, h), xt::range(1, w))));
            return plain_map;
        }

        template<typename graph_t,
                typename T,
                typename value_type = typename T::value_type,
                typename std::enable_if_t<sizeof(value_type) <= 2 && std::is_integral<value_type>::value, int> = 0>
        auto sort_vertices_tree_of_shapes(const graph_t &graph,
                                          const xt::xexpression<T> &xplain_map, index_t exterior_vertex = 0) {

            auto &plain_map = xplain_map.derived_cast();
            auto num_v = num_vertices(graph);
            array_1d<bool> dejavu({num_v}, false);
            array_1d<index_t> sorted_vertex_indices = array_1d<index_t>::from_shape({num_v});
            array_1d<value_type> enqueued_level = array_1d<value_type>::from_shape({num_v});
            integer_level_multi_queue<value_type, index_t> queue(xt::amin(plain_map)(), xt::amax(plain_map)());

            value_type current_level = (value_type) ((plain_map(exterior_vertex, 0) + plain_map(exterior_vertex, 1)) /
                                                     2.0);
            queue.push(current_level, exterior_vertex);
            dejavu(exterior_vertex) = true;

            index_t i = 0;
            while (!queue.empty()) {
                current_level = queue.find_closest_non_empty_level(current_level);
                auto current_point = queue.top(current_level);
                queue.pop(current_level);
                enqueued_level(current_point) = current_level;
                sorted_vertex_indices(i++) = current_point;
                for (auto n: adjacent_vertex_iterator(current_point, graph)) {
                    if (!dejavu(n)) {
                        auto newLevel = (std::min)(plain_map(n, 1), (std::max)(plain_map(n, 0), current_level));
                        queue.push(newLevel, n);
                        dejavu(n) = true;
                    }
                }

            }
            return std::make_pair(std::move(sorted_vertex_indices), std::move(enqueued_level));
        }

        template<typename graph_t,
                typename T,
                typename value_type = typename T::value_type,
                typename std::enable_if_t<3 <= sizeof(value_type) || !std::is_integral<value_type>::value, int> = 0>
        auto sort_vertices_tree_of_shapes(const graph_t &graph,
                                          const xt::xexpression<T> &xplain_map, index_t exterior_vertex = 0) {

            auto &plain_map = xplain_map.derived_cast();
            auto num_v = num_vertices(graph);
            array_1d<bool> dejavu({num_v}, false);
            array_1d<index_t> sorted_vertex_indices = array_1d<index_t>::from_shape({num_v});
            array_1d<value_type> enqueued_level = array_1d<value_type>::from_shape({num_v});

            std::multimap<value_type, index_t> queue{};
            auto find_closest_non_empty_level = [&queue](const auto position) {
                auto next = std::next(position);
                if (position == queue.begin()) {
                    return next;
                }
                auto prev = std::prev(position);
                if (next == queue.end()) {
                    return prev;
                }
                if (next->first - position->first < position->first - prev->first) {
                    return next;
                } else {
                    return prev;
                }


            };

            value_type current_level = (value_type) ((plain_map(exterior_vertex, 0) + plain_map(exterior_vertex, 1)) /
                                                     2.0);

            auto position = queue.insert({current_level, exterior_vertex});
            dejavu(exterior_vertex) = true;

            index_t i = 0;
            do {
                current_level = position->first;
                index_t current_point = position->second;

                enqueued_level(current_point) = current_level;
                sorted_vertex_indices(i++) = current_point;
                for (auto n: adjacent_vertex_iterator(current_point, graph)) {
                    if (!dejavu(n)) {
                        auto newLevel = (std::min)(plain_map(n, 1), (std::max)(plain_map(n, 0), current_level));
                        queue.insert({newLevel, n});
                        dejavu(n) = true;
                    }
                }

                auto new_position = find_closest_non_empty_level(position);
                queue.erase(position);
                position = new_position;
            } while (!queue.empty());
            return std::make_pair(std::move(sorted_vertex_indices), std::move(enqueued_level));
        }

    }

}
