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
#include "higra/image/graph_image.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xindex_view.hpp"

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
            integer_level_multi_queue(level_type min_level, level_type max_level) :
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
            hg_assert(plain_map.dimension() == 2, "Invalid plain map");
            hg_assert(plain_map.shape()[1] == 2, "Invalid plain map");
            hg_assert_vertex_weights(graph, plain_map);
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

    /**
     * Padding mode for the function component_tree_tree_of_shapes
     */
    enum tos_padding {
        none,
        mean,
        zero
    };

    /**
     * Computes the tree of shapes of a 2d image.
     * The Tree of Shapes was described in [1].
     *
     * The algorithm used in this implementation was first described in [2].
     *
     * The tree is computed in the interpolated multivalued Khalimsky space to provide a continuous and autodual representation of
     * input image.
     *
     * If padding is different from tos_padding::none, an extra border of pixels is added to the input image before
     * anything else. This will ensure the existence of a shape encompassing all the shapes inside the input image
     * (if exterior_vertex is inside the extra border): this shape will be the root of the tree.
     * The padding value can be:
     *   - 0 is padding == tos_padding::zero
     *   - the mean value of the boundary pixels of the input image if padding == tos_padding::mean
     *
     * If original_size is true, all the nodes corresponding to pixels not belonging to the input image are removed
     * (except for the root node).
     * If original_size is false, the returned tree is the tree constructed in the interpolated/padded space.
     * In practice if the size of the input image is (h, w), the leaves of the returned tree will correspond to an image of size:
     *   - (h, w) if original_size is true;
     *   - (h * 2 - 1, w * 2 - 1) is original_size is false and padding is tos_padding::none; and
     *   - ((h + 2) * 2 - 1, (w + 2) * 2 - 1) otherwise.
     *
     * :Advanced options:
     * 
     * Use with care the following options may lead to unexpected results:
     * 
     * Immersion defines if the initial image should be first converted as an equivalent continuous representation called a
     * plain map. If the immersion is deactivated the level lines of the shapes of the image may intersect (if the image is not
     * well composed) and the result of the algorithm is undefined. If immersion is deactivated, the factor :math:`*2 - 1`
     * has to be removed in the result sizes given above.
     * 
     * Exterior_vertex defines the linear coordinates of the pixel corresponding to the exterior (interior and exterior
     * of a shape is defined with respect to this point). The coordinate of this point must be given in the
     * padded/interpolated space.
     *
     * [1] Pa. Monasse, and F. Guichard, "Fast computation of a contrast-invariant image representation,"
     *     Image Processing, IEEE Transactions on, vol.9, no.5, pp.860-872, May 2000
     *
     * [2] Th. Géraud, E. Carlinet, S. Crozet, and L. Najman, "A Quasi-linear Algorithm to Compute the Tree
     *     of Shapes of nD Images", ISMM 2013.
     *
     * @tparam T
     * @param ximage Must be a 2d array
     * @param padding Defines if an extra boundary of pixels is added to the original image (see enum tos_padding).
     * @param original_size remove all nodes corresponding to interpolated/padded pixels
     * @param exterior_vertex linear coordinate of the exterior point
     * @return a node weighted tree
     */
    template<typename T>
    auto component_tree_tree_of_shapes_image2d(const xt::xexpression<T> &ximage,
                                               tos_padding padding = tos_padding::mean,
                                               bool original_size = true,
                                               bool immersion = true,
                                               index_t exterior_vertex = 0) {
        HG_TRACE();
        auto &image = ximage.derived_cast();
        hg_assert(image.dimension() == 2, "image must be a 2d array");
        embedding_grid_2d embedding(image.shape());
        auto shape = embedding.shape();
        size_t h = shape[0];
        size_t w = shape[1];
        auto vertex_weights = xt::flatten(image);
        using value_type = typename T::value_type;

        size_t rh;
        size_t rw;

        array_2d<value_type> cooked_vertex_values;

        auto do_padding = [&padding, &h, &w](const auto &image) {
            value_type pad_value;
            switch (padding) {
                case tos_padding::zero:
                    pad_value = 0;
                    break;
                case tos_padding::mean: {
                    auto tmp = xt::sum(xt::view(image, 0, xt::all()))() +
                               xt::sum(xt::view(image, h - 1, xt::all()))();
                    if (h > 2) {
                        tmp += xt::sum(xt::view(image, xt::range(1, h - 1), 0))() +
                               xt::sum(xt::view(image, xt::range(1, h - 1), w - 1))();
                    }
                    pad_value = (value_type) (tmp / ((std::max)(2.0 * (w + h) - 4, 1.0)));
                    break;
                }
                case none:
                default:
                    throw std::runtime_error("Incorrect padding value.");
            }
            array_1d<value_type> padded_vertices = array_1d<value_type>::from_shape({(w + 2) * (h + 2)});
            auto padded_image = xt::reshape_view(padded_vertices, {h + 2, w + 2});

            xt::noalias(xt::view(padded_image, xt::range(1, h + 1), xt::range(1, w + 1))) = image;
            xt::view(padded_image, 0, xt::all()) = pad_value;
            xt::view(padded_image, h + 1, xt::all()) = pad_value;
            xt::view(padded_image, xt::all(), 0) = pad_value;
            xt::view(padded_image, xt::all(), w + 1) = pad_value;
            return padded_vertices;
        };

        auto process_sorted_pixels = [&original_size, &padding, &rh, &rw, &immersion](auto &graph,
                                                                                      auto &sorted_vertex_indices,
                                                                                      auto &enqueued_levels) {
            auto res_tree = component_tree_internal::tree_from_sorted_vertices(graph, enqueued_levels,
                                                                               sorted_vertex_indices);
            auto &tree = res_tree.tree;
            auto &altitudes = res_tree.altitudes;

            if (!original_size || (!immersion && padding == tos_padding::none)) {
                return res_tree;
            }

            array_1d<bool> deleted_vertices({num_leaves(res_tree.tree)}, true);
            auto deleted = xt::reshape_view(deleted_vertices, {rh, rw});
            if (immersion) {
                if (padding != tos_padding::none) {
                    xt::view(deleted, xt::range(2, rh - 2, 2), xt::range(2, rw - 2, 2)) = false;
                } else {
                    xt::view(deleted, xt::range(0, rh, 2), xt::range(0, rw, 2)) = false;
                }
            } else {
                if (padding != tos_padding::none) {
                    xt::view(deleted, xt::range(1, rh - 1), xt::range(1, rw - 1)) = false;
                } // else handled by bypass if on top
            }

            auto all_deleted = accumulate_sequential(tree, deleted_vertices, accumulator_min());

            auto stree = simplify_tree(tree, all_deleted, true);
            array_1d<value_type> saltitudes = xt::index_view(altitudes, stree.node_map);
            return make_node_weighted_tree(std::move(stree.tree), std::move(saltitudes));
        };

        if (immersion) {
            if (padding != tos_padding::none) {


                auto cooked_vertex_values =
                        tree_of_shapes_internal::interpolate_plain_map_khalimsky_2d(
                                do_padding(image),
                                {(index_t) (h + 2), (index_t) (w + 2)});
                rh = (h + 2) * 2 - 1;
                rw = (w + 2) * 2 - 1;
                auto graph = get_4_adjacency_implicit_graph({(index_t) rh, (index_t) rw});
                auto res_sort = tree_of_shapes_internal::sort_vertices_tree_of_shapes(graph, cooked_vertex_values,
                                                                                      exterior_vertex);
                return process_sorted_pixels(graph, res_sort.first, res_sort.second);
            } else {
                auto cooked_vertex_values =
                        tree_of_shapes_internal::interpolate_plain_map_khalimsky_2d(
                                vertex_weights,
                                embedding);
                rh = h * 2 - 1;
                rw = w * 2 - 1;
                auto graph = get_4_adjacency_implicit_graph({(index_t) rh, (index_t) rw});
                auto res_sort = tree_of_shapes_internal::sort_vertices_tree_of_shapes(graph, cooked_vertex_values,
                                                                                      exterior_vertex);
                return process_sorted_pixels(graph, res_sort.first, res_sort.second);
            }
        } else {
            if (padding != tos_padding::none) {
                auto padded_vertices = do_padding(image);

                rh = h + 2;
                rw = w + 2;
                auto graph = get_4_adjacency_implicit_graph({(index_t) rh, (index_t) rw});

                // clearly not optimal
                array_2d<value_type> plain_map = array_2d<value_type>::from_shape({rh * rw, 2});
                xt::noalias(xt::view(plain_map, xt::all(), 0)) = padded_vertices;
                xt::noalias(xt::view(plain_map, xt::all(), 1)) = padded_vertices;
                auto res_sort = tree_of_shapes_internal::sort_vertices_tree_of_shapes(graph, plain_map,
                                                                                      exterior_vertex);
                return process_sorted_pixels(graph, res_sort.first, res_sort.second);
            } else {
                rh = h;
                rw = w;
                auto graph = get_4_adjacency_implicit_graph({(index_t) rh, (index_t) rw});
                array_2d<value_type> plain_map = array_2d<value_type>::from_shape({rh * rw, 2});
                xt::noalias(xt::view(plain_map, xt::all(), 0)) = xt::ravel(image);
                xt::noalias(xt::view(plain_map, xt::all(), 1)) = xt::ravel(image);
                auto res_sort = tree_of_shapes_internal::sort_vertices_tree_of_shapes(graph, plain_map,
                                                                                      exterior_vertex);
                return process_sorted_pixels(graph, res_sort.first, res_sort.second);
            }

        }


    }
};
