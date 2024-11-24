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
#include "xtensor/xio.hpp"

#include <map>
#include <deque>

namespace hg {

    /**
     * Padding mode for the function component_tree_tree_of_shapes
     */
    enum tos_padding {
        none,
        mean,
        zero
    };


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

        template<typename T1, typename T2>
        auto fill_khalimsky_plane_2d(const xt::xexpression<T1> &ximage2d, xt::xexpression<T2> &xplain_map2d) {
            auto &image2d = ximage2d.derived_cast();
            auto &plain_map2d = xplain_map2d.derived_cast();
            size_t h = image2d.shape(0);
            size_t w = image2d.shape(1);


            // 2 faces and horizontal 1 face
            for(size_t y = 0; y < h; y++){
                plain_map2d(2 * y, 0, 0) = image2d(y, 0);
                plain_map2d(2 * y, 0, 1) = image2d(y, 0);
                for(size_t x = 1; x < w; x++){
                    plain_map2d(2 * y, 2 * x, 0) = image2d(y, x);
                    plain_map2d(2 * y, 2 * x, 1) = image2d(y, x);
                    plain_map2d(2 * y, 2 * x - 1, 0) = std::min(image2d(y, x - 1), image2d(y, x));
                    plain_map2d(2 * y, 2 * x - 1, 1) = std::max(image2d(y, x - 1), image2d(y, x));
                }
            }

            for(size_t y = 1; y < 2 * h - 2; y+=2){
                for(size_t x = 0; x < 2 * w - 1; x++){
                    plain_map2d(y, x, 0) = std::min(plain_map2d(y - 1, x, 0), plain_map2d(y + 1, x, 0));
                    plain_map2d(y, x, 1) = std::max(plain_map2d(y - 1, x, 1), plain_map2d(y + 1, x, 1));
                }
            }
        }

        /*template<typename T, typename value_type=typename T::value_type>
        auto interpolate_plain_map_khalimsky_2d(const xt::xexpression<T> &ximage, const embedding_grid_2d &embedding, xt::xexpression<T> &xplain_map) {
            auto &image = ximage.derived_cast();
            auto &plain_map = xplain_map.derived_cast();
            size_t h = embedding.shape()[0];
            size_t w = embedding.shape()[1];
            size_t h2 = h * 2 - 1;
            size_t w2 = w * 2 - 1;

            //array_2d<value_type> plain_map = array_2d<value_type>::from_shape({h2 * w2, 2});
            const auto image2d = xt::reshape_view(image, {h, w});
            auto plain_map2d = xt::reshape_view(plain_map, {h2, w2, (size_t) 2});

            fill_khalimsky_plane_2d(image2d, plain_map2d);
            return plain_map;
        }*/

        /**
        * Interpolate khalimsky function for 3D images.
        *
        * @param ximage    a container for image values
        * @param embedding the associated embedding grid
        * 
        * @return a projection of the image in the Khalimsky grid
        */
        template<typename T, typename T2>
        auto interpolate_plain_map_khalimsky_3d(const xt::xexpression<T> &ximage3d, const embedding_grid_3d &embedding, xt::xexpression<T2> &xplain_map3d) {
            auto &image3d = ximage3d.derived_cast();
            auto &plain_map3d = xplain_map3d.derived_cast();
            size_t d = embedding.shape()[0];
            size_t h = embedding.shape()[1];
            size_t w = embedding.shape()[2];

            size_t d2 = d * 2 - 1;
            size_t h2 = h * 2 - 1;
            size_t w2 = w * 2 - 1;

            // go over x-y odd planes and fill them as 2D Khalimsky plane
            for(size_t z = 0; z < d; z++){
                auto plain_map2d = xt::view(plain_map3d, 2 * z, xt::all(), xt::all(), xt::all());
                fill_khalimsky_plane_2d(xt::view(image3d, z, xt::all(), xt::all()), plain_map2d);
            }

            // go over x-y even planes and fill them
            for(size_t z = 1; z < d2 - 1 ; z += 2){
                for(size_t y = 0; y < h2; y++){
                    for(size_t x = 0; x < w2; x++){
                        plain_map3d(z, y, x, 0) = std::min(plain_map3d(z - 1, y, x, 0), plain_map3d(z + 1, y, x, 0));
                        plain_map3d(z, y, x, 1) = std::max(plain_map3d(z - 1, y, x, 1), plain_map3d(z + 1, y, x, 1));
                    }
                }
            }
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

        template<typename T>
        auto get_padding_value(const xt::xexpression<T> &ximage, tos_padding padding) {
            auto &image = ximage.derived_cast();
            using value_type = typename T::value_type;
            auto dim = image.dimension();


            value_type pad_value = 0;
            switch (padding) {
                case tos_padding::zero:
                    pad_value = 0;
                    break;
                case tos_padding::mean:
                    // compute mean of image boundaries
                    if (dim == 2){
                        size_t h = image.shape(0);
                        size_t w = image.shape(1);
                        size_t num_elements = w;
                        for(size_t x = 0; x < w; x++){
                            pad_value += image(0, x);
                        }
                        if(h > 1){
                            for(size_t x = 0; x < w; x++){
                                pad_value += image(h - 1, x);
                            }
                            num_elements += w;
                        }
                        if(h > 2){
                            for(size_t y = 1; y < h - 1; y++){
                                pad_value += image(y, 0);
                            }
                            num_elements += (h - 2);
                            if (w > 1){
                                for(size_t y = 1; y < h - 1; y++){
                                    pad_value += image(y, w - 1);
                                }
                                num_elements += (h - 2);
                            }
                        }
                        pad_value /= num_elements;
                    }else if(dim == 3){
                        size_t d = image.shape(0);
                        size_t h = image.shape(1);
                        size_t w = image.shape(2);
                        size_t num_elements = w * h;
                        for(size_t y = 0; y < h; y++){
                            for(size_t x = 0; x < w; x++){
                                pad_value += image(0, y, x);
                            }
                        }
                        if(d > 1){
                            for(size_t y = 0; y < h; y++){
                                for(size_t x = 0; x < w; x++){
                                    pad_value += image(d - 1, y, x);
                                }
                            }
                            num_elements += w * h;
                        }
                        if(d > 2){
                            for(size_t z = 1; z < d - 1; z++){
                                for(size_t y = 0; y < h; y++){
                                    pad_value += image(z, y, 0);
                                }
                            }
                            num_elements += (d - 2) * h;
                            if (w > 1){
                                for(size_t z = 1; z < d - 1; z++){
                                    for(size_t y = 0; y < h; y++){
                                        pad_value += image(z, y, w - 1);
                                    }
                                }
                                num_elements += (d - 2) * h;
                            }
                            for(size_t z = 1; z < d - 1; z++){
                                for(size_t x = 1; x < w - 1; x++){
                                    pad_value += image(z, 0, x);
                                }
                            }
                            num_elements += (d - 2) * (w - 2);
                            if (h > 1){
                                for(size_t z = 1; z < d - 1; z++){
                                    for(size_t x = 1; x < w - 1; x++){
                                        pad_value += image(z, h - 1, x);
                                    }
                                }
                                num_elements += (d - 2) * (w - 2);
                            }
                        }
                        pad_value /= num_elements;
                    }else{
                        throw std::runtime_error("Unsupported dimension");
                    }

                    break;
                case tos_padding::none:
                    throw std::runtime_error("Padding value requested for no padding.");
            }
            return pad_value;
        }

        template <typename T, typename value_type=typename T::value_type>
        void fill_padding(xt::xexpression<T> & xplain_map, const value_type padding_value, bool immersion, bool is_input3d){
            auto & plain_map = xplain_map.derived_cast();
            auto shape = plain_map.shape();
            size_t d = shape[0];
            size_t h = shape[1];
            size_t w = shape[2];

            if(is_input3d) {
                // fill all 6 faces of the cube with padding value
                // front and back faces
                for (size_t y = 0; y < h; y++) {
                    for (size_t x = 0; x < w; x++) {
                        plain_map(0, y, x, 0) = padding_value;
                        plain_map(0, y, x, 1) = padding_value;
                        plain_map(d - 1, y, x, 0) = padding_value;
                        plain_map(d - 1, y, x, 1) = padding_value;
                    }
                }
                // top and bottom faces
                for (size_t z = 1; z < d - 1; z++) {
                    for (size_t x = 0; x < w; x++) {
                        plain_map(z, 0, x, 0) = padding_value;
                        plain_map(z, 0, x, 1) = padding_value;
                        plain_map(z, h - 1, x, 0) = padding_value;
                        plain_map(z, h - 1, x, 1) = padding_value;
                    }
                }
                // left and right faces
                for (size_t z = 1; z < d - 1; z++) {
                    for (size_t y = 1; y < h - 1; y++) {
                        plain_map(z, y, 0, 0) = padding_value;
                        plain_map(z, y, 0, 1) = padding_value;
                        plain_map(z, y, w - 1, 0) = padding_value;
                        plain_map(z, y, w - 1, 1) = padding_value;
                    }
                }

                if (immersion) {
                    // interpolate inner faces
                    // front and back faces
                    for (size_t y = 2; y < h - 2; y++) {
                        for (size_t x = 2; x < w - 2; x++) {
                            plain_map(1, y, x, 0) = std::min(plain_map(0, y, x, 0), plain_map(2, y, x, 0));
                            plain_map(1, y, x, 1) = std::max(plain_map(0, y, x, 1), plain_map(2, y, x, 1));
                            plain_map(d - 2, y, x, 0) = std::min(plain_map(d - 1, y, x, 0), plain_map(d - 3, y, x, 0));
                            plain_map(d - 2, y, x, 1) = std::max(plain_map(d - 1, y, x, 1), plain_map(d - 3, y, x, 1));
                        }
                    }
                    // left and right faces
                    for (size_t z = 1; z < d - 1; z++) {
                        for (size_t y = 2; y < h - 2; y++) {
                            plain_map(z, y, 1, 0) = std::min(plain_map(z, y, 0, 0), plain_map(z, y, 2, 0));
                            plain_map(z, y, 1, 1) = std::max(plain_map(z, y, 0, 1), plain_map(z, y, 2, 1));
                            plain_map(z, y, w - 2, 0) = std::min(plain_map(z, y, w - 1, 0), plain_map(z, y, w - 3, 0));
                            plain_map(z, y, w - 2, 1) = std::max(plain_map(z, y, w - 1, 1), plain_map(z, y, w - 3, 1));
                        }
                    }

                    // top and bottom faces
                    for (size_t z = 1; z < d - 1; z++) {
                        for (size_t x = 1; x < w - 1; x++) {
                            plain_map(z, 1, x, 0) = std::min(plain_map(z, 0, x, 0), plain_map(z, 2, x, 0));
                            plain_map(z, 1, x, 1) = std::max(plain_map(z, 0, x, 1), plain_map(z, 2, x, 1));
                            plain_map(z, h - 2, x, 0) = std::min(plain_map(z, h - 1, x, 0), plain_map(z, h - 3, x, 0));
                            plain_map(z, h - 2, x, 1) = std::max(plain_map(z, h - 1, x, 1), plain_map(z, h - 3, x, 1));
                        }
                    }
                }
            } else {
                // fill border of x y plane
                for(size_t x = 0; x < w; x++){
                    plain_map(0, 0, x, 0) = padding_value;
                    plain_map(0, 0, x, 1) = padding_value;
                    plain_map(0, h - 1, x, 0) = padding_value;
                    plain_map(0, h - 1, x, 1) = padding_value;
                }
                for(size_t y = 1; y < h - 1; y++){
                    plain_map(0, y, 0, 0) = padding_value;
                    plain_map(0, y, 0, 1) = padding_value;
                    plain_map(0, y, w - 1, 0) = padding_value;
                    plain_map(0, y, w - 1, 1) = padding_value;
                }
                if (immersion){
                    // interpolate inner border of x y plane
                    for(size_t x = 2; x < w - 2; x++){
                        plain_map(0, 1, x, 0) = std::min(plain_map(0, 0, x, 0), plain_map(0, 2, x, 0));
                        plain_map(0, 1, x, 1) = std::max(plain_map(0, 0, x, 1), plain_map(0, 2, x, 1));
                        plain_map(0, h - 2, x, 0) = std::min(plain_map(0, h - 1, x, 0), plain_map(0, h - 3, x, 0));
                        plain_map(0, h - 2, x, 1) = std::max(plain_map(0, h - 1, x, 1), plain_map(0, h - 3, x, 1));
                    }
                    for(size_t y = 1; y < h - 1; y++){
                        plain_map(0, y, 1, 0) = std::min(plain_map(0, y, 0, 0), plain_map(0, y, 2, 0));
                        plain_map(0, y, 1, 1) = std::max(plain_map(0, y, 0, 1), plain_map(0, y, 2, 1));
                        plain_map(0, y, w - 2, 0) = std::min(plain_map(0, y, w - 1, 0), plain_map(0, y, w - 3, 0));
                        plain_map(0, y, w - 2, 1) = std::max(plain_map(0, y, w - 1, 1), plain_map(0, y, w - 3, 1));
                    }
                }
            }
        }
    }

    /**
     * Computes the tree of shapes of an image.
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
     * In practice if the size of the input image is (h, w, d), the leaves of the returned tree will correspond to an image of size:
     *   - (h, w, d) if original_size is true;
     *   - (h * 2 - 1, w * 2 - 1, d * 2 - 1) is original_size is false and padding is tos_padding::none; and
     *   - ((h + 2) * 2 - 1, (w + 2) * 2 - 1, (d + 2) * 2 - 1) otherwise.
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
     * @param ximage Must be a 2d or 3d array
     * @param padding Defines if an extra boundary of pixels is added to the original image (see enum tos_padding).
     * @param original_size remove all nodes corresponding to interpolated/padded pixels
     * @param exterior_vertex linear coordinate of the exterior point
     * @return a node weighted tree
     */
    template<typename T>
    auto component_tree_tree_of_shapes_image(const xt::xexpression<T> &ximage,
                                             tos_padding padding = tos_padding::mean,
                                             bool original_size = true,
                                             bool immersion = true,
                                             index_t exterior_vertex = 0) {
        HG_TRACE();
        auto &image = ximage.derived_cast();
        const int dim = image.dimension();
        hg_assert(dim == 2 || dim == 3, "image must be a 2d or 3d array");

        using value_type = typename T::value_type;

        bool is_input_3d = dim == 3;

        auto shape = image.shape();

        size_t d = (is_input_3d) ? shape[0] : 1;
        size_t h = (is_input_3d) ? shape[1] : shape[0];
        size_t w = (is_input_3d) ? shape[2] : shape[1];

        auto image3d = xt::reshape_view(image, {d, h, w});
        auto shape3d = image3d.shape();

        // ----------------
        // Compute intermediate plain map representation size
        // ----------------
        bool do_padding = padding != tos_padding::none;
        size_t padding_size = (do_padding) ? 1 : 0;
        size_t immersion_factor = (immersion)? 2 : 1;
        size_t border_size_hw = padding_size * immersion_factor;
        size_t border_size_d = (is_input_3d)? border_size_hw : 0;

        size_t d_plain_map;
        if(is_input_3d){
            d_plain_map = (immersion)? (d + padding_size * 2) * 2 - 1 : d + padding_size * 2;
        }else{
            d_plain_map = 1;
        }
        size_t h_plain_map = (immersion)? (h + padding_size * 2) * 2 - 1 : h + padding_size * 2;
        size_t w_plain_map = (immersion)? (w + padding_size * 2) * 2 - 1 : w + padding_size * 2;


        // ----------------
        // Compute plain map, do Khalimsky interpolation if needed, then fill padding is needed
        // ----------------
        array_4d<value_type> plain_map({d_plain_map, h_plain_map, w_plain_map, (size_t)2});

        auto plain_map_interior = xt::view(plain_map, xt::range(border_size_d, d_plain_map - border_size_d),
                                           xt::range(border_size_hw, h_plain_map - border_size_hw),
                                           xt::range(border_size_hw, w_plain_map - border_size_hw), xt::all());

        if(immersion){
            tree_of_shapes_internal::interpolate_plain_map_khalimsky_3d(image3d, embedding_grid_3d(shape3d), plain_map_interior);
        } else {
            for(size_t z = 0; z < d; z++){
                for(size_t y = 0; y < h; y++){
                    for(size_t x = 0; x < w; x++){
                        plain_map_interior(z, y, x, 0) = image3d(z, y, x);
                        plain_map_interior(z, y, x, 1) = image3d(z, y, x);
                    }
                }
            }
        }

        if(do_padding){
            auto padding_value = tree_of_shapes_internal::get_padding_value(image, padding);
            tree_of_shapes_internal::fill_padding(plain_map, padding_value, immersion, is_input_3d);
        }

        // ----------------
        // Sort vertices with flooding from the exterior vertex and then compute the associated component tree
        // ----------------
        auto graph = get_6_adjacency_implicit_graph({(index_t) d_plain_map, (index_t) h_plain_map, (index_t) w_plain_map});

        auto res_sort = tree_of_shapes_internal::sort_vertices_tree_of_shapes(
                graph,
                xt::reshape_view(plain_map, {d_plain_map * h_plain_map * w_plain_map, (size_t)2}),
                exterior_vertex);
        auto &sorted_vertex_indices = res_sort.first;
        auto &enqueued_levels = res_sort.second;

        auto res_tree = component_tree_internal::tree_from_sorted_vertices(graph, enqueued_levels, sorted_vertex_indices);

        // ----------------
        // Remove nodes corresponding to padding and Khalimsky interpolation if needed
        // ----------------
        if (!original_size || (!immersion && padding == tos_padding::none)) {
            return res_tree;
        }

        auto &tree = res_tree.tree;
        auto &altitudes = res_tree.altitudes;


        array_1d<bool> deleted_vertices({num_leaves(res_tree.tree)}, true);
        auto deleted = xt::reshape_view(deleted_vertices, {d_plain_map, h_plain_map, w_plain_map});
        xt::view(deleted, xt::range(border_size_d, d_plain_map - border_size_d, immersion_factor),
                 xt::range(border_size_hw, h_plain_map - border_size_hw, immersion_factor),
                 xt::range(border_size_hw, w_plain_map - border_size_hw, immersion_factor)) = false;

        auto all_deleted = accumulate_sequential(tree, deleted_vertices, accumulator_min());

        auto stree = simplify_tree(tree, all_deleted, true);
        array_1d<value_type> simplified_altitudes = xt::index_view(altitudes, stree.node_map);
        return make_node_weighted_tree(std::move(stree.tree), std::move(simplified_altitudes));

    }

    /**
     * @deprecated use component_tree_tree_of_shapes_image instead
     * @see description of component_tree_tree_of_shapes_image
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
        return component_tree_tree_of_shapes_image(image, padding, original_size, immersion, exterior_vertex);
    }


    /**
     * @deprecated use component_tree_tree_of_shapes_image instead
     * @see description of component_tree_tree_of_shapes_image
     */
    template<typename T>
    auto component_tree_tree_of_shapes_image3d(const xt::xexpression<T> &ximage,
                                               tos_padding padding = tos_padding::mean,
                                               bool original_size = true,
                                               bool immersion = true,
                                               index_t exterior_vertex = 0) {
        HG_TRACE();
        auto &image = ximage.derived_cast();
        hg_assert(image.dimension() == 3, "image must be a 3d array");
        return component_tree_tree_of_shapes_image(image, padding, original_size, immersion, exterior_vertex);
    }


};

