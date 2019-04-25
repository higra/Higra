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

#include "../graph.hpp"
#include "graph_image.hpp"
#include "../structure/details/iterators.hpp"
#include "../algo/graph_weights.hpp"
#include <stack>
#include <higra/algo/rag.hpp>

namespace hg {

    namespace contour_2d_internal {
        // forward declaration
        template<typename point_type>
        class polyline_contour_2d;

        template<typename point_type>
        class contour_segment_2d_iterator;

        template<typename point_type=point_2d_f>
        class contour_segment_2d {
            const polyline_contour_2d<point_type> &m_polyline;
            const index_t m_first_control_point;
            const index_t m_second_control_point;
            const index_t m_size;

        public:
            using value_type = std::pair<index_t, point_2d_f>;

            contour_segment_2d(const polyline_contour_2d<point_type> &polyline,
                               index_t first_control_point,
                               index_t second_control_point) :
                    m_polyline(polyline),
                    m_first_control_point(first_control_point),
                    m_second_control_point(second_control_point),
                    m_size(m_second_control_point - m_first_control_point + 1) {
            }


            const auto begin() const {
                return contour_segment_2d_iterator<point_type>(*this, 0);
            }

            const auto end() const {
                return contour_segment_2d_iterator<point_type>(*this, m_size);
            }

            decltype(auto) operator[](index_t i) const {
                return std::make_pair(m_polyline.m_contour_elements[i + m_first_control_point],
                                      m_polyline.m_contour_points[i + m_first_control_point]);
            }

            /**
             * First element of the contour segment
             * @return
             */
            decltype(auto) first() const {
                return (*this)[0];
            }

            /**
             * Last element of the contour segment
             * @return
             */
            decltype(auto) last() const {
                return (*this)[m_size - 1];
            }

            /**
             * Number of elements in the contour segment
             * @return
             */
            auto size() const {
                return m_size;
            }

            auto norm() const {
                auto v = first().second;
                auto w = last().second;
                return std::sqrt((v[0] - w[0]) * (v[0] - w[0]) + (v[1] - w[1]) * (v[1] - w[1]));
            };

            auto distance_to_point(const point_type &p) {
                auto v = first().second;
                auto w = last().second;
                auto l2 = std::sqrt((v[0] - w[0]) * (v[0] - w[0]) + (v[1] - w[1]) * (v[1] - w[1]));
                if (l2 == 0.0)
                    return std::sqrt((v[0] - p[0]) * (v[0] - p[0]) + (v[1] - p[1]) * (v[1] - p[1]));;   // v == w case
                return std::abs((w[0] - v[0]) * p[1] - (w[1] - v[1]) * p[0] + w[1] * v[0] - w[0] * v[1]) / l2;
            };

            auto angle() const {
                auto v = first().second;
                auto w = last().second;
                return std::atan2(v[0] - w[0], v[1] - w[1]);
            }

        };

        template<typename point_type=point_2d_f>
        class contour_segment_2d_iterator :
                public forward_iterator_facade<contour_segment_2d_iterator<point_type>,
                        typename contour_segment_2d<point_type>::value_type,
                        typename contour_segment_2d<point_type>::value_type> {
        public:

            contour_segment_2d_iterator(const contour_segment_2d<point_type> &segment, index_t position = 0) :
                    m_segment(segment),
                    m_position(position) {}

            void increment() {
                m_position++;
            }

            bool equal(contour_segment_2d_iterator<point_type> const &other) const {
                return this->m_position == other.m_position;
            }

            decltype(auto) dereference() const {
                return m_segment[m_position];
            }

        private:
            const contour_segment_2d<point_type> &m_segment;
            index_t m_position;

        };


        // forward declaration
        template<typename point_type>
        class polyline_contour_2d_iterator;

        /**
         * A polyline contour is a set of contour segments that represent a connected frontier between two regions.
         */
        template<typename point_type=point_2d_f>
        class polyline_contour_2d {
            std::vector<index_t> m_contour_elements;
            std::vector<point_type> m_contour_points;
            std::vector<index_t> m_control_points;

            friend class contour_segment_2d<point_type>;

        public:
            using value_type = contour_segment_2d<point_type>;

            polyline_contour_2d() {

            }

            void add_contour_element(index_t element, point_type coordinates) {
                m_contour_elements.push_back(element);
                m_contour_points.push_back(coordinates);
                if (m_contour_points.size() == 1) {
                    m_control_points.push_back(0);
                    m_control_points.push_back(0);
                } else {
                    m_control_points.back() = m_contour_points.size() - 1;
                }
            }

            auto operator[](index_t i) const {
                return contour_segment_2d<point_type>(*this, m_control_points[i], m_control_points[i + 1]);
            }

            auto size() const {
                return m_control_points.size() - 1;
            }

            const auto begin() const {
                return polyline_contour_2d_iterator<point_type>(*this, 0);
            }

            const auto end() const {
                return polyline_contour_2d_iterator<point_type>(*this, size());
            }

            auto number_of_contour_elements() const {
                return m_contour_elements.size();
            }

            /**
             * Subdivide the line such that the distance between the line
             * joining the extremities of the contour segment and each of its elements is lower than the threshold (
             * Ramer–Douglas–Peucker algorithm) aor smaller than the minimal specified size.
             *
             * The threshold is equal to
             *  - epsilon if relative_epsilon is false
             *  - epsilon times the distance between the segment extremities if relative_epsilon is true
             *
             * @param epsilon
             * @param relative_epsilon
             * @param minSize
             */
            void subdivide(double epsilon = 0.1,
                           bool relative_epsilon = true,
                           int min_size = 2) {

                // stack elements are the portions of the segment that have to be checked for subdivision
                stackv<std::pair<index_t, index_t>> stack;

                // if i-th element true the polyline has to be subdivided at this element
                std::vector<bool> is_subdivision_element(m_contour_elements.size(), false);

                for (index_t segment_index = 0; segment_index < (index_t) size(); segment_index++) {
                    stack.push({m_control_points[segment_index], m_control_points[segment_index + 1]});

                    // current segment points are preserved
                    is_subdivision_element[m_control_points[segment_index]] = true;
                    is_subdivision_element[m_control_points[segment_index + 1]] = true;


                    // recursive identification of subdivision elements
                    while (!stack.empty()) {
                        auto element_indexes = stack.top();
                        stack.pop();
                        auto first_element = element_indexes.first;
                        auto last_element = element_indexes.second;

                        // nothing to be done
                        if (last_element - first_element < 2)
                            continue;

                        contour_segment_2d<point_type> segment(*this, first_element, last_element);

                        auto norm_segment = segment.norm();

                        double distance_threshold;
                        if (relative_epsilon) {
                            distance_threshold = epsilon * norm_segment;
                        } else {
                            distance_threshold = epsilon;
                        }

                        auto max_distance = distance_threshold;
                        auto max_distance_element = invalid_index;

                        for (index_t i = first_element + 1; i < last_element; i++) {
                            auto &coordinate_element = m_contour_points[i];
                            auto d = segment.distance_to_point(coordinate_element);
                            if (d >= max_distance && d > min_size) {
                                max_distance = d;
                                max_distance_element = i;
                            }
                        }

                        if (max_distance_element != invalid_index) {
                            is_subdivision_element[max_distance_element] = true;
                            stack.push({first_element, max_distance_element});
                            stack.push({max_distance_element, last_element});
                        }
                    }

                    // final subdivision
                    m_control_points.clear();

                    for (index_t i = 0; i < (index_t) m_contour_elements.size(); i++) {
                        if (is_subdivision_element[i]) {
                            m_control_points.push_back(i);
                        }
                    }
                    if (m_control_points.size() == 1)
                        m_control_points.push_back(0);
                }

            }
        };

        template<typename point_type=point_2d_f>
        class polyline_contour_2d_iterator :
                public forward_iterator_facade<polyline_contour_2d_iterator<point_type>,
                        typename polyline_contour_2d<point_type>::value_type,
                        typename polyline_contour_2d<point_type>::value_type> {
        public:

            polyline_contour_2d_iterator(const polyline_contour_2d<point_type> &polyline, index_t position = 0) :
                    m_polyline(polyline),
                    m_position(position) {}

            void increment() {
                m_position++;
            }

            bool equal(polyline_contour_2d_iterator<point_type> const &other) const {
                return this->m_position == other.m_position;
            }

            decltype(auto) dereference() const {
                return m_polyline[m_position];
            }

        private:
            const polyline_contour_2d<point_type> &m_polyline;
            index_t m_position;

        };


        /**
         * A contour is a set of polyline contours that represent the frontiers separating regions.
         */
        template<typename point_type=point_2d_f>
        class contour_2d {
            std::vector<polyline_contour_2d<point_type>> m_polyline_contours;

        public:
            auto &new_polyline_contour_2d() {
                m_polyline_contours.emplace_back();
                return m_polyline_contours[m_polyline_contours.size() - 1];
            }

            auto size() {
                return m_polyline_contours.size();
            }

            auto begin() {
                return m_polyline_contours.begin();
            }

            auto end() {
                return m_polyline_contours.end();
            }

            const auto begin() const {
                return m_polyline_contours.begin();
            }

            const auto end() const {
                return m_polyline_contours.end();
            }

            auto operator[](index_t i) {
                return m_polyline_contours[i];
            }

            /**
             * Subdivide each polyline of the given contours such that the distance between the line
             * joining the extremities of the contour segment and each of its elements is lower than the threshold (
             * Ramer–Douglas–Peucker algorithm) or smaller than the minimal specified size
             *
             * The threshold is equal to
             *  - epsilon if relative_epsilon is false
             *  - epsilon times the distance between the segment extremities if relative_epsilon is true
             *
             * Implementation note: simply call subdivide on each polyline of the contour.
             *
             * @param epsilon
             * @param relative_epsilon
             * @param min_size
             */
            void subdivide(
                    double epsilon = 0.1,
                    bool relative_epsilon = true,
                    int min_size = 2) {
                for (auto &polyline: m_polyline_contours) {
                    polyline.subdivide(epsilon, relative_epsilon, min_size);
                }
            };

        };

    }

    using contour_2d = contour_2d_internal::contour_2d<point_2d_f>;

    using polyline_contour_2d = contour_2d_internal::polyline_contour_2d<point_2d_f>;

    using contour_segment_2d = contour_2d_internal::contour_segment_2d<point_2d_f>;

    /**
     * Construct a contour_2d object from a graph cut of a 2d image with a 4 adjacency (non zero edges are part of the cut).
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param embedding
     * @param xedge_weights
     * @return
     */
    template<typename graph_t, typename T>
    auto
    fit_contour_2d(const graph_t &graph,
                   const embedding_grid_2d &embedding,
                   const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        using point_type = point_2d_f;
        const auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);
        hg_assert(num_vertices(graph) == embedding.size(),
                  "Graph number of vertices does not match the size of the embedding.");

        contour_2d result;

        array_1d<index_t> positive_edge_index = xt::empty<index_t>({num_edges(graph)});
        for (index_t i = 0; i < (index_t) positive_edge_index.size(); i++) {
            if (edge_weights[i] > 0)
                positive_edge_index[i] = i;
            else positive_edge_index[i] = invalid_index;
        }

        auto contours_khalimsky = graph_4_adjacency_2_khalimsky(graph, embedding, positive_edge_index, true,
                                                                invalid_index);

        auto edge_coordinates = [&embedding, &graph](index_t edge_index) {
            auto &e = edge_from_index(edge_index, graph);
            auto s = source(e, graph);
            auto t = target(e, graph);
            point_type coordinates = embedding.lin2grid(s);
            if (s + 1 == t) { // horizontal edge
                coordinates[1] += 0.5;
            } else { // vertical edge
                coordinates[0] += 0.5;
            }
            return coordinates;
        };

        array_2d<bool> processed = xt::zeros<bool>(contours_khalimsky.shape());

        index_t height = contours_khalimsky.shape()[0];
        index_t width = contours_khalimsky.shape()[1];

        auto is_intersection = [&contours_khalimsky, &height, &width](
                index_t y,
                index_t x) {
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
                return true;
            int count = 0;
            if (contours_khalimsky(y, x - 1) != invalid_index)
                count++;
            if (contours_khalimsky(y, x + 1) != invalid_index)
                count++;
            if (contours_khalimsky(y - 1, x) != invalid_index)
                count++;
            if (contours_khalimsky(y + 1, x) != invalid_index)
                count++;
            return count > 2;
        };

        enum direction {
            NORTH, EAST, SOUTH, WEST
        };

        // store the result of explore_contour_part (below)
        std::vector<index_t> contour_part;

        auto explore_contour_part = [&contours_khalimsky, &processed, &is_intersection, &contour_part](
                index_t y,
                index_t x,
                direction dir) {
            //auto &polyline = result.new_polyline_contour_2d();
            contour_part.clear();
            direction previous = dir;
            bool flag;

            do {
                processed(y, x) = true;
                index_t edge_index = contours_khalimsky(y, x);
                contour_part.push_back(edge_index);
                //polyline.add_contour_element(edge_index, edge_coordinates(edge_index));
                if (x % 2 == 0) // horizontal edge
                {
                    if (previous == NORTH) {
                        y++;
                    } else {
                        y--;
                    }
                } else { // vertical edge
                    if (previous == WEST) {
                        x++;
                    } else {
                        x--;
                    }
                }

                flag = processed(y, x) || is_intersection(y, x);
                if (!flag) {
                    processed(y, x) = true;
                    if (previous != NORTH &&
                        contours_khalimsky(y - 1, x) != invalid_index) {
                        previous = SOUTH;
                        y--;
                    } else if (previous != EAST &&
                               contours_khalimsky(y, x + 1) != invalid_index) {
                        previous = WEST;
                        x++;
                    } else if (previous != SOUTH &&
                               contours_khalimsky(y + 1, x) != invalid_index) {
                        previous = NORTH;
                        y++;
                    } else if (previous != WEST &&
                               contours_khalimsky(y, x - 1) != invalid_index) {
                        previous = EAST;
                        x--;
                    }
                }
            } while (!flag);

        };

        auto add_contour_parts_to_polyline = [&contour_part, &edge_coordinates](polyline_contour_2d &polyline,
                                                                                bool reverse = false) {
            if (reverse) {
                for (auto edge_index = contour_part.rbegin(); edge_index != contour_part.rend(); edge_index++) {
                    polyline.add_contour_element(*edge_index, edge_coordinates(*edge_index));
                }
            } else {
                for (auto edge_index = contour_part.begin(); edge_index != contour_part.end(); edge_index++) {
                    polyline.add_contour_element(*edge_index, edge_coordinates(*edge_index));
                }
            }

        };

        for (index_t y = 0; y < height; y += 2) {
            for (index_t x = 0; x < width; x += 2) {
                auto edge_index = contours_khalimsky(y, x);
                if (edge_index != invalid_index && // is there a non zero edge around this 0 face
                    !processed(y, x)) { // if so did we already processed it ?
                    processed(y, x) = true;
                    if (is_intersection(y, x)) { // explore each polyline starting from this point
                        if (x != 0 && contours_khalimsky(y, x - 1) != invalid_index && !processed(y, x - 1)) {
                            explore_contour_part(y, x - 1, EAST);
                            auto &polyline = result.new_polyline_contour_2d();
                            add_contour_parts_to_polyline(polyline);
                        }
                        if (x != width - 1 && contours_khalimsky(y, x + 1) != invalid_index && !processed(y, x + 1)) {
                            explore_contour_part(y, x + 1, WEST);
                            auto &polyline = result.new_polyline_contour_2d();
                            add_contour_parts_to_polyline(polyline);
                        }
                        if (y != 0 && contours_khalimsky(y - 1, x) != invalid_index && !processed(y - 1, x)) {
                            explore_contour_part(y - 1, x, SOUTH);
                            auto &polyline = result.new_polyline_contour_2d();
                            add_contour_parts_to_polyline(polyline);
                        }
                        if (y != height - 1 && contours_khalimsky(y + 1, x) != invalid_index && !processed(y + 1, x)) {
                            explore_contour_part(y + 1, x, NORTH);
                            auto &polyline = result.new_polyline_contour_2d();
                            add_contour_parts_to_polyline(polyline);
                        }
                    } else { // explore the two ends of the polyline passing by this point and join them
                        auto &polyline = result.new_polyline_contour_2d();
                        bool first = true;
                        if (x != 0 && contours_khalimsky(y, x - 1) != invalid_index && !processed(y, x - 1)) {
                            explore_contour_part(y, x - 1, EAST);
                            //if(first){ // impossible at first case
                            add_contour_parts_to_polyline(polyline, true);
                            first = false;
                            //}else{
                            //    add_contour_parts_to_polyline(polyline);
                            //}

                        }
                        if (x != width - 1 && contours_khalimsky(y, x + 1) != invalid_index && !processed(y, x + 1)) {
                            explore_contour_part(y, x + 1, WEST);
                            if (first) {
                                add_contour_parts_to_polyline(polyline, true);
                                first = false;
                            } else {
                                add_contour_parts_to_polyline(polyline);
                            }
                        }
                        if (y != 0 && contours_khalimsky(y - 1, x) != invalid_index && !processed(y - 1, x)) {
                            explore_contour_part(y - 1, x, SOUTH);
                            if (first) {
                                add_contour_parts_to_polyline(polyline, true);
                                first = false;
                            } else {
                                add_contour_parts_to_polyline(polyline);
                            }
                        }
                        if (y != height - 1 && contours_khalimsky(y + 1, x) != invalid_index && !processed(y + 1, x)) {
                            explore_contour_part(y + 1, x, NORTH);
                            if (first) {
                                add_contour_parts_to_polyline(polyline, true);
                                first = false;
                            } else {
                                add_contour_parts_to_polyline(polyline);
                            }
                        }
                    }


                }
            }
        }

        return result;
    }

    /**
     * Estimate the vertex perimeter and the length of the frontier associated to the edges of a
     * region adjacency graph constructed on a 2d 4 adjacency graph.
     *
     * The region boundaries are simplified with Ramer–Douglas–Peucker algorithm and is controlled
     * by the parameters epsilon, relative_epsilon, min_size. See function subdivide of the class contour_2d for more information.
     *
     * @tparam graph_t
     * @param rag_graph Region Adjacency Graph
     * @param xvertex_map Vertex map of the rag_graph
     * @param xedge_map Edge map of the rag_graph
     * @param embedding 2d shape of the input graph
     * @param graph input graph on which the region adjacency graph has been build: must be a 4 adjacency graph whose shape correspond to the given embedding
     * @param epsilon larger epsilon values will provide stronger contour shapes simplification
     * @param relative_epsilon Is epsilon given in relative or absolute units
     * @param min_size Boundaries elements smaller than min_size will be deleted
     * @return a pair composed of two 1d arrays: vertex_perimeter and edge_length.
     */
    template<typename rag_graph_t, typename graph_t, typename T>
    auto rag_2d_vertex_perimeter_and_edge_length(
            const rag_graph_t &rag_graph,
            const xt::xexpression<T> &xvertex_map,
            const xt::xexpression<T> &xedge_map,
            const embedding_grid_2d &embedding,
            const graph_t &graph,
            double epsilon = 0.1,
            bool relative_epsilon = true,
            int min_size = 2) {

        const auto &vertex_map = xvertex_map.derived_cast();
        const auto &edge_map = xedge_map.derived_cast();
        hg_assert_edge_weights(graph, edge_map);
        hg_assert_1d_array(edge_map);
        hg_assert_integral_value_type(edge_map);
        hg_assert_vertex_weights(graph, vertex_map);
        hg_assert_1d_array(vertex_map);
        hg_assert_integral_value_type(vertex_map);

        auto cut = weight_graph(graph, vertex_map, weight_functions::L0);
        auto contour2d = fit_contour_2d(graph, embedding, cut);

        contour2d.subdivide(epsilon, relative_epsilon, min_size);

        array_1d<double> vertex_perimeter = xt::zeros<double>({num_vertices(rag_graph)});
        array_1d<double> edge_length = xt::zeros<double>({num_edges(rag_graph)});

        for (auto &polyline: contour2d) {

            for (auto &segment: polyline) {
                auto segment_length = segment.norm() + 1;
                auto rag_edge_index = edge_map(segment.first().first);
                auto rag_edge = edge_from_index(rag_edge_index, rag_graph);
                edge_length(rag_edge_index) += segment_length;
                vertex_perimeter(source(rag_edge, rag_graph)) += segment_length;
                vertex_perimeter(target(rag_edge, rag_graph)) += segment_length;
            }
        }

        index_t height = embedding.shape()[0];
        index_t width = embedding.shape()[1];

        for (index_t x = 0; x < width; x++) {
            vertex_perimeter(vertex_map(embedding.grid2lin({0, x})))++;
            vertex_perimeter(vertex_map(embedding.grid2lin({height - 1, x})))++;
        }
        for (index_t y = 0; y < height; y++) {
            vertex_perimeter(vertex_map(embedding.grid2lin({y, 0})))++;
            vertex_perimeter(vertex_map(embedding.grid2lin({y, width - 1})))++;
        }

        return std::make_pair(std::move(vertex_perimeter), std::move(edge_length));
    }

    /**
     * Estimate the vertex perimeter and the length of the frontier associated to the edges of a
     * region adjacency graph constructed on a 2d 4 adjacency graph.
     *
     * The region boundaries are simplified with Ramer–Douglas–Peucker algorithm and is controlled
     * by the parameters epsilon, relative_epsilon, min_size. See function subdivide of the class contour_2d for more information.
     *
     * @tparam graph_t
     * @param rag Region Adjacency Graph
     * @param embedding 2d shape of the input graph
     * @param graph input graph on which the region adjacency graph has been build: must be a 4 adjacency graph whose shape correspond to the given embedding
     * @param epsilon larger epsilon values will provide stronger contour shapes simplification
     * @param relative_epsilon Is epsilon given in relative or absolute units
     * @param min_size Boundaries elements smaller than min_size will be deleted
     * @return a pair composed of two 1d arrays: vertex_perimeter and edge_length.
     */
    template<typename graph_t>
    auto rag_2d_vertex_perimeter_and_edge_length(
            const region_adjacency_graph &rag,
            const embedding_grid_2d &embedding,
            const graph_t &graph,
            double epsilon = 0.1,
            bool relative_epsilon = true,
            int min_size = 2) {
        return rag_2d_vertex_perimeter_and_edge_length(
                rag.rag,
                rag.vertex_map,
                rag.edge_map,
                embedding,
                graph,
                epsilon,
                relative_epsilon,
                min_size
        );
    }
}