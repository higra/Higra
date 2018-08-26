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
#include <stack>

namespace hg {

    // forward declaration
    template <typename point_type>
    class polyline_contour_2d;

    template <typename point_type>
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

        /*

          auto & add_segment() {
              m_contour_segments.emplace_back();
              return m_contour_segments.back();
          }

          auto & add_segment(contour_segment_2d & segment) {
              m_contour_segments.push_back(segment);
              return m_contour_segments.back();
          }

          auto & add_segment(contour_segment_2d && segment) {
              m_contour_segments.push_back(std::forward<contour_segment_2d>(segment));
              return m_contour_segments.back();
          }*/
/*
        template<typename input_iterator>
        auto & add_segment(input_iterator begin, input_iterator end) {
            m_contour_segments.emplace_back(begin, end);
            return m_contour_segments.back();
        }

        void concatenate(polyline_contour_2d & polyline){
            for(auto & segment: polyline){
                add_segment(segment);
            }
        }

        void concatenate(polyline_contour_2d && polyline){
            for(auto & segment: polyline){
                add_segment(std::move(segment));
            }
            polyline.clear();
        }

        auto size() const{
            return m_contour_segments.size();
        }

        auto number_of_contour_elements() const{
            size_t count = 0;
            for(const auto & segment: m_contour_segments){
                count += segment.size();
            }
            return count;
        }



        decltype(auto) operator[](index_t i){
            return m_contour_segments[i];
        }

        void clear(){
            m_contour_segments.clear();
        }*/
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

        /*auto &add_polyline_contour_2d(polyline_contour_2d &polyline) {
            m_polyline_contours.push_back(polyline);
            return m_polyline_contours.back();
        }

        auto &add_polyline_contour_2d(polyline_contour_2d &&polyline) {
            m_polyline_contours.push_back(std::forward<polyline_contour_2d>(polyline));
            return m_polyline_contours.back();
        }*/

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
    };

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
        hg_assert(edge_weights.dimension() == 1, "Edge weights must be scalar.");
        hg_assert(num_edges(graph) == edge_weights.size(),
                  "Edge weights size does not match the number of edge in the graph.");
        hg_assert(num_vertices(graph) == embedding.size(),
                  "Graph number of vertices does not match the size of the embedding.");

        contour_2d<point_type> result;

        array_1d<index_t> positive_edge_index = xt::empty<index_t>({num_edges(graph)});
        for (index_t i = 0; i < positive_edge_index.size(); i++) {
            if (edge_weights[i] > 0)
                positive_edge_index[i] = i;
            else positive_edge_index[i] = invalid_index;
        }

        auto contours_khalimsky = graph_4_adjacency_2_khalimsky(graph, embedding, positive_edge_index, true,
                                                                invalid_index);

        auto edge_coordinates = [&embedding, &graph](index_t edge_index) {
            auto e = edge(edge_index, graph);
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

        auto height = contours_khalimsky.shape()[0];
        auto width = contours_khalimsky.shape()[1];

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

        auto explore_contour_part = [&result, &contours_khalimsky, &processed, &is_intersection, &edge_coordinates](
                index_t y,
                index_t x,
                direction dir) {
            auto &polyline = result.new_polyline_contour_2d();

            direction previous = dir;
            bool flag;

            do {
                processed(y, x) = true;
                index_t edge_index = contours_khalimsky(y, x);
                polyline.add_contour_element(edge_index, edge_coordinates(edge_index));
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

                flag = is_intersection(y, x);
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

        for (index_t y = 0; y < height; y += 2) {
            for (index_t x = 0; x < width; x += 2) {
                auto edge_index = contours_khalimsky(y, x);
                if (edge_index != invalid_index && // is there a non zero edge around this 0 face
                    !processed(y, x) && // if so did we already processed it ?
                    is_intersection(y, x)) {
                    processed(y, x) = true;
                    if (x != 0 && contours_khalimsky(y, x - 1) != invalid_index && !processed(y, x - 1)) {
                        explore_contour_part(y, x - 1, EAST);
                    }
                    if (x != width - 1 && contours_khalimsky(y, x + 1) != invalid_index && !processed(y, x + 1)) {
                        explore_contour_part(y, x + 1, WEST);
                    }
                    if (y != 0 && contours_khalimsky(y - 1, x) != invalid_index && !processed(y - 1, x)) {
                        explore_contour_part(y - 1, x, SOUTH);
                    }
                    if (y != height - 1 && contours_khalimsky(y + 1, x) != invalid_index && !processed(y + 1, x)) {
                        explore_contour_part(y + 1, x, NORTH);
                    }

                }
            }
        }

        return result;
    }

    /**
    * Subdivide the contour segment such that the distance between the line
    * joining the extremities of the contour segment and each of its elements is lower than the threshold (
    * Ramer–Douglas–Peucker algorithm)
    *
    * The threshold is equal to
    *  - epsilon if relative_epsilon is false
    *  - epsilon times the distance between the segment extremities if relative_epsilon is true
    *
    * If the distance between the segment extremities is smaller than minSize, the segment is never subdivided.
    * @tparam graph_t
    * @tparam embedding_t
    * @param polyline
    * @param graph
    * @param embedding
    * @param epsilon
    * @param relative_epsilon
    * @param minSize
    * @return
    */
/*
    template<typename graph_t, typename embedding_t>
    auto subdivide_contour(const contour_segment_2d &segment,
                           const graph_t &graph,
                           const embedding_t &embedding,
                           double epsilon = 0.05,
                           bool relative_epsilon = true,
                           int minSize = 2) {
        using point_type = point_2d_f;
        //auto num_elements = polyline.number_of_contour_elements();
        //std::vector<point_type> points(num_elements);
        auto norm = [](const point_type & v, const point_type & w){
            return std::sqrt((v[0]-w[0])*(v[0]-w[0]) + (v[1]-w[1])*(v[1]-w[1]));
        };


        // Return minimum distance between line vw and point p
        auto distance_to_line = [&norm](const point_type & v, const point_type & w, const point_type & p){
            auto l2 = norm(v, w);
            if (l2 == 0.0) return norm(p, v);   // v == w case
            return std::abs((w[0]-v[0])*p[1] - (w[1] - v[1])*p[0] + w[1]*v[0] - w[0]*v[1]) / l2;
        };

        // stack elements are the portions of the segment that have to be checked for subdivision
        std::stack<std::pair<index_t, index_t>> stack;
        stack.push({0, segment.size() - 1});

        // if i-th element true the segment has to be subdivided at this element
        std::vector<bool> is_subdivision_element(segment.size(), false);

        // pre-computed elements coordinates
        std::vector<point_type> coordinates;
        coordinates.reserve(segment.size());
        for(index_t i = 0; i < segment.size(); i++){
            auto edge_index = segment[i];
            auto e = edge(edge_index, graph);
            auto s = source(e, graph);
            auto t = target(e, graph);
            point_type coordinate = embedding.lin2grid(s);
            if(s+1 == t) { // horizontal edge
                coordinate[1] += 0.5;
            }
            else{ // vertical edge
                coordinate[0] += 0.5;
            }
            coordinates.push_back(coordinate);
        }

        // recursive identification of subdivision elements
        while(!stack.empty()){
            auto element_index = stack.top();
            stack.pop();
            auto first_element = element_index.first;
            auto last_element = element_index.second;

            auto & coordinate_first = coordinates[first_element];
            auto & coordinate_last = coordinates[last_element];

            auto distance_first_last = norm(coordinate_first, coordinate_last);

            // segment to small to be subdivided
            if(distance_first_last <= minSize)
                continue;

            double distance_threshold;
            if(relative_epsilon){
                distance_threshold = epsilon * distance_first_last;
            }else{
                distance_threshold = epsilon;
            }

            auto max_distance = distance_threshold;
            auto max_distance_element = invalid_index;

            for(index_t i = first_element + 1; i < last_element; i++){
                auto & coordinate_element = coordinates[i];
                auto d = distance_to_line(coordinate_first, coordinate_last, coordinate_element);
                if(d >= max_distance){
                    max_distance = d;
                    max_distance_element = i;
                }
            }

            if(max_distance_element != invalid_index){
                is_subdivision_element[max_distance_element] = true;
                stack.push({first_element, max_distance_element});
                stack.push({max_distance_element + 1, last_element});
            }
        }

        // final subdivision
        polyline_contour_2d result;
        index_t last_subdivision = 0;
        for(index_t i = 1; i < segment.size(); i++){
            if(is_subdivision_element[i]){
                result.add_segment(segment.begin() + last_subdivision, segment.begin() + i + 1);
                last_subdivision = i + 1;
                i++;
            }
        }
        result.add_segment(segment.begin() + last_subdivision, segment.end());

        return result;
    }
*/
    /**
     * Subdivide the each segment of the given polyline  such that the distance between the line
     * joining the extremities of the contour segment and each of its elements is lower than the threshold (
     * Ramer–Douglas–Peucker algorithm)
     *
     * The threshold is equal to
     *  - epsilon if relative_epsilon is false
     *  - epsilon times the distance between the segment extremities if relative_epsilon is true
     *
     * If the distance between the segment extremities is smaller than minSize, the segment is never subdivided.
     * @tparam graph_t
     * @tparam embedding_t
     * @param polyline
     * @param graph
     * @param embedding
     * @param epsilon
     * @param relative_epsilon
     * @param minSize
     * @return
     */
/*    template<typename graph_t, typename embedding_t>
    auto subdivide_contour(const polyline_contour_2d &polyline,
                           const graph_t &graph,
                           const embedding_t &embedding,
                           double epsilon = 0.05,
                           bool relative_epsilon = true,
                           int minSize = 2) {

        if(polyline.size()==0)
        {
            return polyline;
        }else if(polyline.size() == 1){
            return subdivide_contour(polyline[0], graph, embedding, epsilon, relative_epsilon, minSize);
        }else{
            polyline_contour_2d result;
            for(auto & segment: polyline){
                result.concatenate(subdivide_contour(segment, graph, embedding, epsilon, relative_epsilon, minSize));
            }
            return result;
        }
    };*/

    /**
     * Subdivide the each segment of each polyline of the given contours such that the distance between the line
     * joining the extremities of the contour segment and each of its elements is lower than the threshold (
     * Ramer–Douglas–Peucker algorithm)
     *
     * The threshold is equal to
     *  - epsilon if relative_epsilon is false
     *  - epsilon times the distance between the segment extremities if relative_epsilon is true
     *
     * If the distance between the segment extremities is smaller than minSize, the segment is never subdivided.
     *
     * @tparam graph_t
     * @tparam embedding_t
     * @param contour
     * @param graph
     * @param embedding
     * @param epsilon
     * @param relative_epsilon
     * @param minSize
     * @return
     */
    /*  template<typename graph_t, typename embedding_t>
      auto subdivide_contour(const contour_2d &contour,
                             const graph_t &graph,
                             const embedding_t &embedding,
                             double epsilon = 0.05,
                             bool relative_epsilon = true,
                             int minSize = 2) {

          contour_2d result;
          for(auto & polyline: contour){
              result.add_polyline_contour_2d(subdivide_contour(polyline, graph, embedding, epsilon, relative_epsilon, minSize));
          }
          return result;
      };*/
}