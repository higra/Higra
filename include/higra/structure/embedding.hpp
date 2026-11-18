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

#include "array.hpp"
#include "point.hpp"
#include "../utils.hpp"
#include <iterator>
#include <string>
#include <vector>


namespace hg {

    namespace embedding_internal {

        /**
         * True if T is an hg::array or an hg::array_view (the array overloads apply to them).
         */
        template<typename T, typename = void>
        struct is_array : std::false_type {
        };

        template<typename T>
        struct is_array<T, std::void_t<typename T::value_type>> :
                std::is_base_of<array_view<const typename T::value_type>, T> {
        };

        template<typename T>
        constexpr bool is_array_v = is_array<T>::value;
    }


    namespace embedding_internal {


        /**
         * Grid embeddings are utility classes to ease the manipulation of point coordinates in the d-dimensional integer grid.
         * An embedding has a shape (height and width in 2 dimensions).
         * @tparam dim Dimension of the embedding
         * @tparam coordinates_t Type of coordinates manipulated
         */
        template<int dim, typename coordinates_t=index_t>
        class embedding_grid {
        public:
            using coordinate_type = coordinates_t;
            using point_type = point<index_t, dim>;
            using shape_type = point_type;
            using self_type = embedding_grid<dim, coordinates_t>;
        private:
            size_t nbElement = 0;
            shape_type _shape{}; // signed for safer comparisons
            point<size_t, dim> sum_prod{};

            /**
             * True if 0 <= c < s (s > 0), without signed/unsigned comparison pitfalls.
             */
            template<typename V>
            static bool in_range(V c, index_t s) {
                if constexpr (std::is_signed<V>::value) {
                    return c >= 0 && (index_t) c < s;
                } else {
                    return (unsigned long long) c < (unsigned long long) s;
                }
            }

            /**
             * Number of elements of an iterable sequence of coordinates.
             */
            template<typename T>
            static index_t coordinates_count(const T &coordinates) {
                return (index_t) std::distance(std::begin(coordinates), std::end(coordinates));
            }

            /**
             * Message of the error raised when a sequence of coordinates does not have dim elements.
             */
            static std::string coordinates_count_error(index_t count) {
                return "Coordinates size does not match embedding dimension (expected " + std::to_string(dim) +
                       ", got " + std::to_string(count) + ").";
            }

            void computeSize() {
                if (dim == 0)
                    nbElement = 0;
                else {
                    nbElement = 1;
                    for (auto &d: _shape)
                        nbElement *= d;
                }
            }

            void computeSumProd() {
                sum_prod(dim - 1) = 1;
                for (index_t i = dim - 2; i >= 0; --i) {
                    sum_prod(i) = sum_prod(i + 1) * _shape(i + 1);
                }
            }

            void assert_positive_shape() {
                for (const auto c: _shape) {
                    hg_assert(c > 0, "Axis size must be positive.");
                }

            }

        public:

            static const int _dim = dim;

            /**
             * Creates an embedding of size 0.
             */
            embedding_grid() {}

            /**
             * Creates an embedding with the given shape
             * @param shape list of dim positive integers
             */
            embedding_grid(const std::initializer_list<index_t> &shape) {
                hg_assert(dim == shape.size(), "Shape dimension does not match embedding dimension !");
                _shape.fill(0);
                index_t i = 0;
                for (const auto c:shape){
                    _shape(i++) = c;
                }
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            /**
             * Creates an embedding with the given shape
             * @param shape array of dim positive integers
             */
            template<typename T>
            embedding_grid(array_view<const T> shape) {
                static_assert(std::is_integral<T>::value,
                              "Coordinates must have integral value type.");
                hg_assert(shape.dimension() == 1 && dim == shape.size(),
                          "Shape dimension does not match embedding dimension !");
                for (index_t i = 0; i < dim; ++i) {
                    _shape(i) = (index_t) shape(i);
                }
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            ~embedding_grid() = default;

            embedding_grid(const self_type &other) = default;

            embedding_grid(self_type &&other) = default;

            self_type &operator=(const self_type &) = default;

            self_type &operator=(self_type &&) = default;

            /**
             * Shape of the embedding
             * @return
             */
            const auto &shape() const {
                return _shape;
            }

            /**
             * Creates an embedding with the given shape
             * @param shape list of dim positive integers
             */
            template<typename T>
            embedding_grid(const T &shape) {
                hg_assert(dim == shape.size(), "Shape dimension does not match embedding dimension !");
                _shape.fill(0);
                index_t i = 0;
                for (const auto c:shape)
                    _shape(i++) = c;
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            /**
             * Dimension of the embedding
             * @return
             */
            auto dimension() const {
                return dim;
            }

            /**
             * Number of elements in the embedding (product of every shape dimension)
             * @return
             */
            auto size() const {
                return nbElement;
            }

            /**
             * Convert the coordinates of a point (in the grid coordinate system) into linear coordinates (row major)
             * @tparam T
             * @param coordinates
             * @return
             */
            template<typename T,
                    typename = std::enable_if_t<!embedding_internal::is_array_v<T>>
            >
            index_t grid2lin(const T &coordinates) const {
                hg_assert(coordinates_count(coordinates) == dim,
                          coordinates_count_error(coordinates_count(coordinates)));
                index_t result = 0;
                index_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sum_prod(count++);
                return result;
            }

            /**
             * Convert the coordinates of a point (in the grid coordinate system) into linear coordinates (row major)
             * @tparam T
             * @param coordinates
             * @return
             */
            template<typename T>
            auto grid2lin(const point<T, dim> &coordinates) const {
                /*static_assert(std::is_integral<T>::value,
                              "Coordinates must have integral value type.");*/
                index_t res = coordinates[dim - 1];
                for (index_t i = 0; i < dim - 1; ++i)
                    res += coordinates[i] * sum_prod[i];
                return res;
            }

            /**
             * Convert an array of points coordinates (in the grid coordinate system) into linear coordinates (row major)
             * @tparam T
             * @param coordinates an array of grid coordinates of shape (n1, n2, .., nx, dim)
             * @return an array of linear coordinates of shape (n1, n2, .., nx)
             */
            template<typename T>
            auto grid2lin(array_view<const T> coordinates) const {
                static_assert(std::is_integral<T>::value,
                              "Coordinates must have integral value type.");
                hg_assert(coordinates.dimension() > 0, "Coordinates must have at least one dimension.");
                const index_t last = coordinates.shape(coordinates.dimension() - 1);
                // a last axis of size 1 is broadcast to every coordinate
                hg_assert(last == dim || last == 1, "Coordinates size does not match embedding dimension.");

                std::vector<size_t> result_shape(coordinates.shape().begin(), coordinates.shape().end() - 1);
                array_nd<size_t> result = array_nd<size_t>::from_shape(array_shape::from(result_shape));
                const auto *c = coordinates.data();
                const index_t n = (index_t) result.size();
                size_t total_sum_prod = 0;
                for (index_t i = 0; i < dim; ++i) {
                    total_sum_prod += sum_prod(i);
                }
                auto *r = result.data();
                if (last == dim) {
                    for (index_t j = 0; j < n; ++j, c += dim) {
                        size_t v = 0;
                        for (index_t i = 0; i < dim; ++i) {
                            v += (size_t) c[i] * sum_prod(i);
                        }
                        r[j] = v;
                    }
                } else {
                    for (index_t j = 0; j < n; ++j) {
                        r[j] = (size_t) c[j] * total_sum_prod;
                    }
                }
                return result;
            }

            /**
             * Convert the coordinates of a point (in the grid coordinate system) into linear coordinates (row major)
             * @param coordinates
             * @return
             */
            index_t grid2lin(const std::initializer_list<coordinates_t> &coordinates) const {
                hg_assert(coordinates_count(coordinates) == dim,
                          coordinates_count_error(coordinates_count(coordinates)));
                index_t result = 0;
                index_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sum_prod(count++);
                return result;
            }


            /**
             * Test if the given embedding contains the given point given with its grid coordinates
             * @param coordinates
             * @return
             */
            bool contains(const std::initializer_list<coordinates_t> &coordinates) const {
                hg_assert(coordinates_count(coordinates) == dim,
                          coordinates_count_error(coordinates_count(coordinates)));
                index_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= _shape(count++))
                        return false;
                return true;
            }

            /**
             * Test if the given embedding contains the given point given with its grid coordinates
             * @param coordinates
             * @return
             */
            template<typename T>
            auto contains(const point<T, dim> &coordinates) const {
                for (index_t i = 0; i < dim; ++i)
                    if (coordinates(i) < 0 || coordinates(i) >= _shape(i))
                        return false;
                return true;
            }

            /**
             * Test if the given embedding contains the given points given with their grid coordinates
             * @tparam T
             * @param coordinates an array of grid coordinates of shape (n1, n2, .., nx, dim)
             * @return an array of boolean of shape (n1, n2, .., nx)
             */
            template<typename T>
            auto contains(array_view<const T> coordinates) const {
                static_assert(std::is_integral<T>::value,
                              "Coordinates must have integral value type.");

                hg_assert(coordinates.dimension() > 0 &&
                          coordinates.shape(coordinates.dimension() - 1) == dim,
                          "Coordinates size does not match embedding dimension.");

                std::vector<size_t> result_shape(coordinates.shape().begin(), coordinates.shape().end() - 1);
                array_nd<bool> result = array_nd<bool>::from_shape(array_shape::from(result_shape));
                const auto *c = coordinates.data();
                const index_t n = (index_t) result.size();
                auto *r = result.data();
                for (index_t j = 0; j < n; ++j, c += dim) {
                    bool inside = true;
                    for (index_t i = 0; i < dim; ++i) {
                        if (!in_range(c[i], _shape(i))) {
                            inside = false;
                            break;
                        }
                    }
                    r[j] = inside;
                }
                return result;
            }

            /**
             * Test if the given embedding contains the given point given with its grid coordinates
             * @param coordinates
             * @return
             */
            template<typename T,
                    typename = std::enable_if_t<!embedding_internal::is_array_v<T>>
            >
            bool contains(const T &coordinates) const {
                hg_assert(coordinates_count(coordinates) == dim,
                          coordinates_count_error(coordinates_count(coordinates)));
                index_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= _shape(count++))
                        return false;
                return true;
            }

            /**
             * Converts the coordinates of a point from linear to grid system
             * @param index
             * @return
             */
            auto lin2grid(index_t index) const {
                point_type result;

                for (index_t i = 0; i < dim - 1; ++i) {

                    result[i] = index / sum_prod(i);
                    index = index % sum_prod(i);
                }
                result[dim - 1] = index / sum_prod(dim - 1);
                return result;
            }

            /**
             * Converts the coordinates of points from linear to grid system
             * @tparam T
             * @param xindices an array of points linear coordinates of shape (n1, n2,... nx)
             * @return an array of points grid coordinates of shape (n1, n2,... nx, dim)
             */
            template<typename T>
            array_nd<coordinates_t> lin2grid(array_view<const T> indices) const {
                static_assert(std::is_integral<T>::value,
                              "Indices must have integral value type.");

                const index_t size = indices.size();
                std::vector<index_t> shape(indices.shape().begin(), indices.shape().end());
                shape.push_back(dim);

                array_nd<coordinates_t> result = array_nd<coordinates_t>::from_shape(array_shape::from(shape));
                const auto *indices_data = indices.data();
                auto *r = result.data();

                for (index_t j = 0; j < size; ++j, r += dim) {
                    auto index = indices_data[j];
                    for (index_t i = 0; i < dim - 1; ++i) {

                        r[i] = index / sum_prod[i];
                        index = index % sum_prod[i];
                    }
                    r[dim - 1] = index / sum_prod[dim - 1];
                }
                return result;
            }
        };

    }

    template<int dim>
    using embedding_grid = typename embedding_internal::embedding_grid<dim, index_t>;

    using embedding_grid_1d = typename embedding_internal::embedding_grid<1, index_t>;

    using embedding_grid_2d = typename embedding_internal::embedding_grid<2, index_t>;

    using embedding_grid_3d = typename embedding_internal::embedding_grid<3, index_t>;

    using embedding_grid_4d = typename embedding_internal::embedding_grid<4, index_t>;
}