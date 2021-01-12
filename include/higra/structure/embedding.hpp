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

#include "xtensor/xexpression.hpp"
#include "xtensor/xreducer.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xstrided_view.hpp"
#include "array.hpp"
#include "point.hpp"
#include "../utils.hpp"


namespace hg {


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
            shape_type _shape; // signed for safer comparisons
            point<size_t, dim> sum_prod;

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
                hg_assert(dim == _shape.size(), "Shape dimension does not match embedding dimension !");
                _shape = xt::zeros<index_t>({shape.size()});
                index_t i = 0;
                for (const auto c:shape)
                    _shape(i++) = c;
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            /**
             * Creates an embedding with the given shape
             * @param shape array of dim positive integers
             */
            template<typename T>
            embedding_grid(const xt::xexpression<T> &shape) : _shape(shape.derived_cast()) {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");
                hg_assert(dim == _shape.size(), "Shape dimension does not match embedding dimension !");
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
                _shape = xt::zeros<index_t>({shape.size()});
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
                    typename = std::enable_if_t<!std::is_base_of<xt::xexpression<T>, T>::value>
            >
            index_t grid2lin(const T &coordinates) const {
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
            auto grid2lin(const xt::xexpression<T> &xcoordinates) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");
                const auto &coordinates = xcoordinates.derived_cast();
                //hg_assert(coordinates.shape().back() == dim, "Coordinates size does not match embedding dimension.");

                auto last = coordinates.dimension() - 1;
                return xt::sum(coordinates * sum_prod, {last});
            }

            /**
             * Convert the coordinates of a point (in the grid coordinate system) into linear coordinates (row major)
             * @param coordinates
             * @return
             */
            index_t grid2lin(const std::initializer_list<coordinates_t> &coordinates) const {
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
            auto contains(const xt::xexpression<T> &xcoordinates) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");

                const auto &coordinates = xcoordinates.derived_cast();
                hg_assert(coordinates.shape().back() == dim, "Coordinates size does not match embedding dimension.");

                return xt::amin(coordinates >= 0 && coordinates < _shape, {coordinates.dimension() - 1});
            }

            /**
             * Test if the given embedding contains the given point given with its grid coordinates
             * @param coordinates
             * @return
             */
            template<typename T,
                    typename = std::enable_if_t<!std::is_base_of<xt::xexpression<T>, T>::value>
            >
            bool contains(const T &coordinates) const {

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
            xt::xarray<coordinates_t> lin2grid(const xt::xexpression<T> &xindices) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Indices must have integral value type.");
                const auto &indices = xindices.derived_cast();

                index_t size = (index_t) indices.size();
                auto shapeO = indices.shape();
                std::vector<size_t> shape(shapeO.begin(), shapeO.end());
                shape.push_back(dim);

                array_nd<coordinates_t> result = array_nd<coordinates_t>::from_shape({(size_t) size, (size_t) dim});


                const auto indicesLin = xt::flatten(indices);

                for (index_t j = 0; j < size; ++j) {
                    auto index = indicesLin(j);
                    for (index_t i = 0; i < dim - 1; ++i) {

                        result(j, i) = index / sum_prod[i];
                        index = index % sum_prod[i];
                    }
                    result(j, dim - 1) = index / sum_prod[dim - 1];
                }
                result.reshape(shape);
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