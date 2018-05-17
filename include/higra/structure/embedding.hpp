//
// Created by user on 3/9/18.
//

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


        template<int dim, typename coordinates_t=long>
        class embedding_grid {
        public:
            using coordinate_type = coordinates_t;
            using point_type = point<long, dim>;
            using shape_type = point_type;
        private:
            std::size_t nbElement = 0;
            shape_type _shape; // long for safer comparisons
            point<std::size_t, dim> sum_prod;

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
                for (long i = dim - 2; i >= 0; --i) {
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

            embedding_grid() {}


            embedding_grid(const std::initializer_list<long> &shape){
                hg_assert(dim == _shape.size(), "Shape dimension does not match embedding dimension !");
                _shape = xt::zeros<long>({shape.size()});
                std::size_t i = 0;
                for (const auto c:shape)
                    _shape(i++) = c;
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            template<typename T>
            embedding_grid(const xt::xexpression<T> &shape) : _shape(shape.derived_cast()) {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");
                hg_assert(dim == _shape.size(), "Shape dimension does not match embedding dimension !");
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            const auto &shape() const {
                return _shape;
            }

            template<typename T>
            embedding_grid(const T &shape) {
                hg_assert(dim == shape.size(), "Shape dimension does not match embedding dimension !");
                _shape = xt::zeros<long>({shape.size()});
                std::size_t i = 0;
                for (const auto c:shape)
                    _shape(i++) = c;
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }


            auto dimension() const {
                return dim;
            }

            auto size() const {
                return nbElement;
            }

            template<typename T,
                    typename = std::enable_if_t<!std::is_base_of<xt::xexpression<T>, T>::value>
            >
            std::size_t grid2lin(const T &coordinates) const {
                std::size_t result = 0;
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sum_prod(count++);
                return result;
            }


            template<typename T>
            auto grid2lin(const point<T, dim> &coordinates) const {
                static_assert(std::is_integral<T>::value,
                              "Coordinates must have integral value type.");
                std::size_t res = coordinates[dim - 1];
                for (std::size_t i = 0; i < dim - 1; ++i)
                    res += coordinates[i] * sum_prod[i];
                return res;
            }

            template<typename T>
            auto grid2lin(const xt::xexpression<T> &xcoordinates) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");
                const auto &coordinates = xcoordinates.derived_cast();
                hg_assert(coordinates.shape().back() == dim, "Coordinates size does not match embedding dimension.");

                auto last = coordinates.dimension() - 1;
                return xt::sum(coordinates * sum_prod, {last});
            }

            std::size_t grid2lin(const std::initializer_list<coordinates_t> &coordinates) const {
                std::size_t result = 0;
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sum_prod(count++);
                return result;
            }


            bool contains(const std::initializer_list<coordinates_t> &coordinates) const {
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= _shape(count++))
                        return false;
                return true;
            }

            template<typename T>
            auto contains(const point<T, dim> &coordinates) const {
                for (std::size_t i = 0; i < dim; ++i)
                    if (coordinates(i) < 0 || coordinates(i) >= _shape(i))
                        return false;
                return true;
            }

            template<typename T>
            auto contains(const xt::xexpression<T> &xcoordinates) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");


                const auto &coordinates = xcoordinates.derived_cast();
                hg_assert(coordinates.shape().back() == dim, "Coordinates size does not match embedding dimension.");

                return xt::amin(coordinates >= 0 && coordinates < _shape, {coordinates.dimension() - 1});
            }

            template<typename T,
                    typename = std::enable_if_t<!std::is_base_of<xt::xexpression<T>, T>::value>
            >
            bool contains(const T &coordinates) const {

                std::size_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= (long) _shape(count++))
                        return false;
                return true;
            }

            auto lin2grid(std::size_t index) const {
                point_type result;

                for (std::size_t i = 0; i < dim - 1; ++i) {

                    result[i] = index / sum_prod(i);
                    index = index % sum_prod(i);
                }
                result[dim - 1] = index / sum_prod(dim - 1);
                return result;
            }

            template<typename T>
            xt::xarray<coordinates_t> lin2grid(const xt::xexpression<T> &xindices) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Indices must have integral value type.");
                const auto &indices = xindices.derived_cast();

                auto size = indices.size();
                auto shapeO = indices.shape();
                std::vector<std::size_t> shape(shapeO.begin(), shapeO.end());
                shape.push_back(dim);

                array_nd<coordinates_t> result = xt::zeros<coordinates_t>({size, (std::size_t) dim});


                const auto indicesLin = xt::flatten(indices);

                for (std::size_t j = 0; j < size; ++j) {
                    auto index = indicesLin(j);
                    for (std::size_t i = 0; i < dim - 1; ++i) {

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
    using embedding_grid = typename embedding_internal::embedding_grid<dim, long>;

    using embedding_grid_1d = typename embedding_internal::embedding_grid<1, long>;

    using embedding_grid_2d = typename embedding_internal::embedding_grid<2, long>;

    using embedding_grid_3d = typename embedding_internal::embedding_grid<3, long>;

    using embedding_grid_4d = typename embedding_internal::embedding_grid<4, long>;
}