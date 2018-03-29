//
// Created by user on 3/9/18.
//

#pragma once

#include "xtensor/xexpression.hpp"
#include "xtensor/xreducer.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xio.hpp"
#include "debug.hpp"


namespace hg {


    namespace embedding_internal {


        template<typename coordinates_t=long>
        class embedding_grid {
        private:
            std::size_t dim = 0;
            std::size_t nbElement = 0;
            xt::xarray<long> shape; // long for safer comparisons
            xt::xarray<std::size_t> sum_prod;

            void computeSize() {
                if (dim == 0)
                    nbElement = 0;
                else {
                    nbElement = 1;
                    for (auto &d: shape)
                        nbElement *= d;
                }
            }

            void computeSumProd() {
                sum_prod = xt::zeros<std::size_t>({dim});

                sum_prod(dim - 1) = 1;

                long dims = dim;
                for (long i = dims - 2; i >= 0; --i) {
                    sum_prod(i) = sum_prod(i + 1) * shape(i + 1);
                }
            }

            void assert_positive_shape() {
                for (const auto c: shape) {
                    hg_assert(c > 0, "Axis size must be positive.");
                }

            }

        public:

            embedding_grid() {}


            embedding_grid(const std::initializer_list<long> &_shape) : shape(_shape) {
                dim = shape.size();
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            template<typename T>
            embedding_grid(const xt::xexpression<T> &_shape) : shape(_shape.derived_cast()) {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");
                dim = shape.size();
                assert_positive_shape();
                computeSumProd();
                computeSize();
            }

            const auto &getShape() const {
                return shape;
            }

            template<typename T>
            embedding_grid(const T &_shape) {
                shape = xt::zeros<long>({_shape.size()});
                std::size_t i = 0;
                for (const auto c:_shape)
                    shape(i++) = c;
                dim = shape.size();
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

            template<typename T>
            std::size_t grid2lin(const T &coordinates) const {
                std::size_t result = 0;
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sum_prod(count++);
                return result;
            }


            template<typename T>
            auto grid2linV(const xt::xexpression<T> &xcoordinates) const {
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
                    if (c < 0 || c >= (long) shape(count++))
                        return false;
                return true;
            }

            template<typename T>
            auto contains(const xt::xexpression<T> &xcoordinates) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Coordinates must have integral value type.");
                std::cout << "in\n";

                const auto &coordinates = xcoordinates.derived_cast();
                hg_assert(coordinates.shape().back() == dim, "Coordinates size does not match embedding dimension.");
                std::cout << "assert !\n";
                return xt::amin(coordinates >= 0 && coordinates < shape, {coordinates.dimension() - 1});
            }

            template<typename T,
                    typename = std::enable_if_t<!std::is_base_of<xt::xexpression<T>, T>::value>
            >
            bool contains(const T &coordinates) const {

                std::size_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= (long) shape(count++))
                        return false;
                return true;
            }

            auto lin2grid(std::size_t index) const {
                std::vector<long> result(dim);

                for (std::size_t i = 0; i < dim; ++i) {

                    result[i] = index / sum_prod(i);
                    index = index % sum_prod(i);
                }
                return result;
            }

            template<typename T>
            xt::xarray<coordinates_t> lin2grid(const xt::xexpression<T> &xindices) const {
                static_assert(std::is_integral<typename T::value_type>::value,
                              "Indices must have integral value type.");
                const auto indices = xindices.derived_cast();

                auto size = indices.size();
                auto shape = indices.shape();
                shape.push_back(dim);

                xt::xarray<coordinates_t> result = xt::zeros<coordinates_t>({size, dim});


                const auto indicesLin = xt::flatten(indices);

                for (std::size_t j = 0; j < size; ++j) {
                    auto index = indicesLin(j);
                    for (std::size_t i = 0; i < dim; ++i) {

                        result(j, i) = index / sum_prod[i];
                        index = index % sum_prod[i];
                    }
                }
                result.reshape(shape);
                return result;
            }
        };

    }

    using embedding_grid = typename embedding_internal::embedding_grid<long>;
}