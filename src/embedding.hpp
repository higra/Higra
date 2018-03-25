//
// Created by user on 3/9/18.
//
#include "xtensor/xarray.hpp"
#include "xtensor/xreducer.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xstrided_view.hpp"
#include "debug.hpp"
#pragma once

namespace hg {


    namespace embedding_internal {


        template<typename coordinates_t=long>
        class embedding_grid {
        private:
            std::size_t dim = 0;
            std::size_t nbElement = 0;
            std::vector<std::size_t> shape;
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
                    sum_prod(i) = sum_prod(i + 1) * shape[i + 1];
                }
            }

        public:


            template<typename T>
            static auto make_embedding_grid(T _shape) {

                embedding_grid g;
                for (auto &c: _shape) {
                    g.shape.push_back(c);
                }
                g.dim = _shape.size();

                g.computeSumProd();
                g.computeSize();
                return g;
            }

            embedding_grid() {}


            embedding_grid(const std::initializer_list<std::size_t> &_shape) {

                for (auto &c: _shape)
                    shape.push_back(c);
                dim = shape.size();

                computeSumProd();
                computeSize();
            }

            std::vector<std::size_t> getShape() const {
                return shape;
            }

            std::size_t dims() const {
                return dim;
            }

            std::size_t size() const {
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


            xt::xarray<std::size_t> grid2linV(const xt::xarray<coordinates_t> &coordinates) const {
                hg_assert(coordinates.shape().back() == dims(), "Coordinates size does not match embedding dimension.");
                auto last = coordinates.shape().size() - 1;
                auto tmp = coordinates * sum_prod;
                return xt::sum(tmp, {last});
            }

            std::size_t grid2lin(const std::initializer_list<coordinates_t> &coordinates) const {
                std::size_t result = 0;
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sum_prod(count++);
                return result;
            }

            template<typename T>
            bool isInBound(const T &coordinates) const {
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= (long) shape[count++])
                        return false;
                return true;
            }


            bool isInBound(const std::initializer_list<coordinates_t> &coordinates) const {
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= (long) shape[count++])
                        return false;
                return true;
            }

            xt::xarray<coordinates_t> lin2grid(std::size_t index) const {
                xt::xarray<coordinates_t> result = xt::zeros<coordinates_t>({dim});

                for (std::size_t i = 0; i < dim; ++i) {

                    result[i] = index / sum_prod[i];
                    index = index % sum_prod[i];
                }
                return result;
            }

            xt::xarray<coordinates_t> lin2grid(const xt::xarray<std::size_t> &indices) const {
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