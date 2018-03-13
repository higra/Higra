//
// Created by user on 3/9/18.
//
#include "xtensor/xarray.hpp"

#pragma once

namespace hg {


    namespace embedding_internal {


        template<typename coordinates_t=long>
        class embedding_grid {
        private:
            std::size_t dim = 0;
            std::size_t nbElement = 0;
            std::vector<std::size_t> shape;
            std::vector<std::size_t> sump_prod;

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
                sump_prod.push_back(1);
                for (int i = 1; i < dim; ++i) {
                    sump_prod.push_back(sump_prod[i - 1] * shape[i - 1]);
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


            std::size_t size() const {
                return nbElement;
            }

            template<typename T>
            std::size_t grid2lin(const T &coordinates) const {
                std::size_t result = 0;
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sump_prod[count++];
                return result;
            }

            std::size_t grid2lin(const std::initializer_list<coordinates_t> &coordinates) const {
                std::size_t result = 0;
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    result += c * sump_prod[count++];
                return result;
            }

            template<typename T>
            bool isInBound(const T &coordinates) const {
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= shape[count++])
                        return false;
                return true;
            }


            bool isInBound(const std::initializer_list<coordinates_t> &coordinates) const {
                std::size_t count = 0;
                for (const auto &c: coordinates)
                    if (c < 0 || c >= shape[count++])
                        return false;
                return true;
            }

            xt::xarray<coordinates_t> lin2grid(std::size_t index) const {
                xt::xarray<coordinates_t> result = xt::zeros<coordinates_t>({dim});
                for (int i = dim - 1; i >= 0; --i) {

                    result[i] = index / sump_prod[i];
                    index = index % sump_prod[i];
                }
                return result;
            }
        };

    }

    using embedding_grid = typename embedding_internal::embedding_grid<long>;
}