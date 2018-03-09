//
// Created by user on 3/9/18.
//
#include "xtensor/xarray.hpp"
#pragma once

namespace hg {

    template<typename coordinates_t=long>
    class EmbeddingGrid {
    private:
        size_t dim;
        std::vector <ulong> shape;
        std::vector <ulong> sump_prod;

    public:
        template <typename T>
        EmbeddingGrid(T _shape) {

            for(auto & c: _shape)
                shape.push_back(c);
            dim = shape.size();

            sump_prod.push_back(1);
            for (int i = 1; i < dim; ++i) {
                sump_prod.push_back(sump_prod[i - 1] * shape[i - 1]);
            }
        }

        EmbeddingGrid(std::initializer_list<ulong> _shape) {

            for(auto & c: _shape)
                shape.push_back(c);
            dim = shape.size();

            sump_prod.push_back(1);
            for (int i = 1; i < dim; ++i) {
                sump_prod.push_back(sump_prod[i - 1] * shape[i - 1]);
            }
        }

        std::vector<ulong> getShape() const{
            return shape;
        }

        template <typename T>
        coordinates_t grid2lin(const T coordinates) const {
            coordinates_t result = 0;
            size_t count = 0;
            for(auto & c: coordinates)
                result += c * sump_prod[count++];
            return result;
        }

        coordinates_t grid2lin(const std::initializer_list<coordinates_t> coordinates) const {
            coordinates_t result = 0;
            size_t count = 0;
            for(auto & c: coordinates)
                result += c * sump_prod[count++];
            return result;
        }

        template <typename T>
        bool isInBound(const T coordinates) const {
            size_t count = 0;
            for(auto & c: coordinates)
                if(c < 0 || c >= shape[count++])
                    return false;
            return true;
        }


        bool isInBound(std::initializer_list<coordinates_t> coordinates) const {
            size_t count = 0;
            for(auto & c: coordinates)
                if(c < 0 || c >= shape[count++])
                    return false;
            return true;
        }

        xt::xarray<coordinates_t> lin2grid(ulong index) const {
            xt::xarray<coordinates_t> result = xt::zeros<coordinates_t>({dim});
            for (int i = dim - 1; i >= 0; --i) {

                result[i] = index / sump_prod[i];
                index = index % sump_prod[i];
            }
            return result;
        }
    };

}