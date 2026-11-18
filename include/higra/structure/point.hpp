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

#include "../utils.hpp"
#include "array/array.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <ostream>

namespace hg {

    /**
     * Point (or vector) of ``dim`` coordinates of type ``value_t``, stored inline.
     *
     * A point is an aggregate: ``point<index_t, 2> p{{1, 2}}`` or ``point<index_t, 2> p{1, 2}``.
     * A default-initialized point has unspecified coordinates; use ``fill`` or brace
     * initialization (``point<index_t, 2> p{}`` sets every coordinate to 0).
     *
     * Arithmetic operators ``+`` and ``-`` are elementwise between points of the same type;
     * ``min`` and ``max`` compute elementwise minima and maxima.
     *
     * @tparam value_t coordinate type
     * @tparam dim number of coordinates
     */
    template<typename value_t, unsigned int dim>
    struct point {
        using value_type = value_t;
        using size_type = std::size_t;
        using iterator = value_t *;
        using const_iterator = const value_t *;

        std::array<value_t, dim> coordinates;

        static constexpr index_t size() noexcept {
            return (index_t) dim;
        }

        value_t &operator()(index_t i) noexcept {
            return coordinates[(std::size_t) i];
        }

        const value_t &operator()(index_t i) const noexcept {
            return coordinates[(std::size_t) i];
        }

        value_t &operator[](index_t i) noexcept {
            return coordinates[(std::size_t) i];
        }

        const value_t &operator[](index_t i) const noexcept {
            return coordinates[(std::size_t) i];
        }

        value_t *data() noexcept {
            return coordinates.data();
        }

        const value_t *data() const noexcept {
            return coordinates.data();
        }

        iterator begin() noexcept {
            return coordinates.data();
        }

        iterator end() noexcept {
            return coordinates.data() + dim;
        }

        const_iterator begin() const noexcept {
            return coordinates.data();
        }

        const_iterator end() const noexcept {
            return coordinates.data() + dim;
        }

        const_iterator cbegin() const noexcept {
            return coordinates.data();
        }

        const_iterator cend() const noexcept {
            return coordinates.data() + dim;
        }

        /**
         * Set every coordinate to ``value``.
         */
        void fill(const value_t &value) noexcept {
            coordinates.fill(value);
        }

        /**
         * Conversion to a point with another coordinate type (``static_cast`` of each coordinate).
         */
        template<typename U, typename = std::enable_if_t<!std::is_same<U, value_t>::value>>
        operator point<U, dim>() const noexcept {
            point<U, dim> r;
            for (std::size_t i = 0; i < dim; ++i) {
                r.coordinates[i] = static_cast<U>(coordinates[i]);
            }
            return r;
        }

        friend point operator+(const point &a, const point &b) noexcept {
            point r;
            for (std::size_t i = 0; i < dim; ++i) {
                r.coordinates[i] = a.coordinates[i] + b.coordinates[i];
            }
            return r;
        }

        friend point operator-(const point &a, const point &b) noexcept {
            point r;
            for (std::size_t i = 0; i < dim; ++i) {
                r.coordinates[i] = a.coordinates[i] - b.coordinates[i];
            }
            return r;
        }

        friend bool operator==(const point &a, const point &b) noexcept {
            return a.coordinates == b.coordinates;
        }

        friend bool operator!=(const point &a, const point &b) noexcept {
            return !(a == b);
        }

        friend std::ostream &operator<<(std::ostream &os, const point &p) {
            os << "{";
            for (std::size_t i = 0; i < dim; ++i) {
                os << (i == 0 ? "" : ", ") << p.coordinates[i];
            }
            return os << "}";
        }
    };

    /**
     * Elementwise minimum of two points.
     */
    template<typename value_t, unsigned int dim>
    point<value_t, dim> min(const point<value_t, dim> &a, const point<value_t, dim> &b) noexcept {
        point<value_t, dim> r;
        for (std::size_t i = 0; i < dim; ++i) {
            r.coordinates[i] = (std::min)(a.coordinates[i], b.coordinates[i]);
        }
        return r;
    }

    /**
     * Elementwise maximum of two points.
     */
    template<typename value_t, unsigned int dim>
    point<value_t, dim> max(const point<value_t, dim> &a, const point<value_t, dim> &b) noexcept {
        point<value_t, dim> r;
        for (std::size_t i = 0; i < dim; ++i) {
            r.coordinates[i] = (std::max)(a.coordinates[i], b.coordinates[i]);
        }
        return r;
    }

    /**
     * Copy of the coordinates of a point into a new 1d array of ``dim`` elements.
     */
    template<typename value_t, unsigned int dim>
    array<value_t> to_array(const point<value_t, dim> &p) {
        array<value_t> r(array_shape((index_t) dim));
        std::copy(p.begin(), p.end(), r.data());
        return r;
    }

    using point_1d_f = point<double, 1>;
    using point_1d_i = point<index_t, 1>;

    using point_2d_f = point<double, 2>;
    using point_2d_i = point<index_t, 2>;

    using point_3d_f = point<double, 3>;
    using point_3d_i = point<index_t, 3>;

    using point_4d_f = point<double, 4>;
    using point_4d_i = point<index_t, 4>;
}
