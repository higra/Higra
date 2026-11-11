/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include "../../utils.hpp"
#include "device.hpp"

namespace hg {

    /**
     * Maximum number of dimensions of an array.
     */
    inline constexpr index_t max_rank = 8;

    namespace array_internal {

        template<class I>
        inline constexpr bool is_index_type_v = std::is_integral_v<I> && !std::is_same_v<std::remove_cv_t<I>, bool>;

        /**
         * Converts an integral value to a dimension; throws if it is negative or above INT64_MAX.
         */
        template<class I>
        index_t checked_dimension(I value) {
            static_assert(is_index_type_v<I>, "Array dimensions must be integers.");
            if constexpr (std::is_signed_v<I>) {
                hg_assert(value >= 0, "Array dimensions cannot be negative (got " + std::to_string(value) + ").");
            } else if constexpr (sizeof(I) >= sizeof(index_t)) {
                hg_assert(value <= static_cast<std::make_unsigned_t<index_t>>(std::numeric_limits<index_t>::max()),
                          "Array dimension " + std::to_string(value) + " is too large.");
            }
            return static_cast<index_t>(value);
        }
    }

    /**
     * Shape of an array: between 0 and ``max_rank`` dimensions, each a non-negative ``index_t``.
     *
     * The shape is stored inline (no heap allocation) and is trivially copyable.
     *
     * Construction:
     *  - ``array_shape()`` or ``{}``: rank 0 (a scalar, one element);
     *  - ``array_shape(n0, n1, ...)`` or ``{n0, n1, ...}``: 1 to ``max_rank`` values of any integral
     *    types, signed or unsigned, mixed; negative values and unsigned values above INT64_MAX throw;
     *  - ``array_shape::from(range)``: dimensions copied from any range of integers (``std::vector``,
     *    another shape...); a range longer than ``max_rank`` throws.
     */
    class array_shape {
    public:
        using value_type = index_t;
        using const_iterator = const index_t *;

        HG_HOST_DEVICE constexpr array_shape() noexcept = default;

        template<class... I,
                std::enable_if_t<(sizeof...(I) >= 1) && (sizeof...(I) <= (std::size_t) max_rank) &&
                                 (array_internal::is_index_type_v<I> && ...), int> = 0>
        array_shape(I... dims) : m_ndim(static_cast<index_t>(sizeof...(I))) {
            index_t k = 0;
            ((m_dims[k++] = array_internal::checked_dimension(dims)), ...);
        }

        template<class Range>
        static array_shape from(const Range &range) {
            array_shape s;
            for (const auto &v: range) {
                hg_assert(s.m_ndim < max_rank,
                          "Array rank cannot exceed " + std::to_string(max_rank) + ".");
                s.m_dims[s.m_ndim++] = array_internal::checked_dimension(v);
            }
            return s;
        }

        /**
         * Number of dimensions.
         */
        HG_HOST_DEVICE index_t size() const noexcept {
            return m_ndim;
        }

        HG_HOST_DEVICE index_t operator[](index_t k) const {
#ifdef HG_ARRAY_CHECKS
            hg_assert(k >= 0 && k < m_ndim, "Shape index out of range.");
#endif
            return m_dims[k];
        }

        HG_HOST_DEVICE index_t &operator[](index_t k) {
#ifdef HG_ARRAY_CHECKS
            hg_assert(k >= 0 && k < m_ndim, "Shape index out of range.");
#endif
            return m_dims[k];
        }

        HG_HOST_DEVICE const index_t *data() const noexcept {
            return m_dims.data();
        }

        HG_HOST_DEVICE const index_t *begin() const noexcept {
            return m_dims.data();
        }

        HG_HOST_DEVICE const index_t *end() const noexcept {
            return m_dims.data() + m_ndim;
        }

        /**
         * Number of elements: product of the dimensions, 1 for rank 0.
         */
        HG_HOST_DEVICE index_t numel() const noexcept {
            index_t n = 1;
            for (index_t k = 0; k < m_ndim; ++k) {
                n *= m_dims[k];
            }
            return n;
        }

        /**
         * Shape without its first dimension (``shape[1:]``); rank 0 stays rank 0.
         */
        HG_HOST_DEVICE array_shape tail() const noexcept {
            array_shape s;
            if (m_ndim > 0) {
                s.m_ndim = m_ndim - 1;
                for (index_t k = 1; k < m_ndim; ++k) {
                    s.m_dims[k - 1] = m_dims[k];
                }
            }
            return s;
        }

        HG_HOST_DEVICE friend bool operator==(const array_shape &a, const array_shape &b) noexcept {
            if (a.m_ndim != b.m_ndim) {
                return false;
            }
            for (index_t k = 0; k < a.m_ndim; ++k) {
                if (a.m_dims[k] != b.m_dims[k]) {
                    return false;
                }
            }
            return true;
        }

        HG_HOST_DEVICE friend bool operator!=(const array_shape &a, const array_shape &b) noexcept {
            return !(a == b);
        }

    private:
        std::array<index_t, max_rank> m_dims{};
        index_t m_ndim = 0;
    };

    namespace array_internal {

        /**
         * Number of elements of the shape; throws if the product or the byte size of the
         * elements overflows.
         */
        inline index_t checked_numel(const array_shape &shape, std::size_t element_size) {
            const index_t max_elements = std::numeric_limits<index_t>::max() / static_cast<index_t>(element_size);
            index_t n = 1;
            for (index_t d: shape) {
                if (d == 0) {
                    return 0;
                }
            }
            for (index_t d: shape) {
                hg_assert(n <= max_elements / d, "Array size overflow.");
                n *= d;
            }
            return n;
        }

        inline std::string shape_to_string(const array_shape &s) {
            std::string r = "(";
            for (index_t k = 0; k < s.size(); ++k) {
                if (k > 0) {
                    r += ", ";
                }
                r += std::to_string(s[k]);
            }
            if (s.size() == 1) {
                r += ",";
            }
            return r + ")";
        }
    }
}
