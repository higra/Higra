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

#include <algorithm>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include "array_view.hpp"

namespace hg {

    namespace array_internal {

        // Arrays with more elements than this are summarized.
        inline constexpr index_t print_threshold = 1000;
        // Number of leading and trailing entries kept on each axis of a summarized array.
        inline constexpr index_t print_edge_items = 3;

        template<class T>
        std::string format_element(const T &v) {
            std::ostringstream s;
            s.precision(8);
            if constexpr (std::is_same_v<T, bool>) {
                s << (v ? "true" : "false");
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
                s << static_cast<int>(v);
            } else {
                s << v;
            }
            return s.str();
        }

        // Positions printed along an axis of size n; -1 marks the "..." gap.
        inline std::vector<index_t> printed_positions(index_t n, bool summarize) {
            std::vector<index_t> r;
            if (summarize && n > 2 * print_edge_items) {
                for (index_t i = 0; i < print_edge_items; ++i) r.push_back(i);
                r.push_back(-1);
                for (index_t i = n - print_edge_items; i < n; ++i) r.push_back(i);
            } else {
                for (index_t i = 0; i < n; ++i) r.push_back(i);
            }
            return r;
        }

        template<class T>
        void collect_widths(const T *data, const array_shape &shape, index_t axis, index_t offset,
                            bool summarize, std::size_t &width) {
            index_t stride = 1;
            for (index_t k = axis + 1; k < shape.size(); ++k) stride *= shape[k];
            for (index_t i: printed_positions(shape[axis], summarize)) {
                if (i < 0) continue;
                if (axis + 1 == shape.size()) {
                    width = std::max(width, format_element(data[offset + i]).size());
                } else {
                    collect_widths(data, shape, axis + 1, offset + i * stride, summarize, width);
                }
            }
        }

        template<class T>
        void print_axis(std::ostream &os, const T *data, const array_shape &shape, index_t axis, index_t offset,
                        bool summarize, std::size_t width) {
            const index_t ndim = shape.size();
            index_t stride = 1;
            for (index_t k = axis + 1; k < ndim; ++k) stride *= shape[k];
            os << '[';
            bool first = true;
            for (index_t i: printed_positions(shape[axis], summarize)) {
                if (!first) {
                    if (axis + 1 == ndim) {
                        os << ", ";
                    } else {
                        os << ',' << std::string(static_cast<std::size_t>(ndim - axis - 1), '\n')
                           << std::string(static_cast<std::size_t>(axis + 1), ' ');
                    }
                }
                first = false;
                if (i < 0) {
                    os << "...";
                } else if (axis + 1 == ndim) {
                    std::string e = format_element(data[offset + i]);
                    os << std::string(width - e.size(), ' ') << e;
                } else {
                    print_axis(os, data, shape, axis + 1, offset + i * stride, summarize, width);
                }
            }
            os << ']';
        }
    }

    /**
     * Prints the elements in a NumPy-like layout: nested brackets, one row per line, elements
     * right-aligned. Arrays of more than 1000 elements are summarized: only the first and last 3
     * entries of each axis are printed, separated by ``...``.
     */
    template<class T>
    std::ostream &operator<<(std::ostream &os, array_view<const T> a) {
        if (!is_cpu_accessible(a.device())) {
            return os << "<array on a non-CPU device>";
        }
        if (a.dimension() == 0) {
            return os << array_internal::format_element(a.data()[0]);
        }
        if (a.size() == 0) {
            std::string r;
            for (index_t k = 0; k < a.dimension(); ++k) r += '[';
            for (index_t k = 0; k < a.dimension(); ++k) r += ']';
            return os << r << " (shape " << array_internal::shape_to_string(a.shape()) << ")";
        }
        const bool summarize = a.size() > array_internal::print_threshold;
        std::size_t width = 0;
        array_internal::collect_widths(a.data(), a.shape(), 0, 0, summarize, width);
        array_internal::print_axis(os, a.data(), a.shape(), 0, 0, summarize, width);
        return os;
    }
}
