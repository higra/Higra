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
#include "../../utils.hpp"
#include "../array/array.hpp"
#include <type_traits>

namespace hg {

    namespace details {


        /**
         * An efficient and simple view over the first axis of a C-contiguous (row-major) array:
         * the view refers to one row ``data[position * row_size: (position + 1) * row_size]`` at a time.
         *
         * Mostly provide range functions (begin, end), returning raw pointers, and an assignment operator.
         *
         * The view must know at compile time if the underlying array has more than one dimension
         * (i.e. if the view contain a scalar or more elements) in order to perform compile time optimization.
         *
         * The view does not own the elements: the array must outlive it. Constness is shallow:
         * ``value_t`` is ``const V`` for a read-only view.
         *
         * @tparam vectorial indicates if the underlying array has more than 1 dimension
         * @tparam value_t element type (const-qualified for read-only views)
         */
        template<bool vectorial, typename value_t>
        struct light_axis_view {

            using value_type = std::remove_const_t<value_t>;
            using self_type = light_axis_view<vectorial, value_t>;

            static const bool is_vectorial = vectorial;

            /**
             * @param data pointer to the first element of the array
             * @param row_size number of elements in a row (ignored if not vectorial)
             * @param position initial row
             */
            light_axis_view(value_t *data, index_t row_size, index_t position = 0) :
                    m_data(data),
                    m_stride(vectorial ? row_size : 1),
                    m_position(position) {
            }

            void set_position(index_t i) {
                m_position = i;
            }

            value_t *begin() const {
                if constexpr (vectorial) {
                    return m_data + m_position * m_stride;
                } else {
                    return m_data + m_position;
                }
            }

            value_t *end() const {
                if constexpr (vectorial) {
                    return m_data + (m_position + 1) * m_stride;
                } else {
                    return m_data + (m_position + 1);
                }
            }

            template<bool vectorial2, typename T2>
            self_type &operator=(const light_axis_view<vectorial2, T2> &rhs) {
                static_assert(vectorial == vectorial2, "Mixing vectorial and non vectorial soft axis view!");
                assign(rhs);
                return (*this);
            }

            self_type &operator=(const self_type &rhs) {
                assign(rhs);
                return (*this);
            }

            template<bool vectorial2, typename T2, typename combinator_t>
            void combine(const light_axis_view<vectorial2, T2> &rhs, combinator_t fun) {
                static_assert(vectorial == vectorial2, "Mixing vectorial and non vectorial soft axis view!");
                if constexpr (vectorial) {
                    auto r = rhs.begin();
                    auto v = begin();
                    for (auto e = end(); v != e; v++, r++) {
                        *v = fun(*v, *r);
                    }
                } else {
                    *begin() = fun(*begin(), *(rhs.begin()));
                }
            }


        private:

            template<typename T2>
            void assign(const T2 &rhs) {
                static_assert(vectorial == T2::is_vectorial,
                              "Mixing vectorial and non vectorial soft axis view!");
                if constexpr (vectorial) {
                    auto r = rhs.begin();
                    auto v = begin();
                    for (auto e = end(); v != e; v++, r++) {
                        *v = *r;
                    }
                } else {
                    *begin() = *rhs.begin();
                }
            }

            value_t *m_data;
            index_t m_stride;
            index_t m_position;
        };

        template<typename T>
        index_t light_axis_view_row_size(const T &e) {
            if (e.dimension() == 0) {
                return 1;
            }
            const index_t n = (index_t) e.shape()[0];
            return (n == 0) ? 0 : (index_t) e.size() / n;
        }
    }

    /**
     * Light axis view over an array view.
     */
    template<bool vectorial = true, typename T>
    auto make_light_axis_view(const array_view<T> &e, index_t position = 0) {
        // a mutable view always refers to mutable storage
        return details::light_axis_view<vectorial, T>(const_cast<T *>(e.data()),
                                                      details::light_axis_view_row_size(e), position);
    }

    /**
     * Light axis view over an array.
     */
    template<bool vectorial = true, typename T>
    auto make_light_axis_view(array<T> &e, index_t position = 0) {
        return make_light_axis_view<vectorial>(array_view<T>(e), position);
    }

    template<bool vectorial = true, typename T>
    auto make_light_axis_view(const array<T> &e, index_t position = 0) {
        return make_light_axis_view<vectorial>(e.view(), position);
    }

}
