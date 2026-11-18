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

#include "../utils.hpp"
#include <cstddef>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

namespace hg {

    /**
     * Non-owning, one-dimensional sequence of ``size`` elements of type ``T`` separated by a constant
     * number of elements (the stride), for example a field of an array of structures.
     *
     * The span refers to existing elements and never owns them: the storage must outlive the span
     * and must not be reallocated while the span is used. ``T`` may be const-qualified.
     *
     * @tparam T element type
     */
    template<class T>
    class strided_span {
    public:
        using value_type = std::remove_cv_t<T>;
        using element_type = T;
        using reference = T &;
        using pointer = T *;

        /**
         * Random access iterator over the elements of a strided span.
         */
        class iterator {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type = std::remove_cv_t<T>;
            using difference_type = std::ptrdiff_t;
            using pointer = T *;
            using reference = T &;

            iterator() noexcept = default;

            iterator(T *ptr, index_t stride) noexcept: m_ptr(ptr), m_stride(stride) {
            }

            reference operator*() const noexcept {
                return *m_ptr;
            }

            pointer operator->() const noexcept {
                return m_ptr;
            }

            reference operator[](difference_type n) const noexcept {
                return m_ptr[n * m_stride];
            }

            iterator &operator++() noexcept {
                m_ptr += m_stride;
                return *this;
            }

            iterator operator++(int) noexcept {
                iterator r = *this;
                m_ptr += m_stride;
                return r;
            }

            iterator &operator--() noexcept {
                m_ptr -= m_stride;
                return *this;
            }

            iterator operator--(int) noexcept {
                iterator r = *this;
                m_ptr -= m_stride;
                return r;
            }

            iterator &operator+=(difference_type n) noexcept {
                m_ptr += n * m_stride;
                return *this;
            }

            iterator &operator-=(difference_type n) noexcept {
                m_ptr -= n * m_stride;
                return *this;
            }

            friend iterator operator+(iterator it, difference_type n) noexcept {
                return it += n;
            }

            friend iterator operator+(difference_type n, iterator it) noexcept {
                return it += n;
            }

            friend iterator operator-(iterator it, difference_type n) noexcept {
                return it -= n;
            }

            friend difference_type operator-(const iterator &a, const iterator &b) noexcept {
                return (a.m_ptr - b.m_ptr) / a.m_stride;
            }

            friend bool operator==(const iterator &a, const iterator &b) noexcept {
                return a.m_ptr == b.m_ptr;
            }

            friend bool operator!=(const iterator &a, const iterator &b) noexcept {
                return a.m_ptr != b.m_ptr;
            }

            friend bool operator<(const iterator &a, const iterator &b) noexcept {
                return a.m_stride > 0 ? a.m_ptr < b.m_ptr : a.m_ptr > b.m_ptr;
            }

            friend bool operator>(const iterator &a, const iterator &b) noexcept {
                return b < a;
            }

            friend bool operator<=(const iterator &a, const iterator &b) noexcept {
                return !(b < a);
            }

            friend bool operator>=(const iterator &a, const iterator &b) noexcept {
                return !(a < b);
            }

        private:
            T *m_ptr = nullptr;
            index_t m_stride = 1;
        };

        using const_iterator = iterator;

        /**
         * Empty span.
         */
        strided_span() noexcept = default;

        /**
         * Span of ``size`` elements, the ``i``-th one being ``data[i * stride]``.
         *
         * @param data pointer to the first element (may be null if ``size`` is 0)
         * @param stride distance between two consecutive elements, in number of elements of type ``T``
         * (must not be 0)
         * @param size number of elements
         */
        strided_span(T *data, index_t stride, index_t size) noexcept:
                m_data(data), m_stride(stride), m_size(size) {
        }

        /**
         * A span over a span of non-const elements is a span over const elements.
         */
        template<class U, class = std::enable_if_t<std::is_same<const U, T>::value && !std::is_const<U>::value>>
        strided_span(const strided_span<U> &other) noexcept:
                m_data(other.data()), m_stride(other.stride()), m_size(other.size()) {
        }

        /**
         * Pointer to the first element.
         */
        T *data() const noexcept {
            return m_data;
        }

        /**
         * Distance between two consecutive elements, in number of elements.
         */
        index_t stride() const noexcept {
            return m_stride;
        }

        /**
         * Number of elements.
         */
        index_t size() const noexcept {
            return m_size;
        }

        bool empty() const noexcept {
            return m_size == 0;
        }

        /**
         * The ``i``-th element (not checked).
         */
        T &operator()(index_t i) const noexcept {
            return m_data[i * m_stride];
        }

        /**
         * The ``i``-th element (not checked).
         */
        T &operator[](index_t i) const noexcept {
            return m_data[i * m_stride];
        }

        iterator begin() const noexcept {
            return iterator(m_data, m_stride);
        }

        iterator end() const noexcept {
            return iterator(m_data + m_size * m_stride, m_stride);
        }

        iterator cbegin() const noexcept {
            return begin();
        }

        iterator cend() const noexcept {
            return end();
        }

    private:
        T *m_data = nullptr;
        index_t m_stride = 1;
        index_t m_size = 0;
    };

    /**
     * Strided span over the member ``Member`` of an array of ``size`` structures starting at ``first``.
     *
     * Example: ``member_span(edges.data(), &edge_t::source, edges.size())``.
     *
     * @param first pointer to the first structure (may be null if ``size`` is 0)
     * @param member pointer to the data member
     * @param size number of structures
     */
    template<class S, class M>
    auto member_span(S *first, M std::remove_const_t<S>::*member, index_t size) noexcept {
        static_assert(sizeof(S) % sizeof(M) == 0,
                      "The structure size must be a multiple of the member size (is this a packed structure?).");
        using value_t = std::conditional_t<std::is_const<S>::value, const M, M>;
        value_t *data = (first == nullptr) ? nullptr : &(first->*member);
        return strided_span<value_t>(data, static_cast<index_t>(sizeof(S) / sizeof(M)), size);
    }

    /*
     * One-dimensional indexable sequences.
     *
     * Functions that read edge endpoints accept any 1d indexable sequence ``s``: an object with
     * ``s.size()`` and ``s(i)`` returning an integral value. Edge endpoints of an undirected graph are
     * strided spans, those of a tree are contiguous arrays; Python bindings pass contiguous arrays.
     */
    namespace indexable_1d_internal {
        template<class S, class = void>
        struct is_1d_indexable : std::false_type {
        };

        template<class S>
        struct is_1d_indexable<S, std::void_t<
                decltype(std::declval<const S &>().size()),
                decltype(std::declval<const S &>()(index_t(0)))>> : std::true_type {
        };

        template<class S, class = void>
        struct has_dimension : std::false_type {
        };

        template<class S>
        struct has_dimension<S, std::void_t<decltype(std::declval<const S &>().dimension())>> : std::true_type {
        };
    }

    /**
     * True if ``S`` has ``size()`` and ``operator()(index_t)``.
     */
    template<class S>
    constexpr bool is_1d_indexable_v = indexable_1d_internal::is_1d_indexable<std::decay_t<S>>::value;

    /**
     * Type of the elements of a 1d indexable sequence.
     */
    template<class S>
    using indexable_1d_value_t = std::decay_t<decltype(std::declval<const S &>()(index_t(0)))>;

    /**
     * Throws if ``s`` has a dimension other than 1 (sequences without ``dimension()``, such as
     * ``strided_span``, are one-dimensional).
     */
    template<class S>
    void assert_1d_indexable(const S &s, const char *name) {
        if constexpr (indexable_1d_internal::has_dimension<S>::value) {
            hg_assert(s.dimension() == 1, std::string("The array '") + name + "' must be 1d.");
        }
    }
}
