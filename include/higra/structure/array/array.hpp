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

#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include "../../utils.hpp"
#include "allocator.hpp"
#include "array_view.hpp"

namespace hg {

    namespace array_internal {

        template<class T>
        struct is_initializer_list : std::false_type {
        };

        template<class T>
        struct is_initializer_list<std::initializer_list<T>> : std::true_type {
        };

        template<class T, std::size_t Depth>
        struct nested_initializer_list {
            using type = std::initializer_list<typename nested_initializer_list<T, Depth - 1>::type>;
        };

        template<class T>
        struct nested_initializer_list<T, 0> {
            using type = T;
        };

        template<class T, std::size_t Depth>
        using nested_initializer_list_t = typename nested_initializer_list<T, Depth>::type;

        // Shape of a nested list, read from the first element of each level.
        template<class L>
        void nested_list_shape(const L &list, index_t *dims) {
            if constexpr (is_initializer_list<L>::value) {
                dims[0] = static_cast<index_t>(list.size());
                if (list.size() > 0) {
                    nested_list_shape(*list.begin(), dims + 1);
                } else {
                    using inner = typename L::value_type;
                    if constexpr (is_initializer_list<inner>::value) {
                        nested_list_shape(inner{}, dims + 1);
                    }
                }
            }
        }

        // Throws if any sub-list differs in length from the shape.
        template<class L>
        void check_nested_list(const L &list, const index_t *dims) {
            if constexpr (is_initializer_list<L>::value) {
                hg_assert(static_cast<index_t>(list.size()) == dims[0],
                          "Ragged nested initializer list: all sub-lists of a level must have the same length.");
                for (const auto &sub: list) {
                    check_nested_list(sub, dims + 1);
                }
            }
        }

        template<class T, class L>
        void copy_nested_list(const L &list, T *&out) {
            if constexpr (is_initializer_list<L>::value) {
                for (const auto &sub: list) {
                    copy_nested_list(sub, out);
                }
            } else {
                *out++ = list;
            }
        }

        template<class T>
        void copy_elements(T *dst, const T *src, index_t n) {
            detail::parallel_blocks(n, 4096, [dst, src](index_t b, index_t e) {
                std::memcpy(dst + b, src + b, static_cast<std::size_t>(e - b) * sizeof(T));
            });
        }
    }

    /**
     * Owning, C-contiguous (row-major) array of ``T`` with value semantics.
     *
     * Copying an array copies its elements (into storage from the same allocator); moving an
     * array transfers its buffer. An ``array<T>`` converts to ``array_view<T>`` and
     * ``array_view<const T>`` and provides all their accessors.
     *
     * Brace initialization always describes **data**, never a shape:
     * ``array<int>{5}`` is the 1d array ``[5]`` and ``array<int>{{1, 2}, {3, 4}}`` is a 2 x 2
     * array. Nested lists of up to 4 levels are accepted; ragged lists throw.
     * Use ``array<T>(array_shape)``, ``from_shape``, ``hg::empty`` or ``hg::zeros`` to create an
     * array of a given shape.
     *
     * A zero-size array allocates nothing and its ``data()`` may be ``nullptr``.
     */
    template<class T>
    class array : public array_view<T> {
        using view_base = array_view<T>;

    public:
        using value_type = T;

        /**
         * Empty 1d array (shape ``(0,)``); allocates nothing.
         */
        array() noexcept: view_base(), m_alloc(&cpu_allocator()) {
        }

        /**
         * Array of the given shape with uninitialized elements.
         */
        explicit array(const array_shape &shape, allocator &alloc = cpu_allocator()) : view_base(), m_alloc(&alloc) {
            allocate(shape);
        }

        /**
         * Array of the given shape with all elements equal to ``value``.
         */
        array(const array_shape &shape, const T &value, allocator &alloc = cpu_allocator()) : array(shape, alloc) {
            this->fill(value);
        }

        /**
         * Copy of the elements of a view.
         */
        explicit array(array_view<const T> source, allocator &alloc = cpu_allocator()) : array(source.shape(), alloc) {
            array_internal::assert_cpu_accessible(source.device());
            array_internal::copy_elements(this->data(), source.data(), this->size());
        }

        /**
         * 1d array with the given elements.
         *
         * This constructor is a template only so that overload resolution prefers the 2d literal
         * constructor when both match: ``array<int>{{1}}`` is ``[[1]]`` and ``{{1}, {2}}`` is a
         * 2 x 1 array, as in NumPy.
         */
        template<class Dummy = void>
        array(std::initializer_list<T> values) : array() {
            init_from_list(values);
        }

        /**
         * 2d array with the given rows; ragged lists throw.
         */
        array(array_internal::nested_initializer_list_t<T, 2> values) : array() {
            init_from_list(values);
        }

        /**
         * 3d array; ragged lists throw.
         */
        array(array_internal::nested_initializer_list_t<T, 3> values) : array() {
            init_from_list(values);
        }

        /**
         * 4d array; ragged lists throw.
         */
        array(array_internal::nested_initializer_list_t<T, 4> values) : array() {
            init_from_list(values);
        }

        /**
         * Deep copy, with the allocator of ``other``.
         */
        array(const array &other) : array(other.shape(), *other.m_alloc) {
            array_internal::copy_elements(this->data(), other.data(), this->size());
        }

        /**
         * Takes the buffer of ``other``, which becomes an empty 1d array.
         */
        array(array &&other) noexcept: view_base(static_cast<view_base &>(other)), m_alloc(other.m_alloc) {
            other.reset_to_empty();
        }

        /**
         * Deep copy; this array then uses the allocator of ``other``.
         */
        array &operator=(const array &other) {
            if (this != &other) {
                if (m_alloc == other.m_alloc && this->size() == other.size()) {
                    this->set_shape(other.m_shape);
                    array_internal::copy_elements(this->data(), other.data(), this->size());
                } else {
                    array tmp(other);
                    swap(tmp);
                }
            }
            return *this;
        }

        array &operator=(array &&other) noexcept {
            if (this != &other) {
                free_storage();
                static_cast<view_base &>(*this) = static_cast<view_base &>(other);
                m_alloc = other.m_alloc;
                other.reset_to_empty();
            }
            return *this;
        }

        ~array() {
            free_storage();
        }

        /**
         * Array of the given shape with uninitialized elements.
         */
        static array from_shape(const array_shape &shape) {
            return array(shape);
        }

        /**
         * Changes the shape. The elements are unspecified afterwards, unless the number of
         * elements is unchanged (then they are kept, like ``reshape``).
         */
        void resize(const array_shape &shape) {
            if (array_internal::checked_numel(shape, sizeof(T)) == this->size()) {
                this->set_shape(shape);
            } else {
                array tmp(shape, *m_alloc);
                swap(tmp);
            }
        }

        /**
         * Changes the shape in place; the number of elements must be unchanged.
         */
        void reshape(const array_shape &shape) {
            this->check_reshape(shape);
            this->set_shape(shape);
        }

        allocator &get_allocator() const noexcept {
            return *m_alloc;
        }

        void swap(array &other) noexcept {
            std::swap(static_cast<view_base &>(*this), static_cast<view_base &>(other));
            std::swap(m_alloc, other.m_alloc);
        }

        friend void swap(array &a, array &b) noexcept {
            a.swap(b);
        }

        /**
         * Buffer released by ``array::release``; the receiver must free ``data`` with
         * ``alloc->deallocate(data, bytes)``. ``data`` is ``nullptr`` if ``bytes == 0``.
         */
        struct released_buffer {
            void *data;
            std::size_t bytes;
            allocator *alloc;
            array_shape shape;
        };

        /**
         * Transfers the ownership of the buffer to the caller; the array becomes empty.
         */
        released_buffer release() &&{
            released_buffer r{this->data(), byte_size(), m_alloc, this->m_shape};
            reset_to_empty();
            return r;
        }

    private:

        std::size_t byte_size() const noexcept {
            return static_cast<std::size_t>(this->size()) * sizeof(T);
        }

        void allocate(const array_shape &shape) {
            const index_t n = array_internal::checked_numel(shape, sizeof(T));
            this->m_data = n > 0 ? static_cast<T *>(m_alloc->allocate(static_cast<std::size_t>(n) * sizeof(T)))
                                 : nullptr;
            this->set_shape(shape);
            this->m_device = m_alloc->device();
        }

        void free_storage() noexcept {
            if (this->m_data != nullptr) {
                m_alloc->deallocate(const_cast<T *>(this->m_data), byte_size());
            }
        }

        void reset_to_empty() noexcept {
            this->m_data = nullptr;
            this->set_shape(array_shape(index_t(0)));
            this->m_device = m_alloc->device();
        }

        template<class L>
        void init_from_list(const L &values) {
            index_t dims[max_rank] = {};
            array_internal::nested_list_shape(values, dims);
            constexpr index_t depth = list_depth<L>();
            array_shape shape;
            if constexpr (depth == 1) {
                shape = array_shape(dims[0]);
            } else if constexpr (depth == 2) {
                shape = array_shape(dims[0], dims[1]);
            } else if constexpr (depth == 3) {
                shape = array_shape(dims[0], dims[1], dims[2]);
            } else {
                shape = array_shape(dims[0], dims[1], dims[2], dims[3]);
            }
            array_internal::check_nested_list(values, dims);
            allocate(shape);
            T *out = this->data();
            array_internal::copy_nested_list(values, out);
        }

        template<class L>
        static constexpr index_t list_depth() {
            if constexpr (array_internal::is_initializer_list<L>::value) {
                return 1 + list_depth<typename L::value_type>();
            } else {
                return 0;
            }
        }

        allocator *m_alloc;
    };
}
