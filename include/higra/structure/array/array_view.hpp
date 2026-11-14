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

#include <type_traits>
#include "../../utils.hpp"
#include "device.hpp"
#include "shape.hpp"
#include "parallel.hpp"

namespace hg {

    template<class T>
    class array;

    template<class T>
    class array_view;

    namespace array_internal {

        template<class T>
        using identity_t = typename std::common_type<T>::type;

        inline void assert_cpu_accessible(const device &d) {
            hg_assert(is_cpu_accessible(d), "Array elements are not accessible from the CPU.");
        }

        /**
         * Sequential at every size: the elements stay in the caches of the calling thread, which usually
         * consumes them next. A parallel fill distributes them over the caches of the TBB workers, and a
         * sequential loop that then reads a mid-sized array (up to a few MB) becomes several times slower.
         */
        template<class T>
        void fill(T *data, index_t n, const T &value) {
            for (index_t i = 0; i < n; ++i) {
                data[i] = value;
            }
        }
    }

    /**
     * Read-only, non-owning view of a C-contiguous (row-major) array of ``T``.
     *
     * A view stores a pointer to the first element, a shape and a device. It never owns
     * the elements: the storage must outlive the view.
     *
     * ``array_view<const T>`` is the common base of ``array_view<T>`` and ``array<T>``, so a function
     * ``template<class T> f(array_view<const T> a)`` accepts arrays, views and temporaries of any
     * of these types with a single instantiation per element type.
     *
     * Element access with ``operator()``, ``operator[]`` and ``row`` is checked (rank and bounds)
     * only when ``HG_ARRAY_CHECKS`` is defined (Debug builds). Sub-views created with ``slice`` and
     * ``reshaped`` are always checked.
     */
    template<class T>
    class array_view<const T> {
        static_assert(!std::is_const_v<T> && !std::is_volatile_v<T> && !std::is_reference_v<T>,
                      "array_view element type must be an unqualified object type.");
        static_assert(std::is_trivially_copyable_v<T>, "Array elements must be trivially copyable.");

    public:
        using value_type = T;
        using element_type = const T;
        using const_pointer = const T *;
        using const_iterator = const T *;
        using const_view_type = array_view<const T>;

        /**
         * Empty 1d view (shape ``(0,)``).
         */
        HG_HOST_DEVICE array_view() noexcept: m_data(nullptr), m_shape(index_t(0)), m_size(0), m_device() {
        }

        /**
         * View of the elements ``data[0 .. shape.numel())`` with the given shape.
         */
        HG_HOST_DEVICE array_view(const T *data, const array_shape &shape, hg::device dev = cpu_device) noexcept:
                m_data(data), m_shape(shape), m_size(shape.numel()), m_device(dev) {
        }

        HG_HOST_DEVICE const array_shape &shape() const noexcept {
            return m_shape;
        }

        HG_HOST_DEVICE index_t shape(index_t k) const {
            return m_shape[k];
        }

        HG_HOST_DEVICE index_t dimension() const noexcept {
            return m_shape.size();
        }

        /**
         * Number of elements.
         */
        HG_HOST_DEVICE index_t size() const noexcept {
            return m_size;
        }

        HG_HOST_DEVICE bool empty() const noexcept {
            return size() == 0;
        }

        HG_HOST_DEVICE hg::device device() const noexcept {
            return m_device;
        }

        HG_HOST_DEVICE const T *data() const noexcept {
            return m_data;
        }

        HG_HOST_DEVICE const T *begin() const noexcept {
            return m_data;
        }

        HG_HOST_DEVICE const T *end() const noexcept {
            return m_data + size();
        }

        /**
         * Element at the given indices; exactly ``dimension()`` integral indices are expected.
         */
        template<class... I>
        HG_HOST_DEVICE const T &operator()(I... idx) const {
            return m_data[offset(idx...)];
        }

        /**
         * Element at the given position in the flat (row-major) order.
         */
        HG_HOST_DEVICE const T &operator[](index_t flat_index) const {
            check_flat(flat_index);
            return m_data[flat_index];
        }

        /**
         * Number of elements of a row (``size() / shape(0)``, computed without division):
         * 1 for 1d and 0d views.
         */
        HG_HOST_DEVICE index_t row_size() const noexcept {
            index_t n = 1;
            for (index_t k = 1; k < m_shape.size(); ++k) {
                n *= m_shape[k];
            }
            return n;
        }

        /**
         * View of ``a(i, ...)``: one dimension less, shape ``shape[1:]``.
         */
        HG_HOST_DEVICE array_view<const T> row(index_t i) const {
            return array_view<const T>(m_data + row_offset(i), tail_shape(), m_device);
        }

        /**
         * View of ``a[b:e, ...]``: same rank, first dimension ``e - b``.
         */
        array_view<const T> slice(index_t b, index_t e) const {
            return array_view<const T>(m_data + slice_offset(b, e), slice_shape(b, e), m_device);
        }

        /**
         * View of the same elements with another shape of the same number of elements.
         */
        array_view<const T> reshaped(const array_shape &shape) const {
            check_reshape(shape);
            return array_view<const T>(m_data, shape, m_device);
        }

        /**
         * 1d view of all elements.
         */
        array_view<const T> flat() const {
            return array_view<const T>(m_data, array_shape(size()), m_device);
        }

        /**
         * Read-only view of this object.
         */
        HG_HOST_DEVICE array_view<const T> view() const noexcept {
            return *this;
        }

    protected:

        template<class... I>
        HG_HOST_DEVICE index_t offset(I... idx) const {
            static_assert((array_internal::is_element_index_v<I> && ...), "Array indices must be integers.");
            static_assert(sizeof...(I) <= (std::size_t) max_rank, "Too many array indices.");
#ifdef HG_ARRAY_CHECKS
            hg_assert((index_t) sizeof...(I) == m_shape.size(),
                      "Array accessed with " + std::to_string(sizeof...(I)) + " indices but its dimension is " +
                      std::to_string(m_shape.size()) + ".");
            {
                index_t k = 0;
                (check_index((index_t) idx, k++), ...);
            }
#endif
            return offset_impl(idx...);
        }

        HG_HOST_DEVICE void check_index(index_t i, index_t k) const {
            hg_assert(i >= 0 && i < m_shape[k],
                      "Array index " + std::to_string(i) + " out of bounds for axis " + std::to_string(k) +
                      " of size " + std::to_string(m_shape[k]) + ".");
        }

        HG_HOST_DEVICE index_t offset_impl() const noexcept {
            return 0;
        }

        template<class I0, class... I>
        HG_HOST_DEVICE index_t offset_impl(I0 i0, I... idx) const noexcept {
            index_t off = (index_t) i0;
            index_t k = 1;
            ((off = off * m_shape[k++] + (index_t) idx), ...);
            (void) k;
            return off;
        }

        HG_HOST_DEVICE void check_flat(index_t flat_index) const {
#ifdef HG_ARRAY_CHECKS
            hg_assert(flat_index >= 0 && flat_index < size(), "Array flat index out of bounds.");
#else
            (void) flat_index;
#endif
        }

        HG_HOST_DEVICE index_t row_offset(index_t i) const {
#ifdef HG_ARRAY_CHECKS
            hg_assert(m_shape.size() >= 1, "row() requires an array of dimension at least 1.");
            hg_assert(i >= 0 && i < m_shape[0], "Array row index out of bounds.");
#endif
            return i * row_size();
        }

        HG_HOST_DEVICE array_shape tail_shape() const noexcept {
            return m_shape.tail();
        }

        index_t slice_offset(index_t b, index_t e) const {
            hg_assert(m_shape.size() >= 1, "slice() requires an array of dimension at least 1.");
            hg_assert(0 <= b && b <= e && e <= m_shape[0],
                      "Invalid slice [" + std::to_string(b) + ", " + std::to_string(e) + ") of an array of shape " +
                      array_internal::shape_to_string(m_shape) + ".");
            return b * row_size();
        }

        array_shape slice_shape(index_t b, index_t e) const {
            array_shape s = m_shape;
            s[0] = e - b;
            return s;
        }

        void check_reshape(const array_shape &shape) const {
            hg_assert(shape.numel() == size(),
                      "Cannot reshape an array of shape " + array_internal::shape_to_string(m_shape) + " to shape " +
                      array_internal::shape_to_string(shape) + ".");
        }

        /**
         * Sets the shape and the cached number of elements (every shape change goes through it).
         */
        HG_HOST_DEVICE void set_shape(const array_shape &shape) noexcept {
            m_shape = shape;
            m_size = shape.numel();
        }

        const T *m_data;
        array_shape m_shape;
        index_t m_size; // number of elements, cached: size() and end() are used in loop conditions
        hg::device m_device;
    };

    /**
     * Mutable, non-owning view of a C-contiguous (row-major) array of ``T``.
     *
     * A mutable view can be created from a mutable array, another mutable view, or a pointer
     * to mutable storage; it cannot be created from a const array or from a temporary array.
     *
     * Constness is deep: a ``const array_view<T>`` gives read-only access to the elements. Copying
     * a view, however, gives a new mutable view: ``const`` protects against accidental writes, it
     * does not prevent them.
     */
    template<class T>
    class array_view : public array_view<const T> {
        using base_type = array_view<const T>;

    public:
        using value_type = T;
        using element_type = T;
        using pointer = T *;
        using iterator = T *;
        using view_type = array_view<T>;

        HG_HOST_DEVICE array_view() noexcept = default;

        HG_HOST_DEVICE array_view(T *data, const array_shape &shape, hg::device dev = cpu_device) noexcept:
                base_type(data, shape, dev) {
        }

        HG_HOST_DEVICE array_view(array<T> &a) noexcept: base_type(a.data(), a.shape(), a.device()) {
        }

        array_view(const array<T> &) = delete;

        array_view(array<T> &&) = delete;

        array_view(const array<T> &&) = delete;

        array_view(const array_view &) = default;

        array_view &operator=(const array_view &) = default;

        HG_HOST_DEVICE array_view &operator=(array<T> &a) noexcept {
            return *this = array_view(a);
        }

        array_view &operator=(const array<T> &) = delete;

        array_view &operator=(array<T> &&) = delete;

        array_view &operator=(const array<T> &&) = delete;

        using base_type::data;
        using base_type::begin;
        using base_type::end;
        using base_type::operator();
        using base_type::operator[];
        using base_type::row;
        using base_type::slice;
        using base_type::reshaped;
        using base_type::flat;
        using base_type::view;

        HG_HOST_DEVICE T *data() noexcept {
            return mutable_data();
        }

        HG_HOST_DEVICE T *begin() noexcept {
            return mutable_data();
        }

        HG_HOST_DEVICE T *end() noexcept {
            return mutable_data() + this->size();
        }

        template<class... I>
        HG_HOST_DEVICE T &operator()(I... idx) {
            return mutable_data()[this->offset(idx...)];
        }

        HG_HOST_DEVICE T &operator[](index_t flat_index) {
            this->check_flat(flat_index);
            return mutable_data()[flat_index];
        }

        HG_HOST_DEVICE array_view<T> row(index_t i) {
            return array_view<T>(mutable_data() + this->row_offset(i), this->tail_shape(), this->m_device);
        }

        array_view<T> slice(index_t b, index_t e) {
            return array_view<T>(mutable_data() + this->slice_offset(b, e), this->slice_shape(b, e), this->m_device);
        }

        array_view<T> reshaped(const array_shape &shape) {
            this->check_reshape(shape);
            return array_view<T>(mutable_data(), shape, this->m_device);
        }

        array_view<T> flat() {
            return array_view<T>(mutable_data(), array_shape(this->size()), this->m_device);
        }

        HG_HOST_DEVICE array_view<T> view() noexcept {
            return array_view<T>(mutable_data(), this->m_shape, this->m_device);
        }

        /**
         * Sets every element to ``value``.
         */
        void fill(const T &value) {
            array_internal::assert_cpu_accessible(this->m_device);
            array_internal::fill(mutable_data(), this->size(), value);
        }

    private:
        // Sound: a mutable view is only ever built from mutable storage.
        HG_HOST_DEVICE T *mutable_data() const noexcept {
            return const_cast<T *>(this->m_data);
        }
    };
}
