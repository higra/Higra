/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * TEMPORARY migration bridge between Higra arrays and xtensor.
 *
 * This header exists only while Higra moves from xtensor to its own array library; it is removed
 * together with xtensor. Do not use it in new code that does not need to reach an xtensor-based
 * function.
 *
 * Zero-copy functions (the result refers to the source elements, which must outlive it):
 *  - ``as_xt(view)``: ``xt::xarray_adaptor`` (dynamic rank) over the elements of a view or array;
 *    read-only for ``array_view<const T>`` and ``const array<T>&``, writeable for ``array_view<T>``
 *    and ``array<T>&``;
 *  - ``as_xt<N>(view)``: ``xt::xtensor_adaptor`` of rank ``N``; throws if the rank differs;
 *  - ``as_view(e)``: ``view_holder`` whose ``view`` is an ``array_view<const V>`` of an xtensor
 *    expression. It refers to the expression's elements when they are stored C-contiguously
 *    (containers, adaptors, contiguous strided views); otherwise the expression is evaluated into
 *    storage owned by the holder. Temporary ``xt::xarray`` / ``xt::xtensor`` containers are
 *    rejected at compile time (the view would dangle); use ``to_array`` for them.
 *
 * Copying functions:
 *  - ``to_xt(array)``, ``to_xt<N>(array)``: ``xt::xarray<T>`` / ``xt::xtensor<T, N>`` copy;
 *  - ``to_array(e)``: ``hg::array<V>`` copy of any xtensor expression (evaluated if lazy).
 *
 * Shapes are converted with checks: xtensor dimensions above INT64_MAX or more than
 * ``hg::max_rank`` dimensions throw.
 */

#pragma once

#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xnoalias.hpp"
#include "all.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace hg {
    namespace bridge {

        namespace bridge_internal {

            inline xt::dynamic_shape<std::size_t> xt_shape(const array_shape &shape) {
                return xt::dynamic_shape<std::size_t>(shape.begin(), shape.end());
            }

            template<std::size_t N>
            std::array<std::size_t, N> xt_fixed_shape(const array_shape &shape) {
                hg_assert(shape.size() == static_cast<index_t>(N),
                          "Expected an array of dimension " + std::to_string(N) + " but got dimension " +
                          std::to_string(shape.size()) + ".");
                std::array<std::size_t, N> r{};
                for (std::size_t k = 0; k < N; ++k) {
                    r[k] = static_cast<std::size_t>(shape[static_cast<index_t>(k)]);
                }
                return r;
            }

            // The pointer is passed to xt::adapt as an rvalue: given an lvalue, xt::adapt would
            // store a reference to the local variable ``data`` instead of a copy of the pointer.
            template<class P>
            auto adapt(P data, const array_shape &shape) {
                return xt::adapt(std::move(data), static_cast<std::size_t>(shape.numel()), xt::no_ownership(),
                                 xt_shape(shape));
            }

            template<std::size_t N, class P>
            auto adapt(P data, const array_shape &shape) {
                return xt::adapt(std::move(data), static_cast<std::size_t>(shape.numel()), xt::no_ownership(),
                                 xt_fixed_shape<N>(shape));
            }

            /**
             * True if the elements of ``e`` are stored in row-major order without gaps from
             * ``e.data() + e.data_offset()``; strides of dimensions of size 1 are ignored.
             */
            template<class E>
            bool is_c_contiguous(const E &e) {
                const auto &shape = e.shape();
                const auto &strides = e.strides();
                std::ptrdiff_t expected = 1;
                for (std::size_t k = shape.size(); k-- > 0;) {
                    if (shape[k] != 1) {
                        if (static_cast<std::ptrdiff_t>(strides[k]) != expected) {
                            return false;
                        }
                        expected *= static_cast<std::ptrdiff_t>(shape[k]);
                    }
                }
                return true;
            }

            template<class E, class = void>
            struct has_contiguous_storage_interface : std::false_type {
            };

            template<class E>
            struct has_contiguous_storage_interface<E, std::void_t<
                    decltype(std::declval<const E &>().data()),
                    decltype(std::declval<const E &>().data_offset()),
                    decltype(std::declval<const E &>().strides())>> : std::true_type {
            };
        }

        /**
         * Read-only ``xt::xarray_adaptor`` (dynamic rank) over the elements of a view; no copy.
         */
        template<class T>
        auto as_xt(array_view<const T> v) {
            array_internal::assert_cpu_accessible(v.device());
            return bridge_internal::adapt(v.data(), v.shape());
        }

        /**
         * Writeable ``xt::xarray_adaptor`` (dynamic rank) over the elements of a view; no copy.
         */
        template<class T>
        auto as_xt(array_view<T> v) {
            array_internal::assert_cpu_accessible(v.device());
            return bridge_internal::adapt(v.data(), v.shape());
        }

        template<class T>
        auto as_xt(const array<T> &a) {
            return as_xt(a.view());
        }

        template<class T>
        auto as_xt(array<T> &a) {
            return as_xt(array_view<T>(a));
        }

        /**
         * Adapting a temporary array would leave the adaptor dangling.
         */
        template<class T>
        void as_xt(array<T> &&) = delete;

        /**
         * Read-only ``xt::xtensor_adaptor`` of rank ``N`` over the elements of a view; no copy.
         * Throws if the view does not have dimension ``N``.
         */
        template<std::size_t N, class T>
        auto as_xt(array_view<const T> v) {
            array_internal::assert_cpu_accessible(v.device());
            return bridge_internal::adapt<N>(v.data(), v.shape());
        }

        /**
         * Writeable ``xt::xtensor_adaptor`` of rank ``N`` over the elements of a view; no copy.
         * Throws if the view does not have dimension ``N``.
         */
        template<std::size_t N, class T>
        auto as_xt(array_view<T> v) {
            array_internal::assert_cpu_accessible(v.device());
            return bridge_internal::adapt<N>(v.data(), v.shape());
        }

        template<std::size_t N, class T>
        auto as_xt(const array<T> &a) {
            return as_xt<N>(a.view());
        }

        template<std::size_t N, class T>
        auto as_xt(array<T> &a) {
            return as_xt<N>(array_view<T>(a));
        }

        template<std::size_t N, class T>
        void as_xt(array<T> &&) = delete;

        /**
         * Copy of any xtensor expression (evaluated if lazy) into a new ``hg::array``.
         */
        template<class E>
        auto to_array(const xt::xexpression<E> &expression) {
            const E &e = expression.derived_cast();
            using V = std::decay_t<typename E::value_type>;
            array<V> r(array_shape::from(e.shape()));
            auto target = bridge_internal::adapt(r.data(), r.shape());
            xt::noalias(target) = e;
            return r;
        }

        /**
         * Copy of an array into a new ``xt::xarray``.
         */
        template<class T>
        xt::xarray<T> to_xt(array_view<const T> v) {
            return as_xt(v);
        }

        /**
         * Copy of an array into a new ``xt::xtensor`` of rank ``N``; throws if the rank differs.
         */
        template<std::size_t N, class T>
        xt::xtensor<T, N> to_xt(array_view<const T> v) {
            return as_xt<N>(v);
        }

        /**
         * Result of ``as_view``: a read-only view of an xtensor expression and, when the
         * expression had to be evaluated, the storage the view refers to.
         *
         * The holder is movable but not copyable; moving it keeps ``view`` valid.
         */
        template<class V>
        struct view_holder {
            array_view<const V> view;
            std::optional<array<V>> storage;

            view_holder(array_view<const V> v) : view(v) {
            }

            view_holder(array<V> &&a) : storage(std::move(a)) {
                view = storage->view();
            }

            view_holder(const view_holder &) = delete;

            view_holder &operator=(const view_holder &) = delete;

            view_holder(view_holder &&) noexcept = default;

            view_holder &operator=(view_holder &&) noexcept = default;

            /**
             * True if ``view`` refers to the elements of the expression itself.
             */
            bool is_zero_copy() const noexcept {
                return !storage.has_value();
            }
        };

        /**
         * Read-only view of the elements of an xtensor expression, in row-major order.
         *
         * Zero-copy when the expression exposes C-contiguous storage (checked at run time from its
         * shape and strides): the expression's storage must then outlive the holder. Otherwise
         * the expression is evaluated into storage owned by the holder.
         */
        template<class E>
        auto as_view(const xt::xexpression<E> &expression) {
            const E &e = expression.derived_cast();
            using V = std::decay_t<typename E::value_type>;
            if constexpr (bridge_internal::has_contiguous_storage_interface<E>::value) {
                using pointer = decltype(e.data());
                if constexpr (std::is_same_v<std::decay_t<std::remove_pointer_t<pointer>>, V>) {
                    if (bridge_internal::is_c_contiguous(e)) {
                        const V *data = e.data() + e.data_offset();
                        return view_holder<V>(array_view<const V>(data, array_shape::from(e.shape())));
                    }
                }
            }
            return view_holder<V>(to_array(e));
        }

        /**
         * A view of a temporary container would dangle: use ``to_array`` instead.
         */
        template<class EC, xt::layout_type L, class SC, class Tag>
        void as_view(xt::xarray_container<EC, L, SC, Tag> &&) = delete;

        template<class EC, std::size_t N, xt::layout_type L, class Tag>
        void as_view(xt::xtensor_container<EC, N, L, Tag> &&) = delete;
    }
}
