/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Tests of the temporary hg::bridge between Higra arrays and xtensor.

#include "test_array_utils.hpp"
#include "higra/structure/array/xtensor_bridge.hpp"

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/views/xindex_view.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "xtensor/views/xview.hpp"

#include <type_traits>
#include <utility>

namespace test_xtensor_bridge {

    using hg::array;
    using hg::array_shape;
    using hg::array_view;
    namespace bridge = hg::bridge;

    // Adaptors of read-only sources give const element references: writes do not compile.
    using const_adaptor = decltype(bridge::as_xt(std::declval<array_view<const double>>()));
    using mutable_adaptor = decltype(bridge::as_xt(std::declval<array_view<double>>()));
    using const_array_adaptor = decltype(bridge::as_xt(std::declval<const array<double> &>()));
    using mutable_array_adaptor = decltype(bridge::as_xt(std::declval<array<double> &>()));
    using const_fixed_adaptor = decltype(bridge::as_xt<2>(std::declval<array_view<const double>>()));
    using mutable_fixed_adaptor = decltype(bridge::as_xt<2>(std::declval<array_view<double>>()));

    template<class A>
    constexpr bool element_is_const =
            std::is_const_v<std::remove_reference_t<decltype(std::declval<A &>()(0))>>;

    static_assert(element_is_const<const_adaptor>);
    static_assert(element_is_const<const_array_adaptor>);
    static_assert(element_is_const<const_fixed_adaptor>);
    static_assert(!element_is_const<mutable_adaptor>);
    static_assert(!element_is_const<mutable_array_adaptor>);
    static_assert(!element_is_const<mutable_fixed_adaptor>);
    static_assert(std::is_same_v<typename const_fixed_adaptor::shape_type, std::array<std::size_t, 2>>);

    // The adaptors hold the pointer by value (not a reference to a temporary pointer variable).
    template<class B>
    struct buffer_pointer;

    template<class CP, class O, class A>
    struct buffer_pointer<xt::xbuffer_adaptor<CP, O, A>> {
        using type = CP;
    };

    template<class A>
    using storage_pointer = typename buffer_pointer<std::decay_t<typename A::storage_type>>::type;

    static_assert(std::is_same_v<storage_pointer<const_adaptor>, const double *>);
    static_assert(std::is_same_v<storage_pointer<mutable_adaptor>, double *>);
    static_assert(std::is_same_v<storage_pointer<const_array_adaptor>, const double *>);
    static_assert(std::is_same_v<storage_pointer<mutable_array_adaptor>, double *>);
    static_assert(std::is_same_v<storage_pointer<const_fixed_adaptor>, const double *>);
    static_assert(std::is_same_v<storage_pointer<mutable_fixed_adaptor>, double *>);

    template<class Arg, class = void>
    struct can_adapt : std::false_type {
    };

    template<class Arg>
    struct can_adapt<Arg, std::void_t<decltype(bridge::as_xt(std::declval<Arg>()))>> : std::true_type {
    };

    // Adapting a temporary array would dangle.
    static_assert(can_adapt<array<double> &>::value);
    static_assert(can_adapt<const array<double> &>::value);
    static_assert(!can_adapt<array<double>>::value);

    template<class Arg, class = void>
    struct can_view : std::false_type {
    };

    template<class Arg>
    struct can_view<Arg, std::void_t<decltype(bridge::as_view(std::declval<Arg>()))>> : std::true_type {
    };

    // A view of a temporary container would dangle; lazy expressions and adaptors are accepted.
    static_assert(can_view<xt::xarray<double> &>::value);
    static_assert(can_view<const xt::xtensor<double, 2> &>::value);
    static_assert(!can_view<xt::xarray<double>>::value);
    static_assert(!can_view<xt::xtensor<double, 2>>::value);
    static_assert(can_view<decltype(std::declval<xt::xarray<double> &>() + 1)>::value);
    static_assert(can_view<const_adaptor>::value);

    TEST_CASE("bridge as_xt adapts views without copy", "[array][bridge]") {
        array<double> a{{1, 2, 3}, {4, 5, 6}};

        auto x = bridge::as_xt(a.view());
        REQUIRE(x.dimension() == 2);
        REQUIRE(x.shape()[0] == 2);
        REQUIRE(x.shape()[1] == 3);
        REQUIRE(x.data() == a.data());
        REQUIRE(xt::sum(x)() == 21);
        REQUIRE(x(1, 2) == 6);

        auto m = bridge::as_xt(a);
        m(0, 1) = 20;
        REQUIRE(a(0, 1) == 20);
        m += 1;
        REQUIRE(a(1, 2) == 7);

        array_view<double> v = a;
        auto mv = bridge::as_xt(v);
        mv(0, 0) = -1;
        REQUIRE(a(0, 0) == -1);

        auto f = bridge::as_xt<2>(a.view());
        REQUIRE(f.dimension() == 2);
        REQUIRE(f.data() == a.data());
        REQUIRE(xt::amax(f)() == 21);
        auto fm = bridge::as_xt<2>(a);
        fm(1, 0) = 100;
        REQUIRE(a(1, 0) == 100);

        REQUIRE_THROWS(bridge::as_xt<1>(a.view()));
        REQUIRE_THROWS(bridge::as_xt<3>(a));
    }

    TEST_CASE("bridge as_xt rank 0 and zero size", "[array][bridge]") {
        array<int> s(array_shape{});
        s.fill(7);
        auto xs = bridge::as_xt(s.view());
        REQUIRE(xs.dimension() == 0);
        REQUIRE(xs.size() == 1);
        REQUIRE(xs() == 7);
        auto fs = bridge::as_xt<0>(s.view());
        REQUIRE(fs() == 7);

        array<int> e(array_shape(3, 0));
        auto xe = bridge::as_xt(e.view());
        REQUIRE(xe.dimension() == 2);
        REQUIRE(xe.size() == 0);
        REQUIRE(xe.shape()[0] == 3);
    }

    TEST_CASE("bridge as_view of contiguous storage is zero-copy", "[array][bridge]") {
        xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}};

        auto h = bridge::as_view(a);
        REQUIRE(h.is_zero_copy());
        REQUIRE(h.view.data() == a.data());
        REQUIRE(h.view.shape() == array_shape(2, 3));

        xt::xtensor<int, 1> t = {1, 2, 3, 4};
        auto ht = bridge::as_view(t);
        REQUIRE(ht.is_zero_copy());
        REQUIRE(ht.view.data() == t.data());

        // contiguous rows of a row-major container
        auto rows = xt::view(a, xt::range(1, 2), xt::all());
        auto hr = bridge::as_view(rows);
        REQUIRE(hr.is_zero_copy());
        REQUIRE(hr.view.data() == a.data() + 3);
        REQUIRE(hr.view == array<double>{{4, 5, 6}});

        // adaptor over an hg array
        array<double> b{1, 2, 3};
        auto ad = bridge::as_xt(b.view());
        auto hb = bridge::as_view(ad);
        REQUIRE(hb.is_zero_copy());
        REQUIRE(hb.view.data() == b.data());

        // moving the holder keeps the view valid
        auto moved = std::move(h);
        REQUIRE(moved.view.data() == a.data());
    }

    TEST_CASE("bridge as_view evaluates other expressions", "[array][bridge]") {
        xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}};

        auto lazy = bridge::as_view(a * 2 + 1);
        REQUIRE_FALSE(lazy.is_zero_copy());
        REQUIRE(lazy.view == array<double>{{3, 5, 7}, {9, 11, 13}});

        auto column = bridge::as_view(xt::view(a, xt::all(), 1));
        REQUIRE_FALSE(column.is_zero_copy());
        REQUIRE(column.view == array<double>{2, 5});

        auto transposed = bridge::as_view(xt::transpose(a));
        REQUIRE_FALSE(transposed.is_zero_copy());
        REQUIRE(transposed.view == array<double>{{1, 4}, {2, 5}, {3, 6}});

        xt::xarray<double> flat = {10, 20, 30, 40};
        xt::xarray<std::size_t> idx = {3, 0};
        auto gathered = bridge::as_view(xt::index_view(flat, idx));
        REQUIRE_FALSE(gathered.is_zero_copy());
        REQUIRE(gathered.view == array<double>{40, 10});

        auto every_other = bridge::as_view(xt::view(flat, xt::range(0, 4, 2)));
        REQUIRE_FALSE(every_other.is_zero_copy());
        REQUIRE(every_other.view == array<double>{10, 30});

        auto reduced = bridge::as_view(xt::sum(a));
        REQUIRE(reduced.view.dimension() == 0);
        REQUIRE(reduced.view() == 21);

        // moving an owning holder keeps the view valid
        auto moved = std::move(lazy);
        REQUIRE(moved.view.data() == moved.storage->data());
        REQUIRE(moved.view(1, 2) == 13);

        auto b = bridge::as_view(xt::xarray<bool>{true, false} && xt::xarray<bool>{true, true});
        REQUIRE(b.view == array<bool>{true, false});
    }

    TEST_CASE("bridge copies", "[array][bridge]") {
        array<int> a{{1, 2}, {3, 4}};

        xt::xarray<int> x = bridge::to_xt(a);
        REQUIRE(x.data() != a.data());
        REQUIRE(x == xt::xarray<int>{{1, 2}, {3, 4}});
        x(0, 0) = 10;
        REQUIRE(a(0, 0) == 1);

        xt::xtensor<int, 2> t = bridge::to_xt<2>(std::move(a));
        REQUIRE(t == xt::xtensor<int, 2>{{1, 2}, {3, 4}});
        REQUIRE_THROWS(bridge::to_xt<1>(array<int>{{1, 2}}));

        xt::xarray<double> src = {{1.5, 2.5}};
        auto c = bridge::to_array(src);
        static_assert(std::is_same_v<decltype(c), array<double>>);
        REQUIRE(c.data() != src.data());
        REQUIRE(c == array<double>{{1.5, 2.5}});

        auto lazy = bridge::to_array(src * 2);
        REQUIRE(lazy == array<double>{{3, 5}});

        xt::xarray<double> big = xt::zeros<double>({1, 1, 1, 1, 1, 1, 1, 1, 1});
        REQUIRE_THROWS(bridge::to_array(big));
        REQUIRE_THROWS(bridge::as_view(big));
    }
}
