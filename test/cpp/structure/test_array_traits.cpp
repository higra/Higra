/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Compile-time checks of the const-safety and deduction rules of the array library
// (array_view<const T> / array_view<T> / array<T>). A failing rule breaks the build.

#include "test_array_utils.hpp"

#include <type_traits>
#include <utility>

namespace test_array_traits {

    using hg::array;
    using hg::array_shape;
    using hg::array_view;

    template<class T>
    struct tag {
        using type = T;
    };

    // Probes: a read-only parameter and a mutable parameter, both deducing T (returned as tag<T>
    // so that const is kept).
    template<class T>
    tag<T> reader(array_view<const T>);

    template<class T>
    tag<T> writer(array_view<T>);

    template<class Arg, class = void>
    struct reader_deduces : std::false_type {
    };

    template<class Arg>
    struct reader_deduces<Arg, std::void_t<decltype(reader(std::declval<Arg>()))>> : std::true_type {
        using type = typename decltype(reader(std::declval<Arg>()))::type;
    };

    template<class Arg, class = void>
    struct writer_deduces : std::false_type {
    };

    template<class Arg>
    struct writer_deduces<Arg, std::void_t<decltype(writer(std::declval<Arg>()))>> : std::true_type {
        using type = typename decltype(writer(std::declval<Arg>()))::type;
    };

    template<class Arg, class = void>
    struct writer_int_accepts : std::false_type {
    };

    template<class Arg>
    struct writer_int_accepts<Arg, std::void_t<decltype(writer<int>(std::declval<Arg>()))>> : std::true_type {
    };

    // Can an element be written through an object of type V (an lvalue of V)?
    template<class V, class = void>
    struct element_writable : std::false_type {
    };

    template<class V>
    struct element_writable<V, std::void_t<decltype(std::declval<V &>()(0) = 1)>> : std::true_type {
    };

    template<class V, class = void>
    struct flat_writable : std::false_type {
    };

    template<class V>
    struct flat_writable<V, std::void_t<decltype(std::declval<V &>()[0] = 1)>> : std::true_type {
    };

    template<class V, class = void>
    struct fill_callable : std::false_type {
    };

    template<class V>
    struct fill_callable<V, std::void_t<decltype(std::declval<V &>().fill(1))>> : std::true_type {
    };

    template<class V, class = void>
    struct row_writable : std::false_type {
    };

    template<class V>
    struct row_writable<V, std::void_t<decltype(std::declval<V &>().row(0)(0) = 1)>> : std::true_type {
    };

    using A = array<int>;
    using MV = array_view<int>;
    using CV = array_view<const int>;

    // --- layout of views -----------------------------------------------------------------------
    static_assert(std::is_trivially_copyable_v<MV> && std::is_trivially_copyable_v<CV>);
    static_assert(std::is_standard_layout_v<MV> && std::is_standard_layout_v<CV>);
    static_assert(!std::is_polymorphic_v<MV> && !std::is_polymorphic_v<CV>);
    static_assert(std::is_trivially_copyable_v<array_shape>);
    static_assert(std::is_base_of_v<CV, MV> && std::is_base_of_v<MV, A> && std::is_base_of_v<CV, A>);
    static_assert(std::is_nothrow_move_constructible_v<A> && std::is_nothrow_move_assignable_v<A>);

    // --- reader(array_view<const T>) deduces T = int from the five argument kinds ---------------
    static_assert(reader_deduces<A &>::value && std::is_same_v<reader_deduces<A &>::type, int>);
    static_assert(reader_deduces<const A &>::value && std::is_same_v<reader_deduces<const A &>::type, int>);
    static_assert(reader_deduces<MV &>::value && std::is_same_v<reader_deduces<MV &>::type, int>);
    static_assert(reader_deduces<CV &>::value && std::is_same_v<reader_deduces<CV &>::type, int>);
    static_assert(reader_deduces<A>::value && std::is_same_v<reader_deduces<A>::type, int>); // temporary
    static_assert(reader_deduces<const MV &>::value && reader_deduces<MV>::value);

    // --- writer(array_view<T>) ------------------------------------------------------------------
    // the mutable array deduces T = int (most derived base, CWG 2303)
    static_assert(writer_deduces<A &>::value && std::is_same_v<writer_deduces<A &>::type, int>);
    static_assert(writer_deduces<MV &>::value && std::is_same_v<writer_deduces<MV &>::type, int>);
    // const arrays and temporaries are rejected, with deduction or with explicit T
    static_assert(!writer_deduces<const A &>::value);
    static_assert(!writer_deduces<A>::value);
    static_assert(!writer_deduces<A &&>::value);
    static_assert(!writer_int_accepts<const A &>::value);
    static_assert(!writer_int_accepts<A>::value);
    static_assert(writer_int_accepts<A &>::value);
    static_assert(!writer_int_accepts<CV &>::value);
    // with a read-only view, writer deduces T = const int: the parameter is then a read-only view
    static_assert(std::is_same_v<writer_deduces<CV &>::type, const int>);

    // --- conversions ----------------------------------------------------------------------------
    static_assert(std::is_convertible_v<A &, CV> && std::is_convertible_v<const A &, CV>);
    static_assert(std::is_convertible_v<A, CV>); // read-only view of a temporary (valid for a call)
    static_assert(std::is_convertible_v<MV, CV> && std::is_convertible_v<const MV &, CV>);
    static_assert(std::is_convertible_v<A &, MV>);
    static_assert(!std::is_convertible_v<const A &, MV> && !std::is_constructible_v<MV, const A &>);
    static_assert(!std::is_convertible_v<A, MV> && !std::is_constructible_v<MV, A>); // view of temporary
    static_assert(!std::is_constructible_v<MV, A &&> && !std::is_constructible_v<MV, const A &&>);
    static_assert(!std::is_convertible_v<CV, MV> && !std::is_constructible_v<MV, CV>);
    static_assert(!std::is_constructible_v<MV, const int *, array_shape>);
    static_assert(std::is_constructible_v<MV, int *, array_shape>);
    static_assert(std::is_constructible_v<CV, const int *, array_shape>);
    static_assert(!std::is_convertible_v<CV, A>); // deep copy from a view is explicit
    static_assert(std::is_constructible_v<A, CV>);
    static_assert(!std::is_convertible_v<array<double>, array_view<const int>>);

    // --- assignments rebinding a mutable view ---------------------------------------------------
    static_assert(std::is_assignable_v<MV &, A &>);
    static_assert(std::is_assignable_v<MV &, MV>);
    static_assert(!std::is_assignable_v<MV &, const A &>);
    static_assert(!std::is_assignable_v<MV &, A>);
    static_assert(!std::is_assignable_v<MV &, CV>);
    static_assert(std::is_assignable_v<CV &, const A &>);

    // --- deep const: elements are writable only through mutable non-const objects ---------------
    static_assert(element_writable<A>::value && element_writable<MV>::value);
    static_assert(!element_writable<const A>::value);
    static_assert(!element_writable<const MV>::value);
    static_assert(!element_writable<CV>::value && !element_writable<const CV>::value);
    static_assert(flat_writable<MV>::value && !flat_writable<const MV>::value && !flat_writable<CV>::value);
    static_assert(fill_callable<MV>::value && fill_callable<A>::value);
    static_assert(!fill_callable<const MV>::value && !fill_callable<const A>::value && !fill_callable<CV>::value);
    static_assert(row_writable<A>::value && row_writable<MV>::value);
    static_assert(!row_writable<const A>::value && !row_writable<const MV>::value && !row_writable<CV>::value);
    static_assert(std::is_same_v<decltype(std::declval<const MV &>().data()), const int *>);
    static_assert(std::is_same_v<decltype(std::declval<MV &>().data()), int *>);
    static_assert(std::is_same_v<decltype(std::declval<const A &>().begin()), const int *>);
    static_assert(std::is_same_v<decltype(std::declval<const MV &>().row(0)), CV>);
    static_assert(std::is_same_v<decltype(std::declval<MV &>().row(0)), MV>);
    static_assert(std::is_same_v<decltype(std::declval<A &>().slice(0, 1)), MV>);
    static_assert(std::is_same_v<decltype(std::declval<const A &>().slice(0, 1)), CV>);
    static_assert(std::is_same_v<decltype(std::declval<const A &>().view()), CV>);
    static_assert(std::is_same_v<decltype(std::declval<A &>().flat()), MV>);
    static_assert(std::is_same_v<decltype(std::declval<const MV &>().reshaped({1})), CV>);

    // --- shapes ---------------------------------------------------------------------------------
    static_assert(std::is_constructible_v<array_shape, int, int, int, int, int, int, int, int>);
    static_assert(!std::is_constructible_v<array_shape, int, int, int, int, int, int, int, int, int>);
    static_assert(std::is_constructible_v<array_shape, std::size_t, int, short, unsigned char>);
    static_assert(!std::is_constructible_v<array_shape, bool>);
    static_assert(!std::is_constructible_v<array_shape, double>);
    static_assert(std::is_convertible_v<std::size_t, array_shape>); // enables zeros<T>({n})

    // --- function result types ------------------------------------------------------------------
    struct positive {
        bool operator()(int x) const { return x > 0; }
    };

    struct product {
        double operator()(int x, double y) const { return x * y; }
    };

    static_assert(std::is_same_v<decltype(hg::map(positive(), std::declval<const A &>())), array<bool>>);
    static_assert(std::is_same_v<decltype(hg::map(product(), std::declval<A &>(), std::declval<array<double> &>())),
            array<double>>);
    static_assert(std::is_same_v<decltype(hg::take(std::declval<const A &>(), std::declval<array<long>>())), A>);
    static_assert(std::is_same_v<decltype(hg::sum(std::declval<array<short>>())), std::int64_t>);
    static_assert(std::is_same_v<decltype(hg::sum(std::declval<array<unsigned short>>())), std::uint64_t>);
    static_assert(std::is_same_v<decltype(hg::sum(std::declval<array<double>>())), double>);
    static_assert(std::is_same_v<decltype(hg::sum<double>(std::declval<A>())), double>);
    static_assert(std::is_same_v<decltype(hg::argmin(std::declval<A>())), hg::index_t>);
    static_assert(std::is_same_v<decltype(hg::minmax(std::declval<A>())), std::pair<int, int>>);
    static_assert(std::is_same_v<decltype(std::declval<A>() == std::declval<array<double>>()), bool>);

    TEST_CASE("array compile-time const-safety rules", "[array]") {
        // All checks are static_asserts in this file; reaching this point means they hold.
        SUCCEED();
    }
}
