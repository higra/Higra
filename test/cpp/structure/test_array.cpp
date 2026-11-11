/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "test_array_utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace test_array {

    using hg::array;
    using hg::array_shape;
    using hg::array_view;
    using hg::index_t;
    using array_test_utils::counting_allocator;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    template<class T>
    std::vector<T> values(array_view<const T> a) {
        return std::vector<T>(a.begin(), a.end());
    }

    TEST_CASE("array_shape construction", "[array][array_shape]") {
        array_shape s0;
        REQUIRE(s0.size() == 0);
        REQUIRE(s0.numel() == 1);
        REQUIRE(s0.begin() == s0.end());
        REQUIRE(array_shape{} == s0);

        REQUIRE(array_shape(4).size() == 1);
        REQUIRE(array_shape(4, 5).numel() == 20);
        REQUIRE(array_shape(1, 2, 3).numel() == 6);
        REQUIRE(array_shape(1, 2, 3, 4).numel() == 24);
        REQUIRE(array_shape(1, 2, 3, 4, 5).numel() == 120);
        REQUIRE(array_shape(1, 2, 3, 4, 5, 6).numel() == 720);
        REQUIRE(array_shape(1, 2, 3, 4, 5, 6, 7).numel() == 5040);
        array_shape s8(1, 2, 3, 4, 5, 6, 7, 8);
        REQUIRE(s8.size() == 8);
        REQUIRE(s8.numel() == 40320);
        for (index_t k = 0; k < 8; ++k) {
            REQUIRE(s8[k] == k + 1);
        }

        std::vector<index_t> nine(9, 1);
        REQUIRE_THROWS(array_shape::from(nine));
        std::vector<index_t> eight(8, 2);
        REQUIRE(array_shape::from(eight).numel() == 256);

        // unsigned and mixed integral types
        std::size_t n = 7;
        array_shape s1 = {n};
        REQUIRE(s1.size() == 1);
        REQUIRE(s1[0] == 7);
        array_shape mixed{n, 3, (short) 2, (unsigned char) 1, 4L, 5LL, (std::uint32_t) 6};
        REQUIRE(mixed.size() == 7);
        REQUIRE(mixed[2] == 2);
        REQUIRE(mixed[6] == 6);

        // invalid values
        REQUIRE_THROWS(array_shape(-1));
        REQUIRE_THROWS(array_shape(2, -3));
        REQUIRE_THROWS(array_shape(std::numeric_limits<std::uint64_t>::max()));
        REQUIRE(array_shape((std::uint64_t) std::numeric_limits<index_t>::max())[0] ==
                std::numeric_limits<index_t>::max());
        REQUIRE_THROWS(array_shape::from(std::vector<int>{2, -1}));

        // from ranges
        REQUIRE(array_shape::from(std::vector<int>{2, 3}) == array_shape(2, 3));
        REQUIRE(array_shape::from(std::vector<std::size_t>{2, 3}) == array_shape(2, 3));
        REQUIRE(array_shape::from(mixed) == mixed);
        REQUIRE(array_shape::from(std::vector<int>{}) == array_shape());

        // accessors
        array_shape s(2, 3, 4);
        REQUIRE(std::vector<index_t>(s.begin(), s.end()) == std::vector<index_t>{2, 3, 4});
        REQUIRE(s.data()[1] == 3);
        s[1] = 5;
        REQUIRE(s == array_shape(2, 5, 4));
        REQUIRE(s != array_shape(2, 5));
        REQUIRE(s != array_shape(2, 5, 3));
        REQUIRE(s.tail() == array_shape(5, 4));
        REQUIRE(array_shape(3).tail() == array_shape());
        REQUIRE(array_shape().tail() == array_shape());
        REQUIRE(array_shape(3, 0).numel() == 0);
    }

    TEST_CASE("array construction", "[array]") {
        array<int> a;
        REQUIRE(a.dimension() == 1);
        REQUIRE(a.size() == 0);
        REQUIRE(a.empty());
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.begin() == a.end());

        array<double> b(array_shape(2, 3));
        REQUIRE(b.shape() == array_shape(2, 3));
        REQUIRE(b.shape(0) == 2);
        REQUIRE(b.shape(1) == 3);
        REQUIRE(b.dimension() == 2);
        REQUIRE(b.size() == 6);
        REQUIRE(!b.empty());
        REQUIRE(b.device() == hg::cpu_device);
        REQUIRE(reinterpret_cast<std::uintptr_t>(b.data()) % 64 == 0);

        auto c = array<int>::from_shape({4});
        REQUIRE(c.shape() == array_shape(4));

        array<int> d(array_shape(3, 2), 7);
        REQUIRE(values<int>(d) == std::vector<int>(6, 7));

        array<int> e(2); // shape (2,): an integer converts to a 1d shape
        REQUIRE(e.shape() == array_shape(2));

        // rank 0
        array<int> s(array_shape{});
        REQUIRE(s.dimension() == 0);
        REQUIRE(s.size() == 1);
        s() = 3;
        REQUIRE(s() == 3);
        REQUIRE(s[0] == 3);

        // ranks up to max_rank
        array<char> r8(array_shape(1, 2, 1, 2, 1, 2, 1, 2), 'x');
        REQUIRE(r8.dimension() == hg::max_rank);
        REQUIRE(r8.size() == 16);
        REQUIRE(r8(0, 1, 0, 1, 0, 1, 0, 1) == 'x');

        // from a view
        array<int> v(d.view());
        REQUIRE(v == d);
        REQUIRE(v.data() != d.data());
    }

    TEST_CASE("array zero-size arrays allocate nothing", "[array]") {
        counting_allocator alloc;
        {
            array<int> a(array_shape(0), alloc);
            array<int> b(array_shape(3, 0), alloc);
            array<int> c(array_shape(0, 5), 1, alloc);
            REQUIRE(alloc.allocations == 0);
            REQUIRE(a.size() == 0);
            REQUIRE(b.size() == 0);
            REQUIRE(c.size() == 0);
            REQUIRE(a.data() == nullptr);
            REQUIRE(b.shape() == array_shape(3, 0));
            REQUIRE(b.row_size() == 0);
            REQUIRE(c.row_size() == 5);
            array<int> d(b);
            array<int> e = std::move(c);
            REQUIRE(d.shape() == array_shape(3, 0));
            REQUIRE(e.shape() == array_shape(0, 5));
            REQUIRE(alloc.allocations == 0);
            REQUIRE(hg::sum(b) == 0);
            REQUIRE(b == d);
            auto f = hg::zeros<double>({0});
            REQUIRE(f.size() == 0);
            auto r = std::move(b).release();
            REQUIRE(r.data == nullptr);
            REQUIRE(r.bytes == 0);
        }
        REQUIRE(alloc.deallocations == 0);
    }

    TEST_CASE("array literals", "[array]") {
        array<int> a{5};
        REQUIRE(a.shape() == array_shape(1));
        REQUIRE(a(0) == 5);

        array<int> b = {1, 2, 3};
        REQUIRE(b.shape() == array_shape(3));
        REQUIRE(values<int>(b) == std::vector<int>{1, 2, 3});

        array<double> c{{1, 2.5, 3}, {4, 5, 6}};
        REQUIRE(c.shape() == array_shape(2, 3));
        REQUIRE(c(0, 1) == 2.5);
        REQUIRE(c(1, 2) == 6);

        array<int> d{{{1, 2}, {3, 4}, {5, 6}}};
        REQUIRE(d.shape() == array_shape(1, 3, 2));
        REQUIRE(d(0, 2, 1) == 6);

        array<int> e{{{{1, 2}}, {{3, 4}}}, {{{5, 6}}, {{7, 8}}}};
        REQUIRE(e.shape() == array_shape(2, 2, 1, 2));
        REQUIRE(e(1, 1, 0, 0) == 7);
        REQUIRE(values<int>(e) == std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8});

        // 2d literals take precedence over 1d ones (single-element rows)
        array<int> f{{1}};
        REQUIRE(f.shape() == array_shape(1, 1));
        array<int> g{{1}, {2}, {3}};
        REQUIRE(g.shape() == array_shape(3, 1));
        array<int> h{{1, 2}, {3, 4}};
        REQUIRE(h.shape() == array_shape(2, 2));
        // Known limitation: between 2d, 3d and 4d, a literal whose innermost lists all have one
        // element or are empty, such as {{{1}, {2}}} or {{}, {}}, is ambiguous and does not compile.

        array<int> empty_literal{};
        REQUIRE(empty_literal.shape() == array_shape(0));

        // ragged literals throw
        REQUIRE_THROWS(array<int>{{1, 2}, {3}});
        REQUIRE_THROWS(array<int>{{{1, 2}, {3, 4}}, {{1, 2}, {3}}});
        REQUIRE_THROWS(array<int>{{{{1, 2}}}, {{{3}}}});
    }

    TEST_CASE("array copy, move and assignment", "[array]") {
        counting_allocator alloc;
        {
            array<int> a(array_shape(2, 2), 1, alloc);
            REQUIRE(alloc.allocations == 1);
            REQUIRE(&a.get_allocator() == &alloc);

            // deep copy, same allocator
            array<int> b(a);
            REQUIRE(alloc.allocations == 2);
            REQUIRE(&b.get_allocator() == &alloc);
            REQUIRE(b.data() != a.data());
            b(0, 0) = 5;
            REQUIRE(a(0, 0) == 1);

            // move steals the buffer
            const int *pb = b.data();
            array<int> c(std::move(b));
            REQUIRE(c.data() == pb);
            REQUIRE(c(0, 0) == 5);
            REQUIRE(b.size() == 0);
            REQUIRE(b.dimension() == 1);
            REQUIRE(b.data() == nullptr);
            REQUIRE(alloc.allocations == 2);

            // copy assignment: same size and allocator reuses the buffer
            array<int> d(array_shape(4), 0, alloc);
            const int *pd = d.data();
            d = c;
            REQUIRE(d.data() == pd);
            REQUIRE(d.shape() == array_shape(2, 2));
            REQUIRE(d == c);

            // copy assignment from another allocator adopts it
            array<int> e(array_shape(3), 9);
            d = e;
            REQUIRE(&d.get_allocator() == &hg::cpu_allocator());
            REQUIRE(d == e);
            e = c;
            REQUIRE(&e.get_allocator() == &alloc);
            REQUIRE(e == c);

            // self assignment
            auto &ee = e;
            e = ee;
            REQUIRE(e == c);

            // move assignment
            const int *pc = c.data();
            array<int> f;
            f = std::move(c);
            REQUIRE(f.data() == pc);
            REQUIRE(c.empty());
            REQUIRE(&f.get_allocator() == &alloc);

            // swap
            array<int> g{1, 2, 3};
            swap(f, g);
            REQUIRE(values<int>(f) == std::vector<int>{1, 2, 3});
            REQUIRE(g.data() == pc);
        }
        REQUIRE(alloc.allocations == alloc.deallocations);
        REQUIRE(alloc.live_bytes == 0);
    }

    TEST_CASE("array resize, reshape and release", "[array]") {
        array<int> a{{1, 2, 3}, {4, 5, 6}};
        const int *p = a.data();
        a.reshape({3, 2});
        REQUIRE(a.shape() == array_shape(3, 2));
        REQUIRE(a.data() == p);
        REQUIRE(a(2, 1) == 6);
        REQUIRE_THROWS(a.reshape({4}));

        a.resize({6});
        REQUIRE(a.data() == p);
        REQUIRE(a(5) == 6);
        a.resize({2, 5});
        REQUIRE(a.shape() == array_shape(2, 5));
        REQUIRE(a.size() == 10);

        counting_allocator alloc;
        array<double> b(array_shape(4), 2.0, alloc);
        auto r = std::move(b).release();
        REQUIRE(b.empty());
        REQUIRE(r.bytes == 4 * sizeof(double));
        REQUIRE(r.alloc == &alloc);
        REQUIRE(r.shape == array_shape(4));
        REQUIRE(static_cast<double *>(r.data)[3] == 2.0);
        r.alloc->deallocate(r.data, r.bytes);
        REQUIRE(alloc.live_bytes == 0);
    }

    TEST_CASE("array accessors", "[array]") {
        array<int> a{{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                     {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}};
        REQUIRE(a.shape() == array_shape(2, 3, 4));
        REQUIRE(a(1, 2, 3) == 23);
        REQUIRE(a((short) 1, 2u, 3L) == 23);
        REQUIRE(a(std::size_t(1), 0, 2) == 14);
        REQUIRE(a[13] == 13);
        REQUIRE(a.row_size() == 12);
        REQUIRE(*(a.end() - 1) == 23);

        // row
        auto r1 = a.row(1);
        REQUIRE(r1.shape() == array_shape(3, 4));
        REQUIRE(r1(0, 0) == 12);
        auto r12 = r1.row(2);
        REQUIRE(r12.shape() == array_shape(4));
        REQUIRE(values<int>(r12) == std::vector<int>{20, 21, 22, 23});
        auto scalar = r12.row(3);
        REQUIRE(scalar.dimension() == 0);
        REQUIRE(scalar() == 23);
        a.row(0).row(1)(2) = -6;
        REQUIRE(a(0, 1, 2) == -6);

        // slice
        auto s = a.slice(0, 1);
        REQUIRE(s.shape() == array_shape(1, 3, 4));
        REQUIRE(s.data() == a.data());
        auto s2 = a.row(1).slice(1, 3);
        REQUIRE(s2.shape() == array_shape(2, 4));
        REQUIRE(s2(0, 0) == 16);
        REQUIRE(a.slice(2, 2).size() == 0);
        REQUIRE_THROWS(a.slice(1, 3));
        REQUIRE_THROWS(a.slice(2, 1));
        REQUIRE_THROWS(a.slice(-1, 1));
        a.slice(1, 2).fill(0);
        REQUIRE(hg::sum(a.row(1)) == 0);

        // reshaped / flat / view
        auto rs = a.reshaped({6, 4});
        REQUIRE(rs.shape() == array_shape(6, 4));
        REQUIRE(rs(1, 0) == 4);
        REQUIRE_THROWS(a.reshaped({5, 5}));
        auto fl = a.flat();
        REQUIRE(fl.shape() == array_shape(24));
        REQUIRE(fl(5) == 5);
        fl(5) = 50;
        REQUIRE(a(0, 1, 1) == 50);
        auto v = a.view();
        REQUIRE(v.data() == a.data());
        REQUIRE(v.shape() == a.shape());

        // row_size
        REQUIRE(array<int>{1, 2, 3}.row_size() == 1);
        REQUIRE(array<int>(array_shape()).row_size() == 1);
        REQUIRE(array<int>(array_shape(0, 5)).row_size() == 5);

        // const accessors
        const array<int> &ca = a;
        REQUIRE(ca(1, 2, 3) == 0);
        REQUIRE(ca.row(0).row(0)(1) == 1);
        REQUIRE(ca.flat()[5] == 50);

        // views over existing storage
        std::vector<double> storage{1, 2, 3, 4, 5, 6};
        array_view<double> mv(storage.data(), {3, 2});
        mv(2, 1) = 60;
        REQUIRE(storage[5] == 60);
        array_view<const double> cv(storage.data(), {2, 3});
        REQUIRE(cv(1, 2) == 60);
        array_view<int> empty_view;
        REQUIRE(empty_view.shape() == array_shape(0));
        REQUIRE(empty_view.data() == nullptr);

        // fill
        mv.fill(1.5);
        REQUIRE(storage == std::vector<double>(6, 1.5));
    }

#ifdef HG_ARRAY_CHECKS
    TEST_CASE("array checked element access", "[array]") {
        array<int> a(array_shape(2, 3), 0);
        REQUIRE_NOTHROW(a(1, 2));
        REQUIRE_THROWS(a(1));
        REQUIRE_THROWS(a(1, 2, 0));
        REQUIRE_THROWS(a(2, 0));
        REQUIRE_THROWS(a(0, 3));
        REQUIRE_THROWS(a(-1, 0));
        REQUIRE_THROWS(a[6]);
        REQUIRE_THROWS(a[-1]);
        REQUIRE_THROWS(a.row(2));
        REQUIRE_THROWS(a.row(0)(3));
        const array<int> &ca = a;
        REQUIRE_THROWS(ca(0, 0, 0));
        REQUIRE_THROWS(ca.row(-1));
        REQUIRE_THROWS(a.shape()[2]);
    }
#endif

    TEST_CASE("array creation functions", "[array][array_functions]") {
        auto e = hg::empty<float>({3, 2});
        REQUIRE(e.shape() == array_shape(3, 2));
        REQUIRE(values<int>(hg::zeros<int>({4})) == std::vector<int>(4, 0));
        REQUIRE(values<double>(hg::ones<double>({2, 2})) == std::vector<double>(4, 1));
        REQUIRE(values<char>(hg::full<char>({3}, 'a')) == std::vector<char>(3, 'a'));
        REQUIRE(hg::full<double>({2}, 1).shape() == array_shape(2));
        std::size_t n = 3;
        REQUIRE(hg::zeros<int>({n, 2}).shape() == array_shape(3, 2));

        REQUIRE(values<int>(hg::arange<int>(5)) == std::vector<int>{0, 1, 2, 3, 4});
        REQUIRE(values<index_t>(hg::arange<index_t>(std::size_t(3))) == std::vector<index_t>{0, 1, 2});
        REQUIRE(values<int>(hg::arange<int>(2, 8, 2)) == std::vector<int>{2, 4, 6});
        REQUIRE(values<int>(hg::arange<int>(2, 9, 2)) == std::vector<int>{2, 4, 6, 8});
        REQUIRE(values<int>(hg::arange<int>(5, 0, -2)) == std::vector<int>{5, 3, 1});
        REQUIRE(hg::arange<int>(0).size() == 0);
        REQUIRE(hg::arange<int>(5, 0).size() == 0);
        REQUIRE(hg::arange<unsigned>(5, 0).size() == 0);
        REQUIRE(values<unsigned>(hg::arange<unsigned>(1, 4)) == std::vector<unsigned>{1, 2, 3});
        REQUIRE_THROWS(hg::arange<int>(0, 5, 0));
        REQUIRE(values<double>(hg::arange<double>(0, 1, 0.25)) == std::vector<double>{0, 0.25, 0.5, 0.75});
        REQUIRE(hg::arange<double>(0, 1, 0.3).size() == 4);
        REQUIRE(hg::arange<double>(1, 0, -0.5).size() == 2);

        counting_allocator alloc;
        {
            array<double> a(array_shape(2, 3), 1.5, alloc);
            auto el = hg::empty_like(a);
            auto zl = hg::zeros_like(a);
            auto ol = hg::ones_like(a);
            auto fl = hg::full_like(a, 4);
            REQUIRE(alloc.allocations == 5);
            for (auto *x: {&el, &zl, &ol, &fl}) {
                REQUIRE(x->shape() == a.shape());
                REQUIRE(&x->get_allocator() == &alloc);
            }
            REQUIRE(values<double>(zl) == std::vector<double>(6, 0));
            REQUIRE(values<double>(ol) == std::vector<double>(6, 1));
            REQUIRE(values<double>(fl) == std::vector<double>(6, 4));
            static_assert(std::is_same_v<decltype(fl), array<double>>);

            // from a view: CPU allocator
            auto zv = hg::zeros_like(a.row(0));
            REQUIRE(zv.shape() == array_shape(3));
            REQUIRE(&zv.get_allocator() == &hg::cpu_allocator());
            REQUIRE(hg::full_like(a.view(), 2).shape() == a.shape());
        }
    }

    TEST_CASE("array assign and cast", "[array][array_functions]") {
        array<double> a{{1.5, -2.7}, {3.2, 4}};
        array<int> b(array_shape(2, 2));
        hg::assign(b, a);
        REQUIRE(b == array<int>{{1, -2}, {3, 4}});
        array<double> c(array_shape(2, 2));
        hg::assign(c, a);
        REQUIRE(c == a);
        hg::assign(c, c); // identical source and destination
        REQUIRE(c == a);
        hg::assign(b.row(1), array<double>{9, 8});
        REQUIRE(b == array<int>{{1, -2}, {9, 8}});
        REQUIRE_THROWS(hg::assign(b, array<int>{1, 2, 3, 4}));

        auto d = hg::cast<int>(a);
        static_assert(std::is_same_v<decltype(d), array<int>>);
        REQUIRE(d == array<int>{{1, -2}, {3, 4}});
        auto e = hg::cast<bool>(array<int>{0, 2, -1});
        REQUIRE(values<bool>(e) == std::vector<bool>{false, true, true});
        auto f = hg::cast<std::uint8_t>(array<int>{0, 200, 255});
        REQUIRE(f == array<int>{0, 200, 255});
        REQUIRE(hg::cast<double>(a.row(0)) == array<double>{1.5, -2.7});
    }

    TEST_CASE("array map", "[array][array_functions]") {
        array<int> a{{1, 2}, {3, 4}};
        array<double> b{{0.5, 0.5}, {1, 1}};

        auto r1 = hg::map([](int x) { return x * 2; }, a);
        REQUIRE(r1 == array<int>{{2, 4}, {6, 8}});

        auto r2 = hg::map([](int x, double y) { return x + y; }, a, b);
        static_assert(std::is_same_v<decltype(r2), array<double>>);
        REQUIRE(r2 == array<double>{{1.5, 2.5}, {4, 5}});

        auto r3 = hg::map([](int x, double y, int z) { return x * y > z; }, a, b, array<int>{{0, 1}, {3, 5}});
        static_assert(std::is_same_v<decltype(r3), array<bool>>);
        REQUIRE(r3 == array<bool>{{true, false}, {false, false}});

        auto r4 = hg::map(hg::exec::par, [](int x) { return (float) x; }, a.row(1));
        static_assert(std::is_same_v<decltype(r4), array<float>>);
        REQUIRE(r4 == array<float>{3, 4});
        REQUIRE(hg::map(hg::exec::seq, [](int x) { return -x; }, a) == array<int>{{-1, -2}, {-3, -4}});

        array<long> out(array_shape(2, 2));
        hg::map_into(out, [](int x, double y) { return (long) (x * 10 + y); }, a, b);
        REQUIRE(out == array<long>{{10, 20}, {31, 41}});
        hg::map_into(hg::exec::par, out.row(0), [](int x) { return x; }, a.row(1));
        REQUIRE(out == array<long>{{3, 4}, {31, 41}});
        hg::map_into(hg::exec::seq, out.row(0), [](int x) { return x; }, a.row(0));
        REQUIRE(out == array<long>{{1, 2}, {31, 41}});

        // mutable state is fine with the default sequential policy
        int calls = 0;
        hg::map([&calls](int x) { ++calls; return x; }, a);
        REQUIRE(calls == 4);

        REQUIRE_THROWS(hg::map([](int x, int y) { return x + y; }, a, array<int>{1, 2, 3, 4}));
        REQUIRE_THROWS(hg::map_into(out, [](int x) { return x; }, array<int>{1, 2}));
    }

    TEST_CASE("array reductions", "[array][array_functions]") {
        array<int> a{{3, -1, 4}, {1, -5, 9}};
        REQUIRE(hg::min(a) == -5);
        REQUIRE(hg::max(a) == 9);
        REQUIRE(hg::minmax(a) == std::pair<int, int>(-5, 9));
        REQUIRE(hg::argmin(a) == 4);
        REQUIRE(hg::argmax(a) == 5);
        auto s = hg::sum(a);
        static_assert(std::is_same_v<decltype(s), std::int64_t>);
        REQUIRE(s == 11);
        REQUIRE(hg::sum<double>(a) == 11.0);
        REQUIRE(hg::count_nonzero(array<int>{0, 1, 0, 2}) == 2);
        REQUIRE(hg::all(array<int>{1, 2, 3}));
        REQUIRE(!hg::all(array<int>{1, 0, 3}));
        REQUIRE(hg::any(array<int>{0, 0, 3}));
        REQUIRE(!hg::any(array<int>{0, 0, 0}));

        // promoted sum types
        array<std::uint8_t> u(array_shape(300), 200);
        auto su = hg::sum(u);
        static_assert(std::is_same_v<decltype(su), std::uint64_t>);
        REQUIRE(su == 60000);
        array<bool> bo{true, false, true};
        static_assert(std::is_same_v<decltype(hg::sum(bo)), std::int64_t>);
        REQUIRE(hg::sum(bo) == 2);
        REQUIRE(hg::count_nonzero(bo) == 2);
        REQUIRE(hg::min(bo) == false);
        REQUIRE(hg::argmax(bo) == 0);
        array<float> fl{0.5f, 0.25f};
        static_assert(std::is_same_v<decltype(hg::sum(fl)), float>);
        REQUIRE(hg::sum(fl) == 0.75f);

        // ties return the first index
        array<int> t{2, 0, 5, 0, 5};
        REQUIRE(hg::argmin(t) == 1);
        REQUIRE(hg::argmax(t) == 2);

        // empty inputs
        array<double> e;
        REQUIRE_THROWS(hg::min(e));
        REQUIRE_THROWS(hg::max(e));
        REQUIRE_THROWS(hg::minmax(e));
        REQUIRE_THROWS(hg::argmin(e));
        REQUIRE_THROWS(hg::argmax(e));
        REQUIRE(hg::sum(e) == 0);
        REQUIRE(hg::count_nonzero(e) == 0);
        REQUIRE(hg::all(e));
        REQUIRE(!hg::any(e));

        // rank 0
        array<int> r0(array_shape(), 7);
        REQUIRE(hg::min(r0) == 7);
        REQUIRE(hg::sum(r0) == 7);
    }

    TEST_CASE("array reductions with NaN follow NumPy", "[array][array_functions]") {
        array<double> a{1, nan, -3, nan};
        REQUIRE(std::isnan(hg::min(a)));
        REQUIRE(std::isnan(hg::max(a)));
        auto mm = hg::minmax(a);
        REQUIRE(std::isnan(mm.first));
        REQUIRE(std::isnan(mm.second));
        REQUIRE(hg::argmin(a) == 1);
        REQUIRE(hg::argmax(a) == 1);
        REQUIRE(std::isnan(hg::sum(a)));
        REQUIRE(hg::count_nonzero(a) == 4);
        REQUIRE(hg::all(a));
        REQUIRE(hg::argmin(array<double>{nan, 1}) == 0);
        REQUIRE(std::isnan(hg::max(array<double>{nan, 1})));
        REQUIRE(hg::min(array<double>{inf, -inf, 0}) == -inf);

        // across reduction chunks
        array<double> big(array_shape(3 * HG_ARRAY_REDUCE_CHUNK + 7), 1.0);
        big(2 * HG_ARRAY_REDUCE_CHUNK + 1) = nan;
        big(3 * HG_ARRAY_REDUCE_CHUNK + 2) = -1;
        REQUIRE(hg::argmin(big) == 2 * HG_ARRAY_REDUCE_CHUNK + 1);
        REQUIRE(std::isnan(hg::min(big)));
        REQUIRE(std::isnan(hg::minmax(big).second));
    }

    TEST_CASE("array reductions across chunks", "[array][array_functions]") {
        const index_t n = 5 * HG_ARRAY_REDUCE_CHUNK + 3;
        array<int> a(array_shape(n), 0);
        a(3 * HG_ARRAY_REDUCE_CHUNK + 5) = -2;
        a(4 * HG_ARRAY_REDUCE_CHUNK + 1) = -2;
        a(HG_ARRAY_REDUCE_CHUNK - 1) = 8;
        a(n - 1) = 8;
        REQUIRE(hg::argmin(a) == 3 * HG_ARRAY_REDUCE_CHUNK + 5);
        REQUIRE(hg::argmax(a) == HG_ARRAY_REDUCE_CHUNK - 1);
        REQUIRE(hg::minmax(a) == std::pair<int, int>(-2, 8));
        REQUIRE(hg::sum(a) == 12);
        REQUIRE(hg::count_nonzero(a) == 4);
        REQUIRE(!hg::all(a));
        REQUIRE(hg::any(a));
    }

    TEST_CASE("array take and put", "[array][array_functions]") {
        array<int> a{{0, 1}, {2, 3}, {4, 5}};
        auto t = hg::take(a, array<index_t>{2, 0, 2});
        REQUIRE(t == array<int>{{4, 5}, {0, 1}, {4, 5}});
        auto t1 = hg::take(array<double>{0.5, 1.5, 2.5}, array<std::uint8_t>{1, 1});
        REQUIRE(t1 == array<double>{1.5, 1.5});
        auto t0 = hg::take(a, array<int>());
        REQUIRE(t0.shape() == array_shape(0, 2));
        auto tz = hg::take(array<int>(array_shape(3, 0)), array<int>{2, 0});
        REQUIRE(tz.shape() == array_shape(2, 0));
        REQUIRE_THROWS(hg::take(array<int>(array_shape(3, 0)), array<int>{3}));
        REQUIRE_THROWS(hg::take(a, array<int>{3}));
        REQUIRE_THROWS(hg::take(a, array<int>{-1}));
        REQUIRE_THROWS(hg::take(a, array<int>{{0}}));
        REQUIRE_THROWS(hg::take(array<int>(array_shape()), array<int>{0}));

        hg::put(a, array<int>{1}, 9);
        REQUIRE(a == array<int>{{0, 1}, {9, 9}, {4, 5}});
        hg::put(a, array<std::size_t>{0, 2, 0}, array<int>{{1, 2}, {3, 4}, {5, 6}});
        REQUIRE(a == array<int>{{5, 6}, {9, 9}, {3, 4}}); // last write wins
        array<double> d{1, 2, 3};
        hg::put(d, array<int>{0, 2}, 0); // integer value converted to the element type
        REQUIRE(d == array<double>{0, 2, 0});
        REQUIRE_THROWS(hg::put(a, array<int>{3}, 0));
        REQUIRE_THROWS(hg::put(a, array<int>{0}, array<int>{1, 2}));
        REQUIRE_THROWS(hg::put(a, array<int>{0, 1}, array<int>{{1, 2}}));
    }

    TEST_CASE("array comparison", "[array][array_functions]") {
        array<int> a{{1, 2}, {3, 4}};
        REQUIRE(a == array<int>{{1, 2}, {3, 4}});
        REQUIRE(a != array<int>{{1, 2}, {3, 5}});
        REQUIRE(a != array<int>{1, 2, 3, 4});
        REQUIRE(a == array<double>{{1, 2}, {3, 4}});
        REQUIRE(a != array<double>{{1, 2}, {3, 4.5}});
        REQUIRE(array<int>{-1} != array<unsigned>{std::numeric_limits<unsigned>::max()});
        REQUIRE(array<unsigned>{std::numeric_limits<unsigned>::max()} != array<int>{-1});
        REQUIRE(array<std::uint64_t>{5} == array<std::int8_t>{5});
        REQUIRE(array<bool>{true, false} == array<int>{1, 0});
        REQUIRE(array<double>{nan} != array<double>{nan});
        REQUIRE(a.row(1) == array<long>{3, 4});
        REQUIRE(array<int>() == array<double>());
        REQUIRE(array<int>(array_shape(2, 0)) != array<int>(array_shape(0, 2)));

        REQUIRE(hg::same_shape(a, array<double>(array_shape(2, 2))));
        REQUIRE(!hg::same_shape(a, a.flat()));
        REQUIRE(hg::same_shape(a.shape(), array_shape(2, 2)));

        array<double> x{1.0, 2.0, 1e10};
        REQUIRE(hg::allclose(x, array<double>{1.0 + 1e-9, 2.0, 1e10 + 1}));
        REQUIRE(!hg::allclose(x, array<double>{1.001, 2.0, 1e10}));
        REQUIRE(hg::allclose(x, array<double>{1.001, 2.0, 1e10}, 1e-2));
        REQUIRE(hg::allclose(array<double>{0}, array<double>{1e-9}));
        REQUIRE(!hg::allclose(array<double>{0}, array<double>{1e-9}, 0, 0));
        REQUIRE(hg::allclose(a, array<float>{{1, 2}, {3, 4}}));
        REQUIRE(!hg::allclose(array<double>{nan}, array<double>{nan}));
        REQUIRE(!hg::allclose(array<double>{1, nan}, array<double>{1, 2}));
        REQUIRE(hg::allclose(array<double>{inf, -inf}, array<double>{inf, -inf}));
        REQUIRE(!hg::allclose(array<double>{inf}, array<double>{-inf}));
        REQUIRE(!hg::allclose(array<double>{inf}, array<double>{1e308}));
        REQUIRE(!hg::allclose(a, a.flat()));
        // tolerance is relative to the second argument, as in NumPy
        REQUIRE(hg::allclose(array<double>{100}, array<double>{111}, 0.1, 0));
        REQUIRE(!hg::allclose(array<double>{111}, array<double>{100}, 0.1, 0));
    }

    TEST_CASE("array dtype mapping", "[array]") {
        using hg::dtype;
        using hg::dtype_code;
        using hg::dtype_of;
        static_assert(dtype_of<bool>() == dtype{dtype_code::bool_, 8});
        static_assert(dtype_of<std::int8_t>() == dtype{dtype_code::int_, 8});
        static_assert(dtype_of<std::int16_t>() == dtype{dtype_code::int_, 16});
        static_assert(dtype_of<std::int32_t>() == dtype{dtype_code::int_, 32});
        static_assert(dtype_of<std::int64_t>() == dtype{dtype_code::int_, 64});
        static_assert(dtype_of<std::uint8_t>() == dtype{dtype_code::uint, 8});
        static_assert(dtype_of<std::uint16_t>() == dtype{dtype_code::uint, 16});
        static_assert(dtype_of<std::uint32_t>() == dtype{dtype_code::uint, 32});
        static_assert(dtype_of<std::uint64_t>() == dtype{dtype_code::uint, 64});
        static_assert(dtype_of<float>() == dtype{dtype_code::float_, 32});
        static_assert(dtype_of<double>() == dtype{dtype_code::float_, 64});
        static_assert(dtype_of<const double>() == dtype_of<double>());
        static_assert(dtype_of<signed char>() == dtype{dtype_code::int_, 8});
        static_assert(dtype_of<unsigned char>() == dtype{dtype_code::uint, 8});
        static_assert(dtype_of<short>() == dtype{dtype_code::int_, 8 * sizeof(short)});
        static_assert(dtype_of<int>() == dtype{dtype_code::int_, 8 * sizeof(int)});
        static_assert(dtype_of<unsigned>() == dtype{dtype_code::uint, 8 * sizeof(unsigned)});
        static_assert(dtype_of<long>() == dtype{dtype_code::int_, 8 * sizeof(long)});
        static_assert(dtype_of<unsigned long>() == dtype{dtype_code::uint, 8 * sizeof(long)});
        static_assert(dtype_of<long long>() == dtype{dtype_code::int_, 64});
        static_assert(dtype_of<unsigned long long>() == dtype{dtype_code::uint, 64});
        static_assert(dtype_of<std::size_t>() == dtype{dtype_code::uint, 8 * sizeof(std::size_t)});
        static_assert(dtype_of<std::ptrdiff_t>() == dtype{dtype_code::int_, 8 * sizeof(std::ptrdiff_t)});
        static_assert(dtype_of<char>() ==
                      dtype{std::is_signed_v<char> ? dtype_code::int_ : dtype_code::uint, 8});
        static_assert(dtype_of<hg::index_t>() == dtype{dtype_code::int_, 64});
        static_assert(dtype_of<long>() == dtype_of<std::conditional_t<sizeof(long) == 8, std::int64_t, std::int32_t>>());
        static_assert(dtype_of<float>() != dtype_of<std::int32_t>());
        // DLPack codes
        static_assert(static_cast<int>(dtype_code::int_) == 0 && static_cast<int>(dtype_code::uint) == 1 &&
                      static_cast<int>(dtype_code::float_) == 2 && static_cast<int>(dtype_code::bool_) == 6);
        static_assert(static_cast<int>(hg::device_type::cpu) == 1 && static_cast<int>(hg::device_type::cuda) == 2 &&
                      static_cast<int>(hg::device_type::cuda_host) == 3 &&
                      static_cast<int>(hg::device_type::rocm) == 10 &&
                      static_cast<int>(hg::device_type::cuda_managed) == 13);
        SUCCEED();
    }

    TEST_CASE("array device checks", "[array]") {
        REQUIRE(hg::cpu_device == hg::device{});
        REQUIRE(hg::device{hg::device_type::cuda, 1} != hg::device{hg::device_type::cuda, 0});
        REQUIRE(hg::is_cpu_accessible(hg::device{hg::device_type::cuda_host, 0}));
        REQUIRE(hg::is_cpu_accessible(hg::device{hg::device_type::cuda_managed, 0}));
        REQUIRE(!hg::is_cpu_accessible(hg::device{hg::device_type::cuda, 0}));
        REQUIRE(!hg::is_cpu_accessible(hg::device{hg::device_type::rocm, 0}));

        std::vector<int> storage{1, 2, 3};
        array_view<int> gpu(storage.data(), {3}, hg::device{hg::device_type::cuda, 0});
        REQUIRE(gpu.device().type == hg::device_type::cuda);
        REQUIRE_THROWS(hg::sum(gpu));
        REQUIRE_THROWS(hg::min(gpu));
        REQUIRE_THROWS(gpu.fill(0));
        REQUIRE_THROWS(array<int>(gpu));
        REQUIRE_THROWS(hg::cast<double>(gpu));
        REQUIRE_THROWS(hg::take(gpu, array<int>{0}));
        array<int> dst(array_shape(3));
        REQUIRE_THROWS(hg::assign(dst, gpu));
        array_view<int> managed(storage.data(), {3}, hg::device{hg::device_type::cuda_managed, 0});
        REQUIRE(hg::sum(managed) == 6);
    }

    TEST_CASE("array printing", "[array]") {
        auto str = [](const auto &a) {
            std::ostringstream s;
            s << a;
            return s.str();
        };
        REQUIRE(str(array<int>(array_shape(), 3)) == "3");
        REQUIRE(str(array<int>{1, 2, 3}) == "[1, 2, 3]");
        REQUIRE(str(array<int>{{1, 20}, {300, 4}}) == "[[  1,  20],\n [300,   4]]");
        REQUIRE(str(array<int>{{{1, 2}}, {{3, 4}}}) == "[[[1, 2]],\n\n [[3, 4]]]");
        REQUIRE(str(array<bool>{true, false}) == "[ true, false]");
        REQUIRE(str(array<std::int8_t>{-1, 2}) == "[-1,  2]");
        REQUIRE(str(array<double>{0.5, 1.25}) == "[ 0.5, 1.25]");
        REQUIRE(str(array<int>()) == "[] (shape (0,))");
        REQUIRE(str(array<int>(array_shape(2, 0))) == "[[]] (shape (2, 0))");
        REQUIRE(str(array<int>{{1, 2}}.row(0)) == "[1, 2]");

        auto small = hg::arange<int>(1000);
        REQUIRE(str(small).find("...") == std::string::npos);
        auto big = hg::arange<int>(1001);
        REQUIRE(str(big) == "[   0,    1,    2, ...,  998,  999, 1000]");
        auto big2 = hg::zeros<int>({100, 100});
        auto s2 = str(big2);
        REQUIRE(s2.find("...") != std::string::npos);
        REQUIRE(std::count(s2.begin(), s2.end(), '\n') == 6);

        REQUIRE(Catch::StringMaker<array<int>>::convert(array<int>{1, 2}) == "[1, 2]");
        REQUIRE(Catch::StringMaker<array_view<const int>>::convert(array<int>{3}) == "[3]");
        REQUIRE(Catch::StringMaker<array_shape>::convert(array_shape(2, 3)) == "(2, 3)");
    }
}
