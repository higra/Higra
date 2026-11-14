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



#include "higra/structure/array.hpp"
#include "higra/utils.hpp"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <map>
#include <iterator>
#include <random>
#include <initializer_list>

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_tostring.hpp"
#include <sstream>

/*
 * Catch2 printing of arrays, views and shapes (failure messages).
 */
namespace Catch {

    template<class T>
    struct StringMaker<hg::array<T>> {
        static std::string convert(const hg::array<T> &a) {
            std::ostringstream s;
            s << a;
            return s.str();
        }
    };

    template<class T>
    struct StringMaker<hg::array_view<T>> {
        static std::string convert(const hg::array_view<T> &a) {
            std::ostringstream s;
            s << hg::array_view<const std::remove_const_t<T>>(a);
            return s.str();
        }
    };

    template<>
    struct StringMaker<hg::array_shape> {
        static std::string convert(const hg::array_shape &s) {
            std::string r = "(";
            for (hg::index_t k = 0; k < s.size(); ++k) {
                r += (k > 0 ? ", " : "") + std::to_string(s[k]);
            }
            return r + ")";
        }
    };
}

/**
 * Test if two vectors are equal
 * @tparam T
 * @param v1
 * @param v2
 * @return
 */
template <typename T>
bool vectorEqual(std::vector<T> v1, std::vector<T> v2){
    if (v1.size() != v2.size())
        return false;
    return std::equal(v1.begin(), v1.end(), v2.begin());
}

/**
 * Test if two ranges are equal
 * @tparam T1
 * @tparam T2
 * @param r1
 * @param r2
 * @return
 */
template <typename T1, typename T2>
bool rangeEqual(const T1 & r1, const T2 & r2){
    auto b1 = r1.begin();
    auto b2 = r2.begin();
    auto e1 = r1.end();
    auto e2 = r2.end();
    for(;b1 != e1 && b2 != e2; b1++, b2++){
        if(*b1 != *b2){
            return false;
        }
    }
    if(b1 != e1 || b2 != e2){
        return false;
    }
    return true;
}

/**
 * Test if two containers contain the same elements
 * @tparam T
 * @param v1
 * @param v2
 * @return
 */
template<typename T1, typename T2>
bool vectorSame(const T1 & v1, const T2 & v2) {
    if (v1.size() != v2.size())
        return false;
    return std::is_permutation(v1.begin(), v1.end(), v2.begin());
}

/**
 * Test if there exists a bijective function f such that a[i] = f(b[i]) for all i.
 *
 * Returns false if a.size() != b.size(). Does not test if the shapes of a and b are the same.
 * @tparam T1
 * @tparam T2
 * @param a
 * @param b
 * @return
 */
template <typename T1, typename T2>
bool is_in_bijection(hg::array_view<const T1> a, hg::array_view<const T2> b){
    using vt1 = T1;
    using vt2 = T2;
    auto aa = a.flat();
    auto bb = b.flat();

    if(aa.size() != bb.size())
        return false;

    std::map<vt1, vt2> equiv1;
    std::map<vt2, vt1> equiv2;

    for(long i = 0; i < (long)aa.size(); i++){
        auto v1 = aa[i];
        auto v2 = bb[i];
        if (equiv1.count(v1) > 0){
            if(! (v2 == equiv1[v1]))
                return false;
        } else{
            equiv1[v1] = v2;
        }

        if (equiv2.count(v2) > 0){
            if(! (v1 == equiv2[v2]))
                return false;
        } else{
            equiv2[v2] = v1;
        }
    }
    return true;
};

/**
 * Show a message with type name at runtime
 * @tparam T
 * @param msg
 */
 /*
template<typename T>
void showTypeName(std::string msg = "") {
    std::cout << msg << boost::typeindex::type_id<T>().pretty_name() << std::endl;
}
*/

/**
 * Do not use except if you want a compile error showing the type of the provided template parameter !
 * @tparam T
 */
template<typename T>
struct COMPILE_ERROR;

template <typename T>
bool almost_equal(const T & a, const T& b, T epsilon){
    return std::abs(a - b) < epsilon;
}

inline
bool almost_equal(const double & a, const double& b){
    return almost_equal(a, b, 1e-9);
}

inline
bool almost_equal(const float & a, const float& b){
    return almost_equal(a, b, 1e-4f);
}

template <typename T>
void show_list(const T & l) {
    std::cout << "{";
    std::copy(l.cbegin(), l.cend(), std::ostream_iterator<typename T::value_type>(std::cout, ", "));
    std::cout << "}" << std::endl;
}

/*
 * Test helpers: random arrays, stack, transpose, sort and elementwise arithmetic.
 */
namespace hg_test {

    /**
     * Random engine shared by the helpers below; reseed with random_seed for reproducible tests.
     */
    inline std::mt19937_64 &random_engine() {
        static std::mt19937_64 engine(0);
        return engine;
    }

    inline void random_seed(std::uint64_t seed) {
        random_engine().seed(seed);
    }

    /**
     * Array of the given shape filled with uniform values in [0, 1).
     */
    template<typename T>
    hg::array<T> random_uniform(const hg::array_shape &shape) {
        std::uniform_real_distribution<T> d(T(0), T(1));
        hg::array<T> a(shape);
        for (auto &v: a) {
            v = d(random_engine());
        }
        return a;
    }

    /**
     * Array of the given shape filled with uniform integers in [low, high).
     */
    template<typename T>
    hg::array<T> random_randint(const hg::array_shape &shape, T low, T high) {
        std::uniform_int_distribution<long long> d((long long) low, (long long) high - 1);
        hg::array<T> a(shape);
        for (auto &v: a) {
            v = (T) d(random_engine());
        }
        return a;
    }

    /**
     * Random permutation of [0, n).
     */
    inline hg::array<hg::index_t> random_permutation(hg::index_t n) {
        auto a = hg::arange<hg::index_t>(n);
        std::shuffle(a.begin(), a.end(), random_engine());
        return a;
    }

    /**
     * Stack arrays of the same shape along a new first axis.
     */
    template<typename T>
    hg::array<T> stack(std::initializer_list<hg::array_view<const T>> arrays) {
        hg_assert(arrays.size() > 0, "stack: no array.");
        const auto &first = *arrays.begin();
        std::vector<hg::index_t> shape{(hg::index_t) arrays.size()};
        shape.insert(shape.end(), first.shape().begin(), first.shape().end());
        hg::array<T> res(hg::array_shape::from(shape));
        hg::index_t i = 0;
        for (const auto &a: arrays) {
            hg_assert(a.shape() == first.shape(), "stack: arrays must have the same shape.");
            std::copy(a.begin(), a.end(), res.begin() + i * first.size());
            i++;
        }
        return res;
    }

    template<typename T>
    hg::array<T> stack(const hg::array<T> &a, const hg::array<T> &b) {
        return stack<T>({a.view(), b.view()});
    }

    /**
     * Transpose of a 2d array.
     */
    template<typename T>
    hg::array<T> transpose(hg::array_view<const T> a) {
        hg_assert(a.dimension() == 2, "transpose: the array must be 2d.");
        hg::array<T> res(hg::array_shape(a.shape(1), a.shape(0)));
        for (hg::index_t i = 0; i < a.shape(0); i++) {
            for (hg::index_t j = 0; j < a.shape(1); j++) {
                res(j, i) = a(i, j);
            }
        }
        return res;
    }

    /**
     * Sorted copy of a 1d array.
     */
    template<typename T>
    hg::array<T> sort(hg::array_view<const T> a) {
        hg::array<T> res(a);
        std::sort(res.begin(), res.end());
        return res;
    }

    /**
     * Elementwise a / s (same promotion as the scalar expression, result converted to T).
     */
    template<typename T, typename S>
    hg::array<T> divided(hg::array_view<const T> a, S s) {
        return hg::map([s](const T &v) { return (T) (v / s); }, a);
    }

    /**
     * Elementwise a * s (same promotion as the scalar expression, result converted to T).
     */
    template<typename T, typename S>
    hg::array<T> multiplied(hg::array_view<const T> a, S s) {
        return hg::map([s](const T &v) { return (T) (v * s); }, a);
    }

    /**
     * Elementwise -a.
     */
    template<typename T>
    hg::array<T> negated(hg::array_view<const T> a) {
        return hg::map([](const T &v) { return (T) -v; }, a);
    }
}
