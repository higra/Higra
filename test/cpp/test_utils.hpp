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
#include "xtensor/generators/xgenerator.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/core/xmath.hpp"
#include "higra/utils.hpp"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <map>
#include <iterator>

#include "catch2/catch.hpp"

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
bool is_in_bijection(const xt::xexpression<T1> & a, const xt::xexpression<T2> & b){
    using vt1 = typename T1::value_type;
    using vt2 = typename T2::value_type;
    auto aa = xt::flatten(a.derived_cast());
    auto bb = xt::flatten(b.derived_cast());

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