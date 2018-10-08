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

#include <stdio.h>
#include <exception>
#include <string>
#include <iostream>
#include <stack>
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xio.hpp"
#include "detail/log.hpp"

#define HG_DEBUG

#ifndef __FUNCTION_NAME__
#ifdef WIN32   //WINDOWS
#define __FUNCTION_NAME__   __FUNCTION__
#else          //*NIX
#define __FUNCTION_NAME__   __func__
#endif
#endif

#ifdef HG_DEBUG
#define hg_assert(test, msg) do { \
    if(!(test)) {\
    throw std::runtime_error(std::string() + __FUNCTION_NAME__ + " in file " + __FILE__ + "(line:" + std::to_string(__LINE__) + "): "  + msg);} \
  } while (0)
#else
#define hg_assert(test, msg) ((void)0)
#endif


#define HG_XSTR(a) HG_STR(a)
#define HG_STR(a) #a


#define HG_TEMPLATE_SINTEGRAL_TYPES   char, short, int, long

#define HG_TEMPLATE_UINTEGRAL_TYPES   unsigned char, unsigned short, unsigned int, unsigned long

#define HG_TEMPLATE_INTEGRAL_TYPES    char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long

#define HG_TEMPLATE_FLOAT_TYPES       float, double

#define HG_TEMPLATE_NUMERIC_TYPES     char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, float, double

#define HG_TEMPLATE_SNUMERIC_TYPES     char, short, int, long, float, double

#define HG_SINTEGRAL_TYPES   (char)(short)(int)(long)

#define HG_UINTEGRAL_TYPES   (unsigned char)(unsigned short)(unsigned int)(unsigned long)

#define HG_INTEGRAL_TYPES    (char)(unsigned char)(short)(unsigned short)(int)(unsigned int)(long)(unsigned long)

#define HG_FLOAT_TYPES       (float)(double)

#define HG_NUMERIC_TYPES     (char)(unsigned char)(short)(unsigned short)(int)(unsigned int)(long)(unsigned long)(float)(double)


namespace xt {

    inline
    bool all(bool x) {
        return x;
    }

    inline
    bool any(bool x) {
        return x;
    }

    /**
     * Provides a view semantic over any xtensor container
     * @tparam E
     * @param e
     * @return
     */
    template<typename E,
            typename = std::enable_if_t<!std::is_base_of<xt::xexpression<E>, E>::value> >
    E &&view_all(E &&e) {
        return std::forward<E>(e);
    }

    /**
     * Provides a view semantic over any xtensor container
     * @tparam E
     * @param e
     * @return
     */
    template<typename E>
    auto view_all(xt::xcontainer_semantic<E> &&e) {
        return xt::strided_view(e, {});
    }

    /**
     * Provides a view semantic over any xtensor container
     * @tparam E
     * @param e
     * @return
     */
    template<typename E>
    auto &&view_all(xt::xview_semantic<E> &&e) {
        return std::forward<xt::xview_semantic<E>>(e);
    }

    /**
     * Provides a view semantic over any xtensor container
     * @tparam E
     * @param e
     * @return
     */
    template<typename E>
    auto view_all(xt::xscalar<E> &&e) {
        return xt::strided_view(e, {});
    }

}

namespace hg {

    /**
     * Preferred type to represent an index
     */
    using index_t = std::ptrdiff_t;

    /**
     * Constant used to represent an invalid index (eg. not initialized)
     */
    const index_t invalid_index = -1;

    /**
     * Preferred type to represent a size
     */
    using size_t = std::size_t;

    /**
     * Insert all elements of collection b at the end of collection a.
     * @tparam T1 must have an insert method (STL like) and a range interface (begin, end)
     * @tparam T2 must have a range interface (begin, end)
     * @param a
     * @param b
     */
    template<typename T1, typename T2>
    void extend(T1 &a, const T2 &b) {
        a.insert(std::end(a), std::begin(b), std::end(b));
    };

    template<typename T>
    using stackv = std::stack<T, std::vector<T>>;

    /**
     * Do not use except if you want a compile error showing the type of the provided template parameter !
     * @tparam T
     */
    template<typename T>
    struct COMPILE_ERROR;
}


