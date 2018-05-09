//
// Created by user on 3/30/18.
//

#pragma once


#include <exception>
#include <string>
#include <iostream>
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xio.hpp"

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

    template<typename E,
            typename = std::enable_if_t<!std::is_base_of<xt::xexpression<E>, E>::value> >
    E &&view_all(E &&e) {
        return std::forward<E>(e);
    }

    template<typename E>
    auto view_all(xt::xcontainer_semantic<E> &&e) {
        return xt::dynamic_view(e, {});
    }

    template<typename E>
    auto &&view_all(xt::xview_semantic<E> &&e) {
        return std::forward<xt::xview_semantic<E>>(e);
    }

    template<typename E>
    auto view_all(xt::xscalar<E> &&e) {
        return xt::dynamic_view(e, {});
    }

}