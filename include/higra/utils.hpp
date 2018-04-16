//
// Created by user on 3/30/18.
//

#pragma once


#include <boost/preprocessor/seq/for_each.hpp>

#define HG_XSTR(a) HG_STR(a)
#define HG_STR(a) #a


#define HG_SINTEGRAL_TYPES   (char)(short)(int)(long)

#define HG_UINTEGRAL_TYPES   (unsigned char)(unsigned short)(unsigned int)(unsigned long)

#define HG_INTEGRAL_TYPES    (char)(unsigned char)(short)(unsigned short)(int)(unsigned int)(long)(unsigned long)

#define HG_FLOAT_TYPES       (float)(double)

#define HG_NUMERIC_TYPES     (char)(unsigned char)(short)(unsigned short)(int)(unsigned int)(long)(unsigned long)(float)(double)

//BOOST_PP_SEQ_FOR_EACH(f,  x, t)	f(r, x,t0) f(r, x,t1)...f(r, x,tk)
#define HG_FOREACH(f, t) BOOST_PP_SEQ_FOR_EACH(f, ~, t)