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
#include "xtl/xmeta_utils.hpp"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "higra/utils.hpp"
#include <functional>


template <typename F, typename T, typename module_t, typename... Args>
void add_type_overloads(module_t & m, const char * doc, Args&&... args){
    F::template def<T>(m, doc, std::forward<Args>(args)...);
};

template <typename F, typename T1, typename... Ts, typename module_t, typename... Args>
typename std::enable_if<sizeof...(Ts) != 0>::type
add_type_overloads(module_t & m, const char * doc, Args&&... args){
    F::template def<T1>(m, doc, std::forward<Args>(args)...);
    add_type_overloads<F, Ts...>(m, "", std::forward<Args>(args)...);
};


#if defined(__GNUC__) && !defined(__clang__)
namespace workaround
{
    // Fixes "undefined symbol" issues with xtensor-python
    inline void extra_allocator()
    {
        std::allocator<long long> a;
        std::allocator<unsigned long long> b;
        std::allocator<double> c;
        std::allocator<std::complex<double>> d;
        std::allocator<unsigned short> e;
        std::allocator<short> f;
        std::allocator<float> g;
        std::allocator<unsigned int> h;
        std::allocator<int> i;
        std::allocator<unsigned char> j;
        std::allocator<char> k;
        if(!std::is_same<char, signed char>::value){
            std::allocator<signed char> l;
        }
        std::allocator<unsigned long> m;
        std::allocator<long> n;
        std::allocator<bool> o;
    }
}
#endif
