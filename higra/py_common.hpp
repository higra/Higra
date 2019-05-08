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


template<typename F, typename T, typename module_t, typename... Args>
void add_type_overloads(module_t &m, const char *doc, Args &&... args) {
    F::template def<T>(m, doc, std::forward<Args>(args)...);
};

template<typename F, typename T1, typename... Ts, typename module_t, typename... Args>
typename std::enable_if<sizeof...(Ts) != 0>::type
add_type_overloads(module_t &m, const char *doc, Args &&... args) {
    F::template def<T1>(m, doc, std::forward<Args>(args)...);
    add_type_overloads<F, Ts...>(m, "", std::forward<Args>(args)...);
};


// workaround for undefined symbol bug with manylinux build
namespace xt {
#if defined(__GNUC__) && !defined(__clang__)
    namespace workaround {
        inline void long_allocator() {
            std::allocator<long long> a;
            std::allocator<unsigned long long> b;
        }
    }
#endif
}

