//
// Created by user on 4/18/18.
//

#pragma once

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

