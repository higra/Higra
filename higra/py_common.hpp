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
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "higra/utils.hpp"
#include "higra/python/py_array.hpp"
#include <cstddef>
#include <string>
#include <functional>

/*
 * Array inputs of the bindings are hg::array_view<const T> (or hg::array<T> when the binding keeps the
 * data), converted by higra/python/py_array.hpp: they are always C-contiguous. Array results are returned
 * as hg::array<T>, moved to NumPy without copy.
 */
namespace py_args {
    /**
     * Checks the dimension of a binding argument that must have a fixed rank N: an input of another rank
     * raises TypeError. Returns the argument.
     */
    template<std::size_t N, typename T>
    const hg::array_view<const T> &fixed_rank(const hg::array_view<const T> &v, const char *arg_name) {
        if (v.dimension() != static_cast<hg::index_t>(N)) {
            throw pybind11::type_error(std::string("Argument '") + arg_name + "' must be an array of dimension " +
                                       std::to_string(N) + " (got dimension " + std::to_string(v.dimension()) +
                                       ").");
        }
        return v;
    }
}


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
