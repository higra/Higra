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
#include "higra/python/py_array.hpp"
#include "higra/structure/array/xtensor_bridge.hpp"
#include <cstddef>
#include <functional>

/*
 * Array inputs of the bindings are hg::array_view<const T> (or hg::array<T> when the binding keeps the
 * data), converted by higra/python/py_array.hpp: they are always C-contiguous. The xtensor-based
 * algorithms receive them through hg::bridge::as_xt (temporary, until the algorithms take views).
 *
 * The adaptors made by hg::bridge::as_xt do not own their elements: they refer to the argument of the
 * call. The two specializations below reject at compile time any binding that would return such an
 * adaptor to Python (the result would outlive the elements it refers to).
 */
namespace pybind11 {
    namespace detail {
        template<class P, class A, xt::layout_type L, class SC, class Tag>
        struct type_caster<xt::xarray_adaptor<xt::xbuffer_adaptor<P, xt::no_ownership, A>, L, SC, Tag>> {
            static_assert(sizeof(P) == 0,
                          "A non-owning xtensor adaptor (hg::bridge::as_xt) cannot be returned to Python: "
                          "return an owning array instead.");
        };

        template<class P, class A, std::size_t N, xt::layout_type L, class Tag>
        struct type_caster<xt::xtensor_adaptor<xt::xbuffer_adaptor<P, xt::no_ownership, A>, N, L, Tag>> {
            static_assert(sizeof(P) == 0,
                          "A non-owning xtensor adaptor (hg::bridge::as_xt) cannot be returned to Python: "
                          "return an owning array instead.");
        };
    }
}

namespace py_bridge {
    /**
     * hg::bridge::as_xt<N>(v) for a binding argument that was formerly an xt::pytensor<T, N>: an input of
     * another rank raises TypeError, as the pytensor converter did, instead of RuntimeError.
     */
    template<std::size_t N, typename T>
    auto as_xt_fixed(const hg::array_view<const T> &v, const char *arg_name) {
        if (v.dimension() != static_cast<hg::index_t>(N)) {
            throw pybind11::type_error(std::string("Argument '") + arg_name + "' must be an array of dimension " +
                                       std::to_string(N) + " (got dimension " + std::to_string(v.dimension()) +
                                       ").");
        }
        return hg::bridge::as_xt<N>(v);
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
