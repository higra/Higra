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
#include "xtensor/xadapt.hpp"
//#include "xtensor/xio.hpp"
#include "detail/log.hpp"

#ifdef  HG_USE_TBB

#include "tbb/tbb.h"

#endif

namespace hg {

    /**
     * Preferred type to represent an index
     */
    using index_t = int64_t;

    /**
     * Constant used to represent an invalid index (eg. not initialized)
     */
    const index_t invalid_index = -1;

    /**
     * Preferred type to represent a size
     */
    using size_t = std::size_t;
}

#define HG_MAIN_ASSERT

#ifndef __FUNCTION_NAME__
#ifdef WIN32   //WINDOWS
#define __FUNCTION_NAME__   __FUNCTION__
#else          //*NIX
#define __FUNCTION_NAME__   __func__
#endif
#endif

#ifdef HG_MAIN_ASSERT
#define hg_assert(test, msg) do { \
    if(!(test)) {\
    throw std::runtime_error(std::string() + __FUNCTION_NAME__ + " in file " + __FILE__ + "(line:" + std::to_string(__LINE__) + "): "  + msg);} \
  } while (0)

#define hg_assert_edge_weights(graph, edge_weights) do { \
    hg_assert(edge_weights.dimension() > 0, \
              "The dimension of the array '" #edge_weights "', representing edge data of the graph '" #graph "' must be at least 1.");\
    hg_assert(num_edges(graph) == edge_weights.shape()[0], "The dimension of the provided edge data array '" #edge_weights "' does "\
                                                           "not match the number of edges in the provided graph '" #graph "'.");\
    } while (0)

#define hg_assert_vertex_weights(graph, vertex_weights) do { \
    hg_assert(vertex_weights.dimension() > 0, \
              "The dimension of the array '" #vertex_weights "', representing vertex data of the graph '" #graph "' must be at least 1.");\
    hg_assert(num_vertices(graph) == vertex_weights.shape()[0], "The dimension of the provided vertex data array '" #vertex_weights "' does "\
                                                           "not match the number of vertices in the provided graph '" #graph "'.");\
    } while (0)

#define hg_assert_node_weights(tree, node_weights) do { \
    hg_assert(node_weights.dimension() > 0, \
              "The dimension of the array '" #node_weights "', representing node data of the tree '" #tree "' must be at least 1.");\
    hg_assert(num_vertices(tree) == node_weights.shape()[0], "The dimension of the provided node data array '" #node_weights "' does "\
                                                           "not match the number of nodes in the provided tree '" #tree "'.");\
    } while (0)

#define hg_assert_leaf_weights(tree, leaf_weights) do { \
    hg_assert(leaf_weights.dimension() > 0, \
              "The dimension of the array '" #leaf_weights "', representing leaves data of the tree '" #tree "' must be at least 1.");\
    hg_assert(num_leaves(tree) == leaf_weights.shape()[0], "The dimension of the provided leaf data array '" #leaf_weights "' does "\
                                                           "not match the number of leaves in the provided tree '" #tree "'.");\
    } while (0)

#define hg_assert_1d_array(array) do { \
    hg_assert(array.dimension() == 1, "The array '" #array "' must be 1d."); \
    } while (0)

#define hg_assert_integral_value_type(array) do { \
    static_assert(std::is_integral<typename std::decay_t<decltype(array)>::value_type>::value, "Array values of '" #array "' must be integral (char, short, int, long...)."); \
    } while (0)

#define hg_assert_same_shape(array1, array2) do { \
    hg_assert(xt::same_shape(array1.shape(), array2.shape()), "Shapes of '" #array1 "' and '" #array2 "' must be equal."); \
    } while (0)

#define hg_assert_component_tree(tree) do { \
    hg_assert(hg::category(tree) == hg::tree_category::component_tree, "The category of '" #tree "' must be 'component_tree'."); \
    } while (0)

#define hg_assert_partition_tree(tree) do { \
    hg_assert(hg::category(tree) == hg::tree_category::partition_tree, "The category of '" #tree "' must be 'partition_tree'."); \
    } while (0)

#define hg_assert_vertex_indices(graph, vertex_indices) do { \
    hg_assert((xt::amin)(vertex_indices)() >= 0, \
              "Vertex indices cannot be negative.");\
    hg_assert((hg::index_t)(xt::amax)(vertex_indices)() < (hg::index_t)hg::num_vertices(graph), "Vertex indices must be smaller than the number of vertices in the graph/tree.");\
    } while (0)

#define hg_assert_vertex_index(graph, vertex_index) do { \
    hg_assert(vertex_index >= 0, \
              "Vertex index cannot be negative.");\
    hg_assert((hg::index_t)vertex_index < (hg::index_t)hg::num_vertices(graph), "Vertex index must be smaller than the number of vertices in the graph/tree.");\
    } while (0)

#define hg_assert_edge_indices(graph, edge_indices) do { \
    hg_assert((xt::amin)(edge_indices)() >= 0, \
              "Edge indices cannot be negative.");\
    hg_assert((hg::index_t)(xt::amax)(edge_indices)() < (hg::index_t)hg::num_edges(graph), "Edge indices must be smaller than the number of edges in the graph/tree.");\
    } while (0)

#define hg_assert_edge_index(graph, edge_index) do { \
    hg_assert(edge_index >= 0, \
              "Edge index cannot be negative.");\
    hg_assert((hg::index_t)edge_index < (hg::index_t)hg::num_edges(graph), "Edge index must be smaller than the number of edges in the graph/tree.");\
    } while (0)
#else
#define hg_assert(test, msg) ((void)0)
#define hg_assert_vertex_weights(graph, vertex_weights) ((void)0)
#define hg_assert_edge_weights(graph, vertex_weights) ((void)0)
#define hg_assert_node_weights(tree, node_weights) ((void)0)
#define hg_assert_leaf_weights(tree, leaf_weights) ((void)0)
#define hg_assert_1d_array(array) ((void)0)
#define hg_assert_integral_value_type(array) ((void)0)
#define hg_assert_same_shape(array1, array2) ((void)0)
#define hg_assert_component_tree(tree) ((void)0)
#define hg_assert_partition_tree(tree) ((void)0)
#define hg_assert_vertex_indices(graph, vertex_indices)((void)0)
#define hg_assert_edge_indices(graph, edge_indices)((void)0)
#define hg_assert_vertex_index(graph, vertex_index)((void)0)
#define hg_assert_edge_index(graph, edge_index)((void)0)
#endif

#define HG_XSTR(a) HG_STR(a)
#define HG_STR(a) #a


#define HG_TEMPLATE_SINTEGRAL_TYPES   int8_t, int16_t, int32_t, int64_t

#define HG_TEMPLATE_INTEGRAL_TYPES    int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t

#define HG_TEMPLATE_FLOAT_TYPES       float, double

#define HG_TEMPLATE_NUMERIC_TYPES     int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double

#define HG_TEMPLATE_SNUMERIC_TYPES    int8_t, int16_t, int32_t, int64_t, float, double


namespace xt {

    inline
    bool all(bool x) {
        return x;
    }

    inline
    bool any(bool x) {
        return x;
    }

    template<typename value_type, typename T>
    auto adapt_struct_array(T *data, size_t offset, size_t size) {
        constexpr size_t struct_size = sizeof(T);
        constexpr size_t value_size = sizeof(value_type);

        static_assert(struct_size % value_size == 0, "Incorrect alignment (is this a packed struct?)");
        using shape_type = std::array<size_t, 1>;
        using cv_value_type = typename std::conditional<std::is_const<T>::value, typename std::add_const<value_type>::type, value_type>::type;
        shape_type shape = {size};
        shape_type stride = {struct_size / value_size};
        return xt::adapt((cv_value_type*)((char*)data + offset), size, xt::no_ownership(), shape, stride);
    }

#define HG_ADAPT_STRUCT_ARRAY(pointer, member, size) \
    (xt::adapt_struct_array<decltype(std::decay<decltype(*(pointer))>::type::member)>( \
        pointer, \
        offsetof(typename std::decay<decltype(*(pointer))>::type, member), \
        size))

}

namespace hg {

    template<typename lambda_t>
    void parfor(index_t start_index, index_t end_index, lambda_t fun, index_t step_size = 1) {
#ifdef HG_USE_TBB
        tbb::parallel_for(start_index, end_index, step_size, fun);
#else
        for (index_t i = start_index; i < end_index; i += step_size) {
            fun(i);
        }
#endif
    }


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


