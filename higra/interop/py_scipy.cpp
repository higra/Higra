/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_scipy.hpp"
#include "../py_common.hpp"
#include "higra/image/graph_image.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor/xview.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

using namespace hg;

template<typename tree_t, typename T1, typename T2, typename value_t=double>
auto binary_hierarchy_to_scipy_linkage_matrix(const tree_t &tree,
                                              const xt::xexpression<T1> &xaltitudes,
                                              const xt::xexpression<T2> &xarea) {
    auto &altitudes = xaltitudes.derived_cast();
    auto &area = xarea.derived_cast();
    hg_assert_node_weights(tree, altitudes);
    hg_assert_node_weights(tree, area);

    hg::index_t n_leaves = num_leaves(tree);
    hg::array_2d<value_t> M = xt::empty<value_t>({(size_t) n_leaves - 1, (size_t) 4});
    for (auto i: hg::leaves_to_root_iterator(tree, hg::leaves_it::exclude)) {
        hg::index_t n = i - n_leaves;
        hg_assert(hg::num_children(i, tree) == 2, "Input hierarchy must be a binary hierarchy.");
        M(n, 0) = (value_t)hg::child(0, i, tree);
        M(n, 1) = (value_t)hg::child(1, i, tree);
        M(n, 2) = (value_t)altitudes(i);
        M(n, 3) = (value_t)area(i);
    }
    return M;
};


template<typename T, typename value_t=typename T::value_type>
auto scipy_linkage_matrix_to_binary_hierarchy(const xt::xexpression<T> &xlinkage_matrix) {
    auto &linkage_matrix = xlinkage_matrix.derived_cast();
    hg_assert(linkage_matrix.dimension() == 2, "Linkage matrix must be a 2d array.");
    hg_assert(linkage_matrix.shape()[1] == 4, "Linkage matrix second dimension must be of size 4.");

    index_t n_leaves = linkage_matrix.shape()[0] + 1;
    index_t n_nodes = n_leaves * 2 - 1;

    array_1d<index_t> parents = xt::empty<index_t>({n_nodes});
    array_1d<value_t> altitudes = xt::empty<index_t>({n_nodes});
    array_1d<index_t> area = xt::empty<index_t>({n_nodes});

    xt::view(altitudes, xt::range(0, n_leaves)) = 0;
    xt::view(area, xt::range(0, n_leaves)) = 1;
    parents(n_nodes - 1) = n_nodes - 1;

    for (index_t i = 0; i < n_leaves - 1; i++) {
        index_t n = i + n_leaves;
        parents(static_cast<index_t>(linkage_matrix(i, 0))) = n;
        parents(static_cast<index_t>(linkage_matrix(i, 1))) = n;
        altitudes(n) = linkage_matrix(i, 2);
        area(n) = static_cast<index_t>(linkage_matrix(i, 3));
    }
    return std::make_tuple(hg::tree(parents), std::move(altitudes), std::move(area));
};

struct def_scipy_linkage_matrix_to_binary_hierarchy {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_scipy_linkage_matrix_to_binary_hierarchy", [](
                      const pyarray<value_t> &linkage_matrix) {
                  auto res = scipy_linkage_matrix_to_binary_hierarchy(linkage_matrix);
                  return py::make_tuple(std::move(std::get<0>(res)), std::move(std::get<1>(res)),
                                        std::move(std::get<2>(res)));
              },
              doc,
              py::arg("linkage_matrix"));
    }
};

template<typename tree_t>
struct def_binary_hierarchy_to_scipy_linkage_matrix {
    template<typename value_t>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_binary_hierarchy_to_scipy_linkage_matrix", [](
                      const tree_t &tree,
                      const pyarray<value_t> &altitudes,
                      const pyarray<hg::index_t> &area) {
                  return binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes, area);
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"),
              py::arg("area"));
    }
};

void py_init_scipy(pybind11::module &m) {
    add_type_overloads<def_binary_hierarchy_to_scipy_linkage_matrix<hg::tree>, HG_TEMPLATE_FLOAT_TYPES>
            (m,
             "Converts an Higra binary hierarchy to a SciPy linkage matrix."
            );
    add_type_overloads<def_scipy_linkage_matrix_to_binary_hierarchy, double>
            (m,
             "Converts a SciPy linkage matrix to an Higra binary hierarchy."
            );
}
