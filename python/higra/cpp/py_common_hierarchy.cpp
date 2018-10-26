/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_common_hierarchy.hpp"

#include "py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/graph.hpp"
#include "higra/hierarchy/common.hpp"
#include <string>

template<typename T>
using pyarray = xt::pyarray<T>;

using namespace hg;
namespace py = pybind11;

template<typename tree_t>
struct def_node_weighted_tree {
    template<typename value_t, typename M>
    static
    void def(M &m, const char *doc) {
        using class_t = node_weighted_tree<tree_t, array_1d < value_t> >;
        auto c = py::class_<class_t>(m,
                                     (std::string("NodeWeightedTree_") + typeid(class_t).name()).c_str(),
                                     "A simple structure to hold the result of hierarchy construction algorithms, "
                                     "namely a tree and its associated node altitude array.");
        c.def("tree", [](class_t &self) -> tree_t& {return self.tree;}, "The tree!");
        c.def("altitudes", [](class_t &self) -> array_1d<value_t>& {return self.altitudes;}, "An array of tree node altitude.");
    }
};


void py_init_common_hierarchy(pybind11::module &m) {
    xt::import_numpy();
    add_type_overloads<def_node_weighted_tree<tree>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
}
