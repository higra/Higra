/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_energy_optimization.hpp"
#include "higra/algo/tree_energy_optimization.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;


/*
 * labelisation_optimal_cut_from_energy(const tree_type &tree,
                                              const xt::xexpression<T> &xenergy_attribute,
                                              const accumulator_type accumulator = hg::accumulator_sum())
 */
template<typename tree_t>
struct def_labelisation_optimal_cut_from_energy {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_labelisation_optimal_cut_from_energy", [](const tree_t &tree,
                                                          const pyarray<value_t> &energy_attribute,
                                                          const hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::labelisation_optimal_cut_from_energy(tree,
                                                                          energy_attribute,
                                                                          hg::accumulator_min());
                          break;
                      case hg::accumulators::max:
                          return hg::labelisation_optimal_cut_from_energy(tree,
                                                                          energy_attribute,
                                                                          hg::accumulator_max());
                          break;
                      case hg::accumulators::sum:
                          return hg::labelisation_optimal_cut_from_energy(tree,
                                                                          energy_attribute,
                                                                          hg::accumulator_sum());
                          break;
                      default:
                          throw std::runtime_error("Unsupported accumulator.");
                  }

              },

              doc,
              py::arg("tree"),
              py::arg("energy_attribute"),
              py::arg("accumulator"));
    }
};

template<typename tree_t>
struct def_hierarchy_to_optimal_energy_cut_hierarchy {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_hierarchy_to_optimal_energy_cut_hierarchy", [](const tree_t &tree,
                                                               const pyarray<value_t> &data_fidelity_attribute,
                                                               const pyarray<value_t> &regularization_attribute,
                                                               const int approximation_piecewise_linear_function) {
                  return hg::hierarchy_to_optimal_energy_cut_hierarchy(
                          tree,
                          data_fidelity_attribute,
                          regularization_attribute,
                          approximation_piecewise_linear_function);
              },
              doc,
              py::arg("tree"),
              py::arg("data_fidelity_attribute"),
              py::arg("regularization_attribute"),
              py::arg("approximation_piecewise_linear_function"));
    }
};


void py_init_tree_energy_optimization(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_labelisation_optimal_cut_from_energy<hg::tree>, HG_TEMPLATE_FLOAT_TYPES>(m, "");

    add_type_overloads<def_hierarchy_to_optimal_energy_cut_hierarchy<hg::tree>, HG_TEMPLATE_FLOAT_TYPES>(m, "");

    m.def("_binary_partition_tree_MumfordShah_energy", [](const hg::ugraph &graph,
                                                          const pyarray<double> &vertex_perimeter,
                                                          const pyarray<double> &vertex_area,
                                                          const pyarray<double> &vertex_values,
                                                          const pyarray<double> &squared_vertex_values,
                                                          const pyarray<double> &edge_length) {
              return hg::binary_partition_tree_MumfordShah_energy(
                      graph,
                      vertex_perimeter,
                      vertex_area,
                      vertex_values,
                      squared_vertex_values,
                      edge_length);
          },
          "",
          py::arg("graph"),
          py::arg("vertex_perimeter"),
          py::arg("vertex_area"),
          py::arg("vertex_values"),
          py::arg("squared_vertex_values"),
          py::arg("edge_length"));
}



