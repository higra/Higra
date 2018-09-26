/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_rag.hpp"
#include "higra/algo/rag.hpp"
#include "py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;
template<typename T, std::size_t N>
using pytensor = xt::pytensor<T, N, xt::layout_type::row_major>;

namespace py = pybind11;

template<typename graph_t>
struct def_make_rag {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_make_region_adjacency_graph", [](const graph_t &graph, const pyarray<value_t> &input) {
                  auto res = hg::make_region_adjacency_graph(graph, input);
                  return py::make_tuple(std::move(res.rag), std::move(res.vertex_map), std::move(res.edge_map));
              },
              doc,
              py::arg("graph"),
              py::arg("vertex_labels"));
    }
};

struct def_rag_back_project_weights {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_rag_back_project_weights",
              [](const xt::pyarray<hg::index_t> &rag_map, const pytensor<value_t, 1> &rag_weights) {
                  return hg::rag_back_project_weights(rag_map, rag_weights);
              },
              doc,
              py::arg("rag_map"),
              py::arg("rag_weights"));
    }
};

struct def_rag_accumulate {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_rag_accumulate",
              [](const pytensor<hg::index_t, 1> &rag_map,
                 const pyarray<value_t> &weights,
                 hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::rag_accumulate(rag_map, weights, hg::accumulator_min());
                          break;
                      case hg::accumulators::max:
                          return hg::rag_accumulate(rag_map, weights, hg::accumulator_max());
                          break;
                      case hg::accumulators::mean:
                          return hg::rag_accumulate(rag_map, weights, hg::accumulator_mean());
                          break;
                      case hg::accumulators::counter:
                          return hg::rag_accumulate(rag_map, weights, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::rag_accumulate(rag_map, weights, hg::accumulator_sum());
                          break;
                      case hg::accumulators::prod:
                          return hg::rag_accumulate(rag_map, weights, hg::accumulator_prod());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
              },
              doc,
              py::arg("rag_map"),
              py::arg("weights"),
              py::arg("accumulator")
        );
    }
};


struct def_project_fine_to_coarse_labelisation {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("project_fine_to_coarse_labelisation",
              [](const pyarray<value_t> &labelisation_fine,
                 const pyarray<value_t> &labelisation_coarse,
                 size_t num_regions_fine,
                 size_t num_regions_coarse) {
                  return hg::project_fine_to_coarse_labelisation(
                          labelisation_fine,
                          labelisation_coarse,
                          num_regions_fine,
                          num_regions_coarse);
              },
              doc,
              py::arg("labelisation_fine"),
              py::arg("labelisation_coarse"),
              py::arg("num_regions_fine") = 0,
              py::arg("num_regions_coarse") = 0);
    }
};

void py_init_rag(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_make_rag<hg::ugraph>, HG_TEMPLATE_INTEGRAL_TYPES>
            (m,
             "Create a region adjacency graph of the input graph with regions identified by the provided vertex labels.");

    add_type_overloads<def_rag_back_project_weights, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Projects vertex or edge weights defined on a region adjacency graph back to the original graph space.");

    add_type_overloads<def_rag_accumulate, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Accumulate vertex/edge weights of the original graph on vertex/edge of a region adjacency graph.");

    add_type_overloads<def_project_fine_to_coarse_labelisation, HG_TEMPLATE_INTEGRAL_TYPES>
            (m,
             "Given two labelisations, a fine and a coarse one, of a same set of elements.\n"
             "Find for each label (ie. region) of the fine labelisation, the label of the region in the\n"
             "coarse labelisation that maximises the intersection with the \"fine\" region.\n"
             "\n"
             "Pre-condition:\n"
             "\trange(labelisation_fine) = [0..num_regions_fine[\n"
             "\trange(labelisation_coarse) = [0..num_regions_coarse[\n"
             "\n"
             "If num_regions_fine or num_regions_coarse are not provided, they will\n"
             "be determined as max(labelisation_fine) + 1 and max(labelisation_coarse) + 1");

}


