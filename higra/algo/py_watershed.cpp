/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_watershed.hpp"
#include "higra/algo/watershed.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py_watershed {

    template<typename T>
    using pyarray = xt::pyarray<T>;

    namespace py = pybind11;


    template<typename graph_t>
    struct def_labelisation_watershed {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_labelisation_watershed", [](const graph_t &graph, const pyarray<value_t> &edge_weights) {
                      return hg::labelisation_watershed(graph, edge_weights);
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("edge_weights"));
        }
    };

    template<typename graph_t>
    struct def_labelisation_seeded_watershed {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_labelisation_seeded_watershed",
                  [](const graph_t &graph,
                     const pyarray<value_t> &edge_weights,
                     const pyarray<hg::index_t> &vertex_seeds,
                     const hg::index_t background_label) {
                      return hg::labelisation_seeded_watershed(graph, edge_weights, vertex_seeds, background_label);
                  },
                  doc,
                  py::arg("graph"),
                  py::arg("edge_weights"),
                  py::arg("vertex_seeds"),
                  py::arg("background_label"));
        }
    };


    void py_init_watershed(pybind11::module &m) {
        //xt::import_numpy();

        add_type_overloads<def_labelisation_watershed<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_labelisation_seeded_watershed<hg::ugraph>, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    }
}



