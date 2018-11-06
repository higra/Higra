/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_hierarchy_mean_pb.hpp"
#include "../py_common.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/image/hierarchy_mean_pb.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "pybind11/functional.h"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

template<typename graph_t>
struct def_hierarchy_mean_pb {
    template<typename value_t, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_mean_pb_hierarchy", [](const graph_t &graph,
                                       const std::vector<size_t> &shape,
                                       const pyarray<value_t> &edge_weights,
                                       const pyarray<value_t> &edge_orientations) {
                  auto res = hg::mean_pb_hierarchy(graph,
                                                   hg::embedding_grid_2d(shape),
                                                   edge_weights,
                                                   edge_orientations);
                  return py::make_tuple(std::move(res.first.rag),
                                        std::move(res.first.vertex_map),
                                        std::move(res.first.edge_map),
                                        std::move(res.second.tree),
                                        std::move(res.second.altitudes)
                  );
              },
              doc,
              py::arg("graph"),
              py::arg("shape"),
              py::arg("edge_weights"),
              py::arg("edge_orientations") = pyarray<value_t>()
        );
    }
};


void py_init_hierarchy_mean_pb(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_hierarchy_mean_pb<hg::ugraph>, HG_TEMPLATE_FLOAT_TYPES>
            (m,
             "Compute the mean pb hierarchy as described in \n"
             "P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, \"Contour Detection and Hierarchical Image Segmentation,\" "
             "in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 5, pp. 898-916, May 2011.\n"
             "doi: 10.1109/TPAMI.2010.161\n"
             "\n"
             "This does not include gradient estimation."
            );


}