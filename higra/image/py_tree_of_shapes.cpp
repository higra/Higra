/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_of_shapes.hpp"
#include "../py_common.hpp"
#include "higra/image/tree_of_shapes.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include <string>

namespace py_tree_of_shapes {
    template<typename T>
    using pyarray = xt::pyarray<T>;

    namespace py = pybind11;


struct def_tree_of_shapes {
    template<typename value_t, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_component_tree_tree_of_shapes_image", [](const pyarray<value_t> &image,
                                                           const std::string &padding,
                                                           bool original_size,
                                                           bool immersion,
                                                           hg::index_t exterior_vertex) {
                  hg::tos_padding tpadding;
                  if (padding == "none") {
                      tpadding = hg::tos_padding::none;
                  } else if (padding == "zero") {
                      tpadding = hg::tos_padding::zero;
                  } else if (padding == "mean") {
                      tpadding = hg::tos_padding::mean;
                  } else {
                      throw std::runtime_error("tree_of_shapes: Unknown padding option.");
                  }

                  auto res = hg::component_tree_tree_of_shapes_image(image, tpadding, original_size, immersion,
                                                                   exterior_vertex);
                  return py::make_tuple(std::move(res.tree), std::move(res.altitudes));
              },
              doc,
              py::arg("image"),
              py::arg("padding") = "mean",
              py::arg("original_size") = true,
              py::arg("immersion") = true,
              py::arg("exterior_vertex") = 0
        );
    }
};


    void py_init_tree_of_shapes_image(pybind11::module &m) {
        //xt::import_numpy();

    add_type_overloads<def_tree_of_shapes, uint8_t, uint16_t, int32_t, int64_t, float, double>
            (m, "");


    }
}