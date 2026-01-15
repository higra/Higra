/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_embedding.hpp"

#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/structure/embedding.hpp"
#include "xtensor/core/xeval.hpp"
#include <string>

namespace py_embedding {
    template<typename T>
    using pyarray = xt::pyarray<T>;

    using namespace hg;
    namespace py = pybind11;

    template<typename class_t>
    struct def_lin2grid {
        template<typename type, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("lin2grid",
                  [](const class_t &e, const pyarray<type> &a) {
                      return xt::eval(e.lin2grid(a));
                  },
                  doc,
                  py::arg("points"));
        }
    };

    template<typename class_t>
    struct def_contains {
        template<typename type, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("contains",
                  [](const class_t &e, const pyarray<type> &a) {
                      return xt::eval(e.contains(a));
                  },
                  doc,
                  py::arg("points"));
        }
    };

    template<typename class_t>
    struct def_grid2lin {
        template<typename type, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("grid2lin",
                  [](const class_t &e, const pyarray<type> &a) {
                      return xt::eval(e.grid2lin(a));
                  },
                  doc,
                  py::arg("points"));
        }
    };


    template<int dim>
    void py_init_embedding_impl(pybind11::module &m) {

        using class_t = embedding_grid<dim>;

        auto c = py::class_<class_t>(m,
                                     ("EmbeddingGrid" + std::to_string(dim) + "d").c_str(),
                                     py::dynamic_attr());
        /* c.def(py::init<const std::vector<hg::index_t> &>(),
               "Create a new grid embedding. Shape must be a 1d array with striclty positive values.",
               py::arg("shape"));*/

        c.def(py::init<const pyarray<hg::index_t> &>(),
              "Create a new grid embedding. Shape must be a 1d array with striclty positive values.",
              py::arg("shape"));

        c.def("shape", [](const class_t &e) { return pyarray<hg::index_t>(e.shape()); },
              "Get the shape/dimensions of the grid embedding");

        c.def("size", &class_t::size, "Get the total number of points contained in the embedding.");

        c.def("dimension", &class_t::dimension,
              "Get the dimension of the embedding (aka self.shape().size()).");

        c.def("contains", [](const class_t &e, const std::vector<hg::index_t> &a) { return e.contains(a); },
              "Takes a list or tuple representing the coordinates of a point and returns true if the point is contained in the embedding.",
              py::arg("coordinates"));

        add_type_overloads<def_contains<class_t>, HG_TEMPLATE_SINTEGRAL_TYPES>
                (c,
                 "Takes a n1 x n2 x ... nk  array, with nk = self.dimension(), and returns a boolean array of dimension n1 x n2 x ... n(k-1) "
                 "indicating if each point is contained in the embedding.");

        // ::TODO:: remove cast to xt::pyarray when xtesnor python supports xtensor_fixed
        c.def("lin2grid", [](const class_t &e, const index_t a) { return pyarray<hg::index_t>(e.lin2grid(a)); },
              "Compute the nd coordinates of a point given its linear coordinate.",
              py::arg("index"));

        add_type_overloads<def_lin2grid<class_t>, HG_TEMPLATE_SINTEGRAL_TYPES>
                (c,
                 "Takes a n1 x n2 x ... nk  array, and returns an array of dimension n1 x n2 x ... nk x self.dimension() "
                 "where each value, seen as the linear coordinates of a point, has been replaced by the corresponding nd coordinates.");


        c.def("grid2lin", [](const class_t &e, const std::vector<hg::index_t> &a) { return e.grid2lin(a); },
              "Compute the linear coordinate of a point given its nd coordinates.",
              py::arg("coordinates"));


        add_type_overloads<def_grid2lin<class_t>, HG_TEMPLATE_SINTEGRAL_TYPES>
                (c,
                 "Takes a n1 x n2 x ... nk  array, with nk = self.dimension(), and returns a uint64 array of dimension n1 x n2 x ... n(k-1) "
                 "giving the linear coordinate of each point.");


    }

    void py_init_embedding(pybind11::module &m) {
        //xt::import_numpy();
        py_init_embedding_impl<1>(m);
        py_init_embedding_impl<2>(m);
        py_init_embedding_impl<3>(m);
        py_init_embedding_impl<4>(m);
        py_init_embedding_impl<5>(m);
    }
}
