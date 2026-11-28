/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_common.hpp"
#include "py_sorting.hpp"
#include "higra/sorting.hpp"
#include "higra/parallel.hpp"
#include <algorithm>
#include <limits>
namespace py_sorting {

    namespace py = pybind11;


    struct def_sort {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_sort", [](const hg::array_view<value_t> &array) {
                      auto xarray = array;
                      hg::sort(xarray);
                  },
                  doc,
                  py::arg("array"));
        }
    };

    struct def_stable_sort {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_stable_sort", [](const hg::array_view<value_t> &array) {
                      auto xarray = array;
                      hg::stable_sort(xarray);
                  },
                  doc,
                  py::arg("array"));
        }
    };

    struct def_arg_sort {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_arg_sort", [](const hg::array_view<const value_t> &array) {
                      return hg::arg_sort(array);
                  },
                  doc,
                  py::arg("array"));
        }
    };

    struct def_stable_arg_sort {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_stable_arg_sort", [](const hg::array_view<const value_t> &array) {
                      return hg::stable_arg_sort(array);
                  },
                  doc,
                  py::arg("array"));
        }
    };

    void py_init_sorting(pybind11::module &m) {

        m.def("set_num_threads", [](size_t num_threads) {
                  hg::set_num_threads(static_cast<hg::index_t>(
                          (std::min)(num_threads, static_cast<size_t>(std::numeric_limits<hg::index_t>::max()))));
              },
              "Set the maximum number of threads usable in parallel computing. If :attr:`num_threads` is equal to 0, "
              "the maximum number of threads resets to its default value (number of logical cores available on the machine).\n\n"
              "The limit applies to the whole process. It can also be set with the environment variable "
              "``HG_NUM_THREADS`` when Higra is imported.\n\n"
              ":param num_threads: maximum number of threads (positive integer), or 0 to restore the default\n"
              ":raises TypeError: if :attr:`num_threads` is negative",
              py::arg("num_threads"));

        m.def("get_num_threads", &hg::get_num_threads,
              "Get the maximum number of threads usable in parallel computing: the value given to "
              ":func:`~higra.set_num_threads`, or the default value (number of logical cores available on the machine). "
              "Always 1 if Higra was compiled without multi-threading.\n\n"
              ":return: maximum number of threads");

        add_type_overloads<def_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_stable_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_arg_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_stable_arg_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    }
}