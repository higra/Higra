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
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#ifdef HG_USE_TBB
    #include <memory>
    #include <oneapi/tbb/global_control.h>
    #include <oneapi/tbb/info.h>
#endif
namespace py_sorting {

    namespace py = pybind11;


    struct def_sort {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("_sort", [](const hg::array_view<value_t> &array) {
                      auto xarray = hg::bridge::as_xt(array);
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
                      auto xarray = hg::bridge::as_xt(array);
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
                      return hg::arg_sort(hg::bridge::as_xt(array));
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
                      return hg::stable_arg_sort(hg::bridge::as_xt(array));
                  },
                  doc,
                  py::arg("array"));
        }
    };

#ifdef HG_USE_TBB
    static size_t max_threads = oneapi::tbb::info::default_concurrency();
    static std::unique_ptr<oneapi::tbb::global_control> tbb_thread_limit;
#endif

    void py_init_sorting(pybind11::module &m) {

        m.def("set_num_threads", [](size_t num_threads) {
#ifdef HG_USE_TBB
                  if (num_threads == 0) {
                      num_threads = max_threads;
                  }
                  tbb_thread_limit = std::make_unique<oneapi::tbb::global_control>(oneapi::tbb::global_control::max_allowed_parallelism, static_cast<int>(num_threads));
#else
                  HG_LOG_WARNING("Warning: trying to set maximum number of threads but Higra was compiled without multi-threading!");
#endif
              },
              "Set the maximum number of threads usable in parallel computing. If :attr:`num_threads` is equal to 0, "
              "the maximum number of threads resets to its default value (number of logical cores available on the machine).",
              py::arg("num_threads"));

        add_type_overloads<def_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_stable_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_arg_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
        add_type_overloads<def_stable_arg_sort, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
    }
}