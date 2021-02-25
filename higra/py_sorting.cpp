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

template<typename T>
using pyarray = xt::pyarray<T>;
namespace py = pybind11;


struct def_sort {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_sort", [](pyarray<value_t> &array) {
                  hg::sort(array);
              },
              doc,
              py::arg("array"));
    }
};

struct def_stable_sort {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_stable_sort", [](pyarray<value_t> &array) {
                  hg::stable_sort(array);
              },
              doc,
              py::arg("array"));
    }
};

struct def_arg_sort {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_arg_sort", [](pyarray<value_t> &array) {
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
        c.def("_stable_arg_sort", [](pyarray<value_t> &array) {
                  return hg::stable_arg_sort(array);
              },
              doc,
              py::arg("array"));
    }
};

#ifdef HG_USE_TBB
static size_t max_threads = tbb::task_scheduler_init::default_num_threads();
static tbb::task_scheduler_init tbb_scheduler(tbb::task_scheduler_init::default_num_threads());
#endif

void py_init_sorting(pybind11::module &m) {

    m.def("set_num_threads", [](size_t num_threads) {
#ifdef HG_USE_TBB
              if (num_threads == 0) {
                  num_threads = max_threads;
              }
              if (tbb_scheduler.is_active()) {
                  tbb_scheduler.terminate();
              }
              tbb_scheduler.initialize(num_threads);
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
