/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_accumulators.hpp"
#include "../py_common.hpp"
#include "higra/accumulator/accumulator.hpp"
namespace py_accumulator{
    namespace py = pybind11;

    void py_init_accumulators(pybind11::module &m) {

        // only exposes enumeration
        py::enum_<hg::accumulators>(m, "Accumulators",
                "The Accumulators class enumerates the various way of combining a list of values into a single value "
                "that are used in various algorithms.\n\n"
                "An accumulator can be applied to an empty list of values, for example to the children of a leaf in "
                ":func:`~higra.accumulate_parallel`: the result is then the *empty value* given for each accumulator "
                "below.\n\n"
                "If the values are vectors (or arrays of higher dimension), ``min``, ``max``, ``mean``, ``sum``, "
                "``prod``, ``first`` and ``last`` work component-wise and their result has the shape of a value. "
                "``counter``, ``argmin`` and ``argmax`` give a single integer (dtype ``int64``) whatever the shape of "
                "the values.\n\n"
                "``argmin`` (``argmax``) gives the position, starting at 0, of the smallest (largest) value in the "
                "list. For scalar values, the first smallest (largest) value is chosen. For vectors, the values are "
                "read in order and a value replaces the current choice if it is smaller (larger) than or equal to it "
                "on every component. Two vectors may be such that neither is smaller than the other: the result then "
                "depends on the order of the values. Among equal vectors, the last one is chosen.\n\n"
                "The functions that accumulate their own results (:func:`~higra.accumulate_sequential`, "
                ":func:`~higra.accumulate_and_add_sequential` and the similar functions, and "
                ":func:`~higra.propagate_sequential_and_accumulate`) do not accept ``argmin`` and ``argmax``, and "
                "their results have the dtype of the input values even with ``counter``.")
                .value("min", hg::accumulators::min,
                       "Minimum of the values; empty value: largest value of the dtype.")
                .value("max", hg::accumulators::max,
                       "Maximum of the values; empty value: lowest value of the dtype.")
                .value("mean", hg::accumulators::mean,
                       "Mean of the values; empty value: 0.")
                .value("counter", hg::accumulators::counter,
                       "Number of values; empty value: 0.")
                .value("sum", hg::accumulators::sum,
                       "Sum of the values; empty value: 0.")
                .value("prod", hg::accumulators::prod,
                       "Product of the values; empty value: 1.")
                .value("first", hg::accumulators::first,
                       "First value; empty value: 0.")
                .value("last", hg::accumulators::last,
                       "Last value; empty value: 0.")
                .value("argmin", hg::accumulators::argmin,
                       "Position of the smallest value; empty value: -1.")
                .value("argmax", hg::accumulators::argmax,
                       "Position of the largest value; empty value: -1.");
    }

}