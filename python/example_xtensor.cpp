//
// Created by user on 3/13/18.
//

#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyvectorize.hpp"
#include <numeric>
#include <cmath>

namespace py = pybind11;

double scalar_func(double i, double j) {
    return std::sin(i) - std::cos(j);
}

PYBIND11_MODULE(xtensor_python_test, m
)
{
xt::import_numpy();

m.

doc() = "Test module for xtensor python bindings";

m.def("vectorized_func",
xt::pyvectorize(scalar_func),
"");
}