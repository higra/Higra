#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}
//example.cpython-36m-x86_64-linux-gnu /home/user/Desktop/Higra/lib/include/xtl
namespace py = pybind11;

PYBIND11_MODULE(example, m
) {
m.

doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

m.def("subtract", [](
int i,
int j
) {
return i -
j;
}, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
m.attr("__version__") = VERSION_INFO;
#else
m.attr("__version__") = "dev";
#endif
}