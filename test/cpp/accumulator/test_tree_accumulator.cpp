/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include <functional>
#include <limits>
#include <type_traits>


using namespace hg;
using namespace std;

namespace tree_accumulator {

    struct _data {

        hg::tree t;

        _data() : t(hg::array<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;

    TEST_CASE("accumulator tree scalar", "[tree_accumulator]") {

        auto tree = data.t;

        array_1d<index_t> input{1, 1, 1, 1, 1, 1, 1, 1};

        auto res1 = accumulate_parallel(tree, input, hg::accumulator_sum());
        array_1d<index_t> ref1{0, 0, 0, 0, 0, 2, 3, 2};
        REQUIRE(hg::allclose(ref1, res1));

        array_1d<index_t> vertex_data{1, 1, 1, 1, 1};
        auto res2 = accumulate_sequential(tree, vertex_data, hg::accumulator_sum());
        array_1d<index_t> ref2{1, 1, 1, 1, 1, 2, 3, 5};
        REQUIRE(hg::allclose(ref2, res2));

        auto res3 = accumulate_and_combine_sequential(tree, input, vertex_data, hg::accumulator_max(),
                                                      std::plus<index_t>());
        array_1d<index_t> ref3{1, 1, 1, 1, 1, 2, 2, 3};
        REQUIRE(hg::allclose(ref3, res3));

    }

    TEST_CASE("accumulator tree vectorial", "[tree_accumulator]") {

        auto tree = data.t;
        tree.compute_children();

        REQUIRE(tree.children_computed());
        array_2d<index_t> input{{1, 0},
                                {1, 1},
                                {1, 2},
                                {1, 3},
                                {1, 4},
                                {1, 5},
                                {1, 6},
                                {1, 7}};

        auto res1 = accumulate_parallel(tree, input, hg::accumulator_min());
        auto longmax = std::numeric_limits<index_t>::max();
        array_2d<index_t> ref1{{longmax, longmax},
                               {longmax, longmax},
                               {longmax, longmax},
                               {longmax, longmax},
                               {longmax, longmax},
                               {1,       0},
                               {1,       2},
                               {1,       5}};
        REQUIRE(hg::allclose(ref1, res1));

        array_2d<index_t> vertex_data{{1, 0},
                                      {1, 1},
                                      {1, 2},
                                      {1, 3},
                                      {1, 4}};
        auto res2 = accumulate_sequential(tree, vertex_data, hg::accumulator_sum());
        array_2d<index_t> ref2{{1, 0},
                               {1, 1},
                               {1, 2},
                               {1, 3},
                               {1, 4},
                               {2, 1},
                               {3, 9},
                               {5, 10}};
        REQUIRE(hg::allclose(ref2, res2));

        auto res3 = accumulate_and_combine_sequential(tree, input, vertex_data, hg::accumulator_sum(),
                                                      std::plus<index_t>());
        array_2d<index_t> ref3{{1, 0},
                               {1, 1},
                               {1, 2},
                               {1, 3},
                               {1, 4},
                               {3, 6},
                               {4, 15},
                               {8, 28}};
        REQUIRE(hg::allclose(ref3, res3));

    }

    TEST_CASE("accumulator tree vectorial no children", "[tree_accumulator]") {

        auto tree = data.t;

        REQUIRE(!tree.children_computed());

        array_2d<index_t> input{{1, 0},
                                {1, 1},
                                {1, 2},
                                {1, 3},
                                {1, 4},
                                {1, 5},
                                {1, 6},
                                {1, 7}};

        auto res1 = accumulate_parallel(tree, input, hg::accumulator_min());
        auto longmax = std::numeric_limits<index_t>::max();
        array_2d<index_t> ref1{{longmax, longmax},
                               {longmax, longmax},
                               {longmax, longmax},
                               {longmax, longmax},
                               {longmax, longmax},
                               {1,       0},
                               {1,       2},
                               {1,       5}};
        REQUIRE(hg::allclose(ref1, res1));

        array_2d<index_t> vertex_data{{1, 0},
                                      {1, 1},
                                      {1, 2},
                                      {1, 3},
                                      {1, 4}};
        auto res2 = accumulate_sequential(tree, vertex_data, hg::accumulator_sum());
        array_2d<index_t> ref2{{1, 0},
                               {1, 1},
                               {1, 2},
                               {1, 3},
                               {1, 4},
                               {2, 1},
                               {3, 9},
                               {5, 10}};
        REQUIRE(hg::allclose(ref2, res2));

        auto res3 = accumulate_and_combine_sequential(tree, input, vertex_data, hg::accumulator_sum(),
                                                      std::plus<index_t>());
        array_2d<index_t> ref3{{1, 0},
                               {1, 1},
                               {1, 2},
                               {1, 3},
                               {1, 4},
                               {3, 6},
                               {4, 15},
                               {8, 28}};
        REQUIRE(hg::allclose(ref3, res3));

    }

    TEST_CASE("propagate tree scalar", "[tree_accumulator]") {
        auto tree = data.t;
        array_1d<int> input{1, 2, 3, 4, 5, 6, 7, 8};
        array_1d<bool> condition{true, false, true, false, true, true, false, false};

        auto output = propagate_parallel(tree, input);
        array_1d<int> ref{6, 6, 7, 7, 7, 8, 8, 8};
        REQUIRE(hg::allclose(ref, output));

        auto output2 = propagate_parallel(tree, input, condition);
        array_1d<int> ref2{6, 2, 7, 4, 7, 8, 7, 8};
        REQUIRE(hg::allclose(ref2, output2));

        auto output3 = propagate_sequential(tree, input, condition);
        array_1d<int> ref3{8, 2, 7, 4, 7, 8, 7, 8};
        REQUIRE(hg::allclose(ref3, output3));

        auto output4 = propagate_sequential_and_accumulate(tree, input, hg::accumulator_sum());
        array_1d<int> ref4{15, 16, 18, 19, 20, 14, 15, 8};
        REQUIRE(hg::allclose(ref4, output4));

        auto output5 = propagate_sequential_and_accumulate(tree, input, hg::accumulator_prod(), condition);
        array_1d<int> ref5{48, 2, 21, 4, 35, 48, 7, 8};
        REQUIRE(hg::allclose(ref5, output5));

        condition(root(tree)) = true;
        auto output6 = propagate_sequential_and_accumulate(tree, input, hg::accumulator_counter(), condition);
        array_1d<int> ref6{2, 2, 2, 4, 2, 2, 7, 1};
        REQUIRE(hg::allclose(ref6, output6));
    }

    TEST_CASE("propagate tree vectorial", "[tree_accumulator]") {
        auto tree = data.t;
        array_2d<int> input{{1, 8},
                            {2, 7},
                            {3, 6},
                            {4, 5},
                            {5, 4},
                            {6, 3},
                            {7, 2},
                            {8, 1}};

        array_1d<bool> condition{true, false, true, false, true, true, false, false};

        auto output = propagate_parallel(tree, input);
        array_2d<int> ref{{6, 3},
                          {6, 3},
                          {7, 2},
                          {7, 2},
                          {7, 2},
                          {8, 1},
                          {8, 1},
                          {8, 1}};
        REQUIRE(hg::allclose(ref, output));

        auto output2 = propagate_parallel(tree, input, condition);
        array_2d<int> ref2{{6, 3},
                           {2, 7},
                           {7, 2},
                           {4, 5},
                           {7, 2},
                           {8, 1},
                           {7, 2},
                           {8, 1}};
        REQUIRE(hg::allclose(ref2, output2));

        auto output3 = propagate_sequential(tree, input, condition);
        array_2d<int> ref3{{8, 1},
                           {2, 7},
                           {7, 2},
                           {4, 5},
                           {7, 2},
                           {8, 1},
                           {7, 2},
                           {8, 1}};
        REQUIRE(hg::allclose(ref3, output3));

        auto output4 = propagate_sequential_and_accumulate(tree, input, hg::accumulator_sum());
        array_2d<int> ref4{{15, 12},
                           {16, 11},
                           {18, 9},
                           {19, 8},
                           {20, 7},
                           {14, 4},
                           {15, 3},
                           {8,  1}};
        REQUIRE(hg::allclose(ref4, output4));

        auto output5 = propagate_sequential_and_accumulate(tree, input, hg::accumulator_prod(), condition);
        array_2d<int> ref5{{48, 24},
                           {2, 7},
                           {21, 12},
                           {4, 5},
                           {35, 8},
                           {48, 3},
                           {7, 2},
                           {8,  1}};
        REQUIRE(hg::allclose(ref5, output5));
    }

    template<typename array_t>
    static bool has_shape(const array_t &a, const std::vector<size_t> &ref) {
        return std::equal(a.shape().begin(), a.shape().end(), ref.begin(), ref.end());
    }

    template<typename T>
    static void check_empty_tree_accumulators(const hg::tree &tree, const std::vector<size_t> &shape) {
        // shape: (0, *trailing dimensions) of the node and leaf data
        hg::array<T> input = hg::zeros<T>(array_shape::from(shape));
        array_1d<bool> condition = hg::zeros<bool>({0});

        REQUIRE(has_shape(accumulate_parallel(tree, input, hg::accumulator_sum()), shape));
        REQUIRE(has_shape(accumulate_parallel(tree, input, hg::accumulator_max()), shape));
        REQUIRE(has_shape(accumulate_sequential(tree, input, hg::accumulator_sum()), shape));
        REQUIRE(has_shape(accumulate_and_combine_sequential(tree, input, input, hg::accumulator_max(),
                                                            std::plus<T>()), shape));
        REQUIRE(has_shape(propagate_parallel(tree, input), shape));
        REQUIRE(has_shape(propagate_parallel(tree, input, condition), shape));
        REQUIRE(has_shape(propagate_sequential(tree, input, condition), shape));
        REQUIRE(has_shape(propagate_sequential_and_accumulate(tree, input, hg::accumulator_sum()), shape));
        REQUIRE(has_shape(propagate_sequential_and_accumulate(tree, input, hg::accumulator_prod(), condition),
                          shape));
    }

    TEST_CASE("accumulators and propagators on an empty tree", "[tree_accumulator]") {
        hg::tree tree(array_1d<index_t>{});
        REQUIRE(num_vertices(tree) == 0);

        SECTION("children not computed") {
            check_empty_tree_accumulators<index_t>(tree, {0});
            check_empty_tree_accumulators<double>(tree, {0, 3});
            REQUIRE(!tree.children_computed());
        }

        SECTION("children computed") {
            tree.compute_children();
            REQUIRE(tree.children_computed());
            check_empty_tree_accumulators<index_t>(tree, {0});
            check_empty_tree_accumulators<double>(tree, {0, 3});
        }
    }

    template<typename T1, typename T2>
    bool same_values(const T1 &a, const T2 &b) {
        if (!hg::same_shape(a.shape(), b.shape())) {
            return false;
        }
        for (index_t i = 0; i < (index_t) a.size(); i++) {
            auto x = a.data()[i];
            auto y = b.data()[i];
            if (!(x == y || (x != x && y != y))) {
                return false;
            }
        }
        return true;
    }

    // accumulates on a single-node tree, without and with computed children: the results must be equal
    // (argmin and argmax are rejected by accumulate_sequential)
    template<typename T, typename accumulator_t>
    void check_single_node_accumulate(const array_nd<T> &input, const accumulator_t &acc) {
        hg::tree t1(array_1d<index_t>{0});
        hg::tree t2(array_1d<index_t>{0});
        t2.compute_children();
        REQUIRE(!t1.children_computed());

        auto p1 = accumulate_parallel(t1, input, acc);
        auto p2 = accumulate_parallel(t2, input, acc);
        REQUIRE(same_values(p1, p2));

        if constexpr (!is_index_accumulator<accumulator_t>::value) {
            auto s1 = accumulate_sequential(t1, input, acc);
            auto s2 = accumulate_sequential(t2, input, acc);
            REQUIRE(same_values(s1, s2));
        } else {
            REQUIRE_THROWS(accumulate_sequential(t1, input, acc));
        }
        REQUIRE(!t1.children_computed());
    }

    // accumulate_and_combine_sequential on a single-node tree: the root is a leaf and takes the vertex data
    template<typename T, typename accumulator_t, typename combine_t>
    void check_single_node_combine(const array_nd<T> &input, const array_nd<T> &vertex_data,
                                   const accumulator_t &acc, const combine_t &combine) {
        hg::tree t1(array_1d<index_t>{0});
        hg::tree t2(array_1d<index_t>{0});
        t2.compute_children();

        auto r1 = accumulate_and_combine_sequential(t1, input, vertex_data, acc, combine);
        auto r2 = accumulate_and_combine_sequential(t2, input, vertex_data, acc, combine);
        REQUIRE(same_values(r1, r2));
        REQUIRE(same_values(r1, vertex_data));
        REQUIRE(!t1.children_computed());
    }

    template<typename T>
    void check_single_node_accumulators(const array_nd<T> &input, const array_nd<T> &vertex_data) {
        check_single_node_accumulate(input, hg::accumulator_sum());
        check_single_node_accumulate(input, hg::accumulator_mean());
        check_single_node_accumulate(input, hg::accumulator_min());
        check_single_node_accumulate(input, hg::accumulator_max());
        check_single_node_accumulate(input, hg::accumulator_counter());
        check_single_node_accumulate(input, hg::accumulator_prod());
        check_single_node_accumulate(input, hg::accumulator_first());
        check_single_node_accumulate(input, hg::accumulator_last());
        check_single_node_accumulate(input, hg::accumulator_argmin());
        check_single_node_accumulate(input, hg::accumulator_argmax());

        check_single_node_combine(input, vertex_data, hg::accumulator_sum(), std::plus<T>());
        check_single_node_combine(input, vertex_data, hg::accumulator_sum(), std::multiplies<T>());
        check_single_node_combine(input, vertex_data, hg::accumulator_max(),
                                  [](T a, T b) { return (std::min)(a, b); });
        check_single_node_combine(input, vertex_data, hg::accumulator_min(),
                                  [](T a, T b) { return (std::max)(a, b); });
        check_single_node_combine(input, vertex_data, hg::accumulator_mean(), std::plus<T>());
        check_single_node_combine(input, vertex_data, hg::accumulator_prod(), std::plus<T>());
    }

    TEST_CASE("accumulators on a single-node tree", "[tree_accumulator]") {
        SECTION("scalar") {
            check_single_node_accumulators<double>(array_nd<double>{1.}, array_nd<double>{2.});
            check_single_node_accumulators<index_t>(array_nd<index_t>{1}, array_nd<index_t>{2});
        }
        SECTION("vectorial") {
            check_single_node_accumulators<double>(array_nd<double>{{1., 3.}}, array_nd<double>{{2., 5.}});
            check_single_node_accumulators<index_t>(array_nd<index_t>{{1, 3}}, array_nd<index_t>{{2, 5}});
        }
    }

    // checks that every element of the rows [begin, end) of a equals v
    template<typename array_t, typename V>
    static void check_rows(const array_t &a, index_t begin, index_t end, V v) {
        index_t row = (a.dimension() == 0 || a.shape()[0] == 0) ? 0 : (index_t) (a.size() / a.shape()[0]);
        for (index_t i = begin * row; i < end * row; i++) {
            REQUIRE(a.data()[i] == v);
        }
    }

    template<typename T>
    static void check_parallel_leaves(hg::tree tree, const array_nd<T> &input) {
        const index_t nl = num_leaves(tree);
        const index_t n = num_vertices(tree);
        check_rows(accumulate_parallel(tree, input, hg::accumulator_sum()), 0, nl, 0);
        check_rows(accumulate_parallel(tree, input, hg::accumulator_mean()), 0, nl, 0);
        check_rows(accumulate_parallel(tree, input, hg::accumulator_prod()), 0, nl, 1);
        check_rows(accumulate_parallel(tree, input, hg::accumulator_min()), 0, nl, (std::numeric_limits<T>::max)());
        check_rows(accumulate_parallel(tree, input, hg::accumulator_max()), 0, nl, std::numeric_limits<T>::lowest());
        check_rows(accumulate_parallel(tree, input, hg::accumulator_first()), 0, nl, 0);
        check_rows(accumulate_parallel(tree, input, hg::accumulator_last()), 0, nl, 0);

        auto counter = accumulate_parallel(tree, input, hg::accumulator_counter());
        static_assert(std::is_same<typename decltype(counter)::value_type, index_t>::value);
        REQUIRE(has_shape(counter, {(size_t) n}));
        REQUIRE((counter == array_1d<index_t>{0, 0, 0, 0, 0, 2, 3, 2}));

        auto argmin = accumulate_parallel(tree, input, hg::accumulator_argmin());
        auto argmax = accumulate_parallel(tree, input, hg::accumulator_argmax());
        static_assert(std::is_same<typename decltype(argmin)::value_type, index_t>::value);
        static_assert(std::is_same<typename decltype(argmax)::value_type, index_t>::value);
        REQUIRE(has_shape(argmin, {(size_t) n}));
        REQUIRE(has_shape(argmax, {(size_t) n}));
        check_rows(argmin, 0, nl, -1);
        check_rows(argmax, 0, nl, -1);
    }

    TEST_CASE("accumulate_parallel on leaves and result types", "[tree_accumulator]") {
        hg::tree tree(hg::array<index_t>{5, 5, 6, 6, 6, 7, 7, 7});
        // node 5: children 0, 1; node 6: children 2, 3, 4; node 7: children 5, 6
        array_nd<double> scalar{3, 1, 2, 0, 5, 1, 1, 1};
        array_nd<int> vectorial{{1, 1}, {0, 0}, {2, 2}, {0, 1}, {5, 5}, {1, 1}, {1, 1}, {0, 0}};

        for (bool computed: {false, true}) {
            if (computed) {
                tree.compute_children();
            }
            check_parallel_leaves(tree, scalar);
            check_parallel_leaves(tree, vectorial);

            REQUIRE((accumulate_parallel(tree, scalar, hg::accumulator_argmin()) ==
                     array_1d<index_t>{-1, -1, -1, -1, -1, 1, 1, 0}));
            REQUIRE((accumulate_parallel(tree, scalar, hg::accumulator_argmax()) ==
                     array_1d<index_t>{-1, -1, -1, -1, -1, 0, 2, 0}));
            REQUIRE((accumulate_parallel(tree, vectorial, hg::accumulator_argmin()) ==
                     array_1d<index_t>{-1, -1, -1, -1, -1, 1, 1, 1}));
            REQUIRE((accumulate_parallel(tree, vectorial, hg::accumulator_argmax()) ==
                     array_1d<index_t>{-1, -1, -1, -1, -1, 0, 2, 1}));
        }
    }

    TEST_CASE("argmin and argmax with small integer types", "[tree_accumulator]") {
        // a root with 300 children: the selected index does not fit in the input type
        array_1d<index_t> parents = hg::zeros<index_t>({301});
        for (auto &p: parents) {
            p = 300;
        }
        hg::tree tree(parents);
        array_1d<unsigned char> input = hg::ones<unsigned char>({301});
        input(299) = 0;
        input(298) = 2;
        auto argmin = accumulate_parallel(tree, input, hg::accumulator_argmin());
        auto argmax = accumulate_parallel(tree, input, hg::accumulator_argmax());
        auto counter = accumulate_parallel(tree, input, hg::accumulator_counter());
        REQUIRE(argmin(300) == 299);
        REQUIRE(argmax(300) == 298);
        REQUIRE(counter(300) == 300);
        REQUIRE(argmin(0) == -1);
        REQUIRE(counter(0) == 0);
    }

    TEST_CASE("sequential functions reject argmin and argmax", "[tree_accumulator]") {
        auto tree = data.t;
        array_1d<double> input{1, 2, 3, 4, 5, 6, 7, 8};
        array_1d<double> leaf_data{1, 2, 3, 4, 5};
        array_1d<bool> condition{true, false, true, false, true, true, false, false};
        for (int i = 0; i < 2; i++) {
            if (i == 1) {
                tree.compute_children();
            }
            REQUIRE_THROWS(accumulate_sequential(tree, leaf_data, hg::accumulator_argmin()));
            REQUIRE_THROWS(accumulate_sequential(tree, leaf_data, hg::accumulator_argmax()));
            REQUIRE_THROWS(accumulate_and_combine_sequential(tree, input, leaf_data, hg::accumulator_argmin(),
                                                             std::plus<double>()));
            REQUIRE_THROWS(propagate_sequential_and_accumulate(tree, input, hg::accumulator_argmax()));
            REQUIRE_THROWS(propagate_sequential_and_accumulate(tree, input, hg::accumulator_argmin(), condition));
        }
    }

    TEST_CASE("counter keeps the input type in sequential functions", "[tree_accumulator]") {
        auto tree = data.t;
        array_1d<double> input{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5};
        array_1d<double> leaf_data{0.5, 0.5, 0.5, 0.5, 0.5};
        array_1d<bool> condition{true, false, true, false, true, true, false, false};

        auto r1 = propagate_sequential_and_accumulate(tree, input, hg::accumulator_counter(), condition);
        static_assert(std::is_same<typename decltype(r1)::value_type, double>::value);
        REQUIRE((r1 == array_1d<double>{2, 2.5, 2, 4.5, 2, 2, 7.5, 8.5}));

        auto r2 = accumulate_and_combine_sequential(tree, input, leaf_data, hg::accumulator_counter(),
                                                    std::plus<double>());
        static_assert(std::is_same<typename decltype(r2)::value_type, double>::value);
        REQUIRE((r2 == array_1d<double>{0.5, 0.5, 0.5, 0.5, 0.5, 8.5, 10.5, 10.5}));
    }

    TEST_CASE("mean accumulator on several nodes", "[tree_accumulator]") {
        array_1d<double> input{0, 1, 2, 3, 4, 5, 6, 7};
        array_1d<double> leaf_data{0, 1, 2, 3, 4};
        array_1d<double> ref_parallel{0, 0, 0, 0, 0, 0.5, 3, 5.5};
        array_1d<double> ref_sequential{0, 1, 2, 3, 4, 0.5, 3, 1.75};
        for (bool computed: {false, true}) {
            hg::tree tree(hg::array<index_t>{5, 5, 6, 6, 6, 7, 7, 7});
            if (computed) {
                tree.compute_children();
            }
            REQUIRE(hg::allclose(accumulate_parallel(tree, input, hg::accumulator_mean()), ref_parallel));
            REQUIRE(hg::allclose(accumulate_sequential(tree, leaf_data, hg::accumulator_mean()), ref_sequential));
        }
        hg::tree tree(hg::array<index_t>{5, 5, 6, 6, 6, 7, 7, 7});
        auto propagated = propagate_sequential_and_accumulate(tree, input, hg::accumulator_mean());
        // output(root) = input(root), output(i) = (output(parent(i)) + input(i)) / 2
        array_1d<double> ref{(6 + 0) / 2., (6 + 1) / 2., (6.5 + 2) / 2, (6.5 + 3) / 2, (6.5 + 4) / 2,
                             (7 + 5) / 2., (7 + 6) / 2., 7};
        REQUIRE(hg::allclose(propagated, ref));
    }
}
