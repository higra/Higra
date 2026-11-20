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
    // (accumulate_parallel is skipped for first and last, which leave the value of leaves unset, and for
    // vectorial argmin and argmax, which only set the first component of leaves)
    template<typename T, typename accumulator_t>
    void check_single_node_accumulate(const array_nd<T> &input, const accumulator_t &acc, bool parallel = true) {
        hg::tree t1(array_1d<index_t>{0});
        hg::tree t2(array_1d<index_t>{0});
        t2.compute_children();
        REQUIRE(!t1.children_computed());

        if (parallel) {
            auto p1 = accumulate_parallel(t1, input, acc);
            auto p2 = accumulate_parallel(t2, input, acc);
            REQUIRE(same_values(p1, p2));
        }

        auto s1 = accumulate_sequential(t1, input, acc);
        auto s2 = accumulate_sequential(t2, input, acc);
        REQUIRE(same_values(s1, s2));
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
        check_single_node_accumulate(input, hg::accumulator_first(), false);
        check_single_node_accumulate(input, hg::accumulator_last(), false);
        check_single_node_accumulate(input, hg::accumulator_argmin(), input.dimension() == 1);
        check_single_node_accumulate(input, hg::accumulator_argmax(), input.dimension() == 1);

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
}