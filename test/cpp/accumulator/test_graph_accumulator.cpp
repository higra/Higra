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
#include <type_traits>
#include "higra/accumulator/graph_accumulator.hpp"
#include "higra/image/graph_image.hpp"

namespace test_graph_accumulator {

    using namespace hg;
    using namespace std;

    TEST_CASE("accumulator graph vertices", "[graph_accumulator]") {

        ugraph g = get_4_adjacency_graph({2, 3});

        array_1d<int> vertex_weights{1, 2, 3, 4, 5, 6};
        auto res1 = accumulate_graph_vertices(g, vertex_weights, accumulator_max());
        array_1d<int> ref1{4, 5, 6, 5, 6, 5};
        REQUIRE(hg::allclose(ref1, res1));

        array_2d<int> vertex_weights2{
                {1, 6},
                {2, 5},
                {3, 4},
                {4, 3},
                {5, 2},
                {6, 1}
        };
        auto res2 = accumulate_graph_vertices(g, vertex_weights2, accumulator_sum());
        array_2d<int> ref2{
                {6,  8},
                {9,  12},
                {8,  6},
                {6,  8},
                {12, 9},
                {8,  6}
        };
        REQUIRE(hg::allclose(ref2, res2));
    }

    TEST_CASE("accumulator graph edges", "[graph_accumulator]") {

        ugraph g = get_4_adjacency_graph({2, 3});

        array_1d<int> edge_weights{1, 2, 3, 4, 6, 5, 7};
        auto res1 = accumulate_graph_edges(g, edge_weights, accumulator_max());
        array_1d<int> ref1{2, 4, 6, 5, 7, 7};
        REQUIRE(hg::allclose(ref1, res1));

        array_2d<int> edge_weights2{
                {1, 6},
                {2, 5},
                {3, 4},
                {4, 3},
                {5, 2},
                {6, 1},
                {7, 9}
        };
        auto res2 = accumulate_graph_edges(g, edge_weights2, accumulator_sum());
        array_2d<int> ref2{
                {3,  11},
                {8,  13},
                {8,  6},
                {8,  6},
                {17, 13},
                {12, 11}
        };
        REQUIRE(hg::allclose(ref2, res2));
    }

    TEST_CASE("accumulator graph mean", "[graph_accumulator]") {
        ugraph g = get_4_adjacency_graph({2, 3});

        array_1d<double> edge_weights{1, 2, 3, 4, 6, 5, 7};
        auto sum_e = accumulate_graph_edges(g, edge_weights, accumulator_sum());
        auto count_e = accumulate_graph_edges(g, edge_weights, accumulator_counter());
        auto mean_e = accumulate_graph_edges(g, edge_weights, accumulator_mean());
        for (index_t i = 0; i < num_vertices(g); i++) {
            REQUIRE(mean_e(i) == sum_e(i) / (double) count_e(i));
        }
        REQUIRE(mean_e(0) == 1.5);

        array_2d<double> vertex_weights{{1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}};
        auto mean_v = accumulate_graph_vertices(g, vertex_weights, accumulator_mean());
        array_2d<double> ref{{3, 4}, {3, 4}, {4, 3}, {3, 4}, {4, 3}, {4, 3}};
        REQUIRE(hg::allclose(mean_v, ref));
    }

    TEST_CASE("accumulator graph empty neighbourhood and result types", "[graph_accumulator]") {
        ugraph g(3);
        add_edge(0, 1, g);

        array_1d<double> vertex_weights{1, 2, 3};
        array_1d<double> edge_weights{4};
        REQUIRE((accumulate_graph_vertices(g, vertex_weights, accumulator_first()) == array_1d<double>{2, 1, 0}));
        REQUIRE((accumulate_graph_vertices(g, vertex_weights, accumulator_last()) == array_1d<double>{2, 1, 0}));
        REQUIRE((accumulate_graph_edges(g, edge_weights, accumulator_first()) == array_1d<double>{4, 4, 0}));
        REQUIRE((accumulate_graph_edges(g, edge_weights, accumulator_mean()) == array_1d<double>{4, 4, 0}));

        array_2d<unsigned char> vertex_weights2{{1, 2}, {3, 4}, {5, 6}};
        auto argmin = accumulate_graph_vertices(g, vertex_weights2, accumulator_argmin());
        auto counter = accumulate_graph_vertices(g, vertex_weights2, accumulator_counter());
        static_assert(std::is_same<typename decltype(argmin)::value_type, index_t>::value);
        static_assert(std::is_same<typename decltype(counter)::value_type, index_t>::value);
        REQUIRE((argmin == array_1d<index_t>{0, 0, -1}));
        REQUIRE((counter == array_1d<index_t>{1, 1, 0}));
    }

    // star graph: vertex 0 is adjacent to the vertices 1, ..., n
    static ugraph star_graph(index_t n) {
        ugraph g(n + 1);
        for (index_t i = 1; i <= n; i++) {
            add_edge(0, i, g);
        }
        return g;
    }

    TEST_CASE("accumulator graph integer mean with a large degree", "[graph_accumulator]") {
        for (index_t n: {256, 257}) {
            auto g = star_graph(n);
            // the sum at vertex 0 is n - 255 (0 or 1): its mean is 0 with integer division
            array_1d<unsigned char> edge_weights = hg::zeros<unsigned char>({(size_t) n});
            edge_weights(0) = (unsigned char) (n - 256);
            auto res_e = accumulate_graph_edges(g, edge_weights, accumulator_mean());
            static_assert(std::is_same<typename decltype(res_e)::value_type, unsigned char>::value);
            REQUIRE(res_e(0) == 0);

            array_1d<unsigned char> vertex_weights = hg::zeros<unsigned char>({(size_t) n + 1});
            vertex_weights(1) = (unsigned char) (n - 256);
            REQUIRE(accumulate_graph_vertices(g, vertex_weights, accumulator_mean())(0) == 0);

            array_2d<unsigned char> edge_weights2 = hg::zeros<unsigned char>({(size_t) n, 2});
            edge_weights2(0, 1) = (unsigned char) (n - 256);
            auto res2 = accumulate_graph_edges(g, edge_weights2, accumulator_mean());
            REQUIRE((res2(0, 0) == 0 && res2(0, 1) == 0));

            array_2d<unsigned char> vertex_weights2 = hg::zeros<unsigned char>({(size_t) n + 1, 2});
            auto res3 = accumulate_graph_vertices(g, vertex_weights2, accumulator_mean());
            REQUIRE((res3(0, 0) == 0 && res3(0, 1) == 0));
        }

        // signed values: the integer mean truncates toward zero
        auto g = star_graph(128);
        array_1d<short> w = hg::zeros<short>({128});
        w(0) = -100;
        w(1) = -100;
        w(2) = -56;   // sum -256, mean -2
        REQUIRE(accumulate_graph_edges(g, w, accumulator_mean())(0) == -2);
        w(2) = -57;   // sum -257, mean -2.0078 truncated to -2
        REQUIRE(accumulate_graph_edges(g, w, accumulator_mean())(0) == -2);
        w(0) = -1;
        w(1) = 0;
        w(2) = 0;     // sum -1, mean truncated to 0
        REQUIRE(accumulate_graph_edges(g, w, accumulator_mean())(0) == 0);
    }
}
