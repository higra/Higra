/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/io/pink_graph_io.hpp"
#include <sstream>
#include <string>
#include "../test_utils.hpp"
#include "xtensor/xgenerator.hpp"

namespace pink_graph_io {
    
    using namespace hg;
    using namespace std;
    
    string s(
            "#rs 5 cs 3\n"
            "15 14\n"
            "val sommets\n"
            "0 1\n"
            "1 2\n"
            "2 3\n"
            "3 4\n"
            "4 5\n"
            "5 6\n"
            "6 7\n"
            "7 8\n"
            "8 9\n"
            "9 10\n"
            "10 11\n"
            "11 12\n"
            "12 13\n"
            "13 14\n"
            "14 15\n"
            "arcs values\n"
            "0 1 3\n"
            "1 2 0\n"
            "2 3 0\n"
            "3 4 1\n"
            "4 5 3\n"
            "5 6 0\n"
            "6 7 1\n"
            "7 8 0\n"
            "8 9 2\n"
            "9 10 0\n"
            "10 11 1\n"
            "11 12 0\n"
            "12 13 3\n"
            "13 14 0\n");

    string s2(
            "15 14\n"
            "val sommets\n"
            "0 1\n"
            "1 2\n"
            "2 3\n"
            "3 4\n"
            "4 5\n"
            "5 6\n"
            "6 7\n"
            "7 8\n"
            "8 9\n"
            "9 10\n"
            "10 11\n"
            "11 12\n"
            "12 13\n"
            "13 14\n"
            "14 15\n"
            "arcs values\n"
            "0 1 3\n"
            "1 2 0\n"
            "2 3 0\n"
            "3 4 1\n"
            "4 5 3\n"
            "5 6 0\n"
            "6 7 1\n"
            "7 8 0\n"
            "8 9 2\n"
            "9 10 0\n"
            "10 11 1\n"
            "11 12 0\n"
            "12 13 3\n"
            "13 14 0\n");

    TEST_CASE("read graph from stream", "[pink_graph_io]") {
            istringstream in(s);

            auto res = read_pink_graph(in);

            std::vector<size_t> shape = { 3, 5 };

            std::vector<ugraph::edge_descriptor > edges;
            for (index_t i = 0; i < 14; ++i)
            edges.emplace_back(i, i + 1, i);

            array_1d<double> vertex_weights = xt::arange<double>(1, 16);
            array_1d<double> edge_weights = { 3, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0 };

            std::vector<ugraph::edge_descriptor> res_edges;
            for (auto e : edge_iterator(res.graph))
            res_edges.push_back(e);

            REQUIRE(vectorEqual(edges, res_edges));
            REQUIRE(vectorEqual(shape, res.shape));
            REQUIRE(xt::allclose(vertex_weights, res.vertex_weights));
            REQUIRE(xt::allclose(edge_weights, res.edge_weights));
    }

    TEST_CASE("read graph from stream no shape", "[pink_graph_io]") {
        istringstream in(s2);

        auto res = read_pink_graph(in);

        std::vector<size_t> shape = {15};

        std::vector<ugraph::edge_descriptor > edges;
        for (index_t i = 0; i < 14; ++i)
            edges.emplace_back(i, i + 1, i);

        array_1d<double> vertex_weights = xt::arange<double>(1, 16);
        array_1d<double> edge_weights = { 3, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0 };

        std::vector<ugraph::edge_descriptor> res_edges;
        for (auto e : edge_iterator(res.graph))
            res_edges.push_back(e);

        REQUIRE(vectorEqual(edges, res_edges));
        REQUIRE(vectorEqual(shape, res.shape));
        REQUIRE(xt::allclose(vertex_weights, res.vertex_weights));
        REQUIRE(xt::allclose(edge_weights, res.edge_weights));
    }

    TEST_CASE("write graph to stream", "[pink_graph_io]") {
            array_1d<double> vertex_weights = xt::arange<double>(1, 16);
            array_1d<double> edge_weights = { 3, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0 };
            std::vector<size_t> shape = { 3, 5 };

            ugraph g(15);

            std::vector<std::pair<index_t, index_t> > edges;
            for (index_t i = 0; i < 14; ++i)
            add_edge(i, i + 1, g);

            ostringstream out;
            save_pink_graph(out, g, vertex_weights, edge_weights, shape);
            string res = out.str();
            REQUIRE(s == res);
    }

    TEST_CASE("write graph to stream no vertex weights no edge weights no shape", "[pink_graph_io]") {

        string s2(
                "15 14\n"
                "val sommets\n"
                "0 1\n"
                "1 1\n"
                "2 1\n"
                "3 1\n"
                "4 1\n"
                "5 1\n"
                "6 1\n"
                "7 1\n"
                "8 1\n"
                "9 1\n"
                "10 1\n"
                "11 1\n"
                "12 1\n"
                "13 1\n"
                "14 1\n"
                "arcs values\n"
                "0 1 1\n"
                "1 2 1\n"
                "2 3 1\n"
                "3 4 1\n"
                "4 5 1\n"
                "5 6 1\n"
                "6 7 1\n"
                "7 8 1\n"
                "8 9 1\n"
                "9 10 1\n"
                "10 11 1\n"
                "11 12 1\n"
                "12 13 1\n"
                "13 14 1\n");

        array_1d<double> vertex_weights{};
        array_1d<double> edge_weights{};
        std::vector<size_t> shape{};

        ugraph g(15);

        std::vector<std::pair<index_t, index_t> > edges;
        for (index_t i = 0; i < 14; ++i)
            add_edge(i, i + 1, g);

        ostringstream out;
        save_pink_graph(out, g, vertex_weights, edge_weights, shape);
        string res = out.str();
        REQUIRE(s2 == res);
    }

    TEST_CASE("write graph to stream no vertex weights no edge weights shape 1d", "[pink_graph_io]") {

        string s2(
                "#rs 15 cs 1\n"
                "15 14\n"
                "val sommets\n"
                "0 1\n"
                "1 1\n"
                "2 1\n"
                "3 1\n"
                "4 1\n"
                "5 1\n"
                "6 1\n"
                "7 1\n"
                "8 1\n"
                "9 1\n"
                "10 1\n"
                "11 1\n"
                "12 1\n"
                "13 1\n"
                "14 1\n"
                "arcs values\n"
                "0 1 1\n"
                "1 2 1\n"
                "2 3 1\n"
                "3 4 1\n"
                "4 5 1\n"
                "5 6 1\n"
                "6 7 1\n"
                "7 8 1\n"
                "8 9 1\n"
                "9 10 1\n"
                "10 11 1\n"
                "11 12 1\n"
                "12 13 1\n"
                "13 14 1\n");

        array_1d<double> vertex_weights{};
        array_1d<double> edge_weights{};
        std::vector<size_t> shape{15};

        ugraph g(15);

        std::vector<std::pair<index_t, index_t> > edges;
        for (index_t i = 0; i < 14; ++i)
            add_edge(i, i + 1, g);

        ostringstream out;
        save_pink_graph(out, g, vertex_weights, edge_weights, shape);
        string res = out.str();
        REQUIRE(s2 == res);
    }

    TEST_CASE("write graph to stream no vertex weights no edge weights invalid shape", "[pink_graph_io]") {

        array_1d<double> vertex_weights{};
        array_1d<double> edge_weights{};
        std::vector<size_t> shape{3, 5, 1};

        ugraph g(15);

        std::vector<std::pair<index_t, index_t> > edges;
        for (index_t i = 0; i < 14; ++i)
            add_edge(i, i + 1, g);

        ostringstream out;
        REQUIRE_THROWS(save_pink_graph(out, g, vertex_weights, edge_weights, shape));
    }
}