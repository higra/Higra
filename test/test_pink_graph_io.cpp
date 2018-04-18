//
// Created by user on 4/16/18.
//

#include <boost/test/unit_test.hpp>
#include "higra/io/pink_graph_io.hpp"
#include <sstream>
#include <string>
#include "test_utils.hpp"
#include "xtensor/xgenerator.hpp"

BOOST_AUTO_TEST_SUITE(pinkGraphIO);

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

    BOOST_AUTO_TEST_CASE(test_read) {


        istringstream in(s);

        auto res = read_pink_graph(in);

        std::vector<std::size_t> shape = {3, 5};

        std::vector<std::pair<std::size_t, std::size_t> > edges;
        for (std::size_t i = 0; i < 14; ++i)
            edges.push_back({i, i + 1});

        array_1d<double> vertex_weights = xt::arange<double>(1, 16);
        array_1d<double> edge_weights = {3, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0};

        std::vector<std::pair<std::size_t, std::size_t> > res_edges;
        for (auto e : edge_iterator(res.graph))
            res_edges.push_back(e);

        BOOST_CHECK(vectorEqual(edges, res_edges));
        BOOST_CHECK(vectorEqual(shape, res.shape));
        BOOST_CHECK(xt::allclose(vertex_weights, res.vertex_weights));
        BOOST_CHECK(xt::allclose(edge_weights, res.edge_weights));

    }


    BOOST_AUTO_TEST_CASE(test_save) {

        array_1d<double> vertex_weights = xt::arange<double>(1, 16);
        array_1d<double> edge_weights = {3, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0};
        std::vector<std::size_t> shape = {3, 5};

        ugraph g(15);

        std::vector<std::pair<std::size_t, std::size_t> > edges;
        for (std::size_t i = 0; i < 14; ++i)
            add_edge(i, i + 1, g);

        ostringstream out;
        save_pink_graph(out, g, vertex_weights, edge_weights, shape);
        string res = out.str();
        BOOST_CHECK(s == res);

    }

BOOST_AUTO_TEST_SUITE_END();