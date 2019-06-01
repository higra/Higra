/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "../graph.hpp"
#include <istream>
#include <ostream>
#include <fstream>
#include <string>
#include <vector>
#include "higra/structure/array.hpp"
#include "xtensor/xscalar.hpp"


namespace hg {

    template<typename A=array_1d<double>,
            typename B=A>
    struct pink_graph {
        ugraph graph;
        std::vector<std::size_t> shape;
        A vertex_weights;
        B edge_weights;
    };

    inline
    auto read_pink_graph(std::istream &in) {
        HG_TRACE();
        std::string discard;

        std::vector<std::size_t> shape;

        // maybe shape
        if (in.peek() == '#') {
            std::size_t rs;
            std::size_t cs;
            in >> discard >> rs >> discard >> cs;
            shape.push_back(cs);
            shape.push_back(rs);
        }

        index_t num_points;
        index_t num_edges;
        in >> num_points >> num_edges;

        hg_assert(num_points > 0, "The number of vertices cannot be negative.");
        hg_assert(num_edges > 0, "The number of edges cannot be negative.");

        if (shape.empty()) // construct valid shape
        {
            shape.push_back(num_points);
        }

        //useless line to announce vertex list
        in >> discard >> discard;

        // vertex list
        ugraph g(num_points);
        auto vertex_weight = array_1d<double>::from_shape({(size_t)num_points});

        for (index_t l = 0; l < num_points; ++l) {
            index_t i;
            double d;
            in >> i >> d;
            hg_assert(0 <= i && i < num_points, "Invalid graph file: vertex index out of range.");
            vertex_weight(i) = d;
        }

        //useless line to announce edge list
        in >> discard >> discard;

        auto edge_weight = array_1d<double>::from_shape({(size_t)num_edges});

        for (index_t l = 0; l < num_edges; ++l) {
            index_t i, j;
            double d;
            in >> i >> j >> d;
            hg_assert(0 <= i && 0 <= j && i < num_points && j < num_points,
                      "Invalid graph file: vertex index out of range in edge definition.");
            g.add_edge(i, j);
            edge_weight(l) = d;
        }

        return pink_graph<>{std::move(g), std::move(shape), std::move(vertex_weight), std::move(edge_weight)};
    }

    inline
    auto read_pink_graph(const std::string &filename) {
        std::ifstream file(filename);
        return read_pink_graph(file);
    };


    template<typename graph_t,
            typename T1,
            typename T2,
            typename S>
    auto save_pink_graph(std::ostream &out,
                         const graph_t &graph,
                         const xt::xexpression<T1> &xvertex_values = xt::xscalar<char>(0),
                         const xt::xexpression<T2> &xedge_values = xt::xscalar<char>(0),
                         S &shape = std::vector<std::size_t>()
    ) {
        HG_TRACE();
        auto &vertex_values = xvertex_values.derived_cast();
        auto &edge_values = xedge_values.derived_cast();

        hg_assert(vertex_values.dimension() <= 1, "Too many dimensions for vertex values!");
        hg_assert(edge_values.dimension() <= 1, "Too many dimensions for edge values!");

        switch (shape.size()) {
            case 0:
                break;
            case 1:
                out << "#rs " << shape[0] << " cs 1\n";
                break;
            case 2:
                out << "#rs " << shape[1] << " cs " << shape[0] << "\n";
                break;
            default:
                throw std::runtime_error("Too many dimensions !");
        }

        out << num_vertices(graph) << " " << num_edges(graph) << "\n";

        out << "val sommets\n";

        if (vertex_values.size() == 0) {
            for (std::size_t i = 0; i < num_vertices(graph); ++i) {
                out << i << " 1\n";
            }
        } else {
            hg_assert_vertex_weights(graph, vertex_values);
            for (std::size_t i = 0; i < num_vertices(graph); ++i) {
                out << i << " " << vertex_values(i) << "\n";
            }
        }

        out << "arcs values\n";

        if (edge_values.size() == 0) {
            for (auto e: edge_iterator(graph)) {
                out << source(e, graph) << " " << target(e, graph) << " 1\n";
            }
        } else {
            hg_assert_edge_weights(graph, edge_values);
            for (auto &e: edge_iterator(graph)) {
                out << source(e, graph) << " " << target(e, graph) << " " << edge_values(e) << "\n";
            }
        }

    }

    template<typename graph_t,
            typename T1,
            typename T2,
            typename S>
    void save_pink_graph(const std::string &filename,
                         const graph_t &graph,
                         const xt::xexpression<T1> &xvertex_values = xt::xscalar<char>(0),
                         const xt::xexpression<T2> &xedge_values = xt::xscalar<char>(0),
                         S &shape = std::vector<std::size_t>()
    ) {
        std::ofstream file(filename);
        save_pink_graph(file, graph, xvertex_values, xedge_values, shape);
    };

}
