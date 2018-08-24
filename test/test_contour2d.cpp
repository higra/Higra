/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>

#include "higra/image/contour_2d.hpp"

#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE(test_contour_2d);

    using namespace hg;
    using namespace std;

    template<typename graph_t, typename T>
    auto contour_2_khalimsky(const graph_t &graph, const T &shape, const contour_2d &contour, bool interp=false) {
        std::array<size_t, 2> res_shape{shape[0] * 2 -1, shape[1] * 2 - 1};
        array_2d<index_t> result = xt::zeros<index_t>(res_shape);
        embedding_grid_2d embedding{shape};
        long count = 0;

        for (const auto &polyline: contour) {

            for (const auto &segment: polyline) {
                count++;
                for (const auto &ei: segment) {
                    auto e = edge(ei, graph);
                    auto s = source(e, graph);
                    auto t = target(e, graph);
                    if (t > s) {
                        auto ti = embedding.lin2grid(t);
                        auto si = embedding.lin2grid(s);

                        result[ti + si] = count;
                    }
                }
            }
        }

        if(interp){
            embedding_grid_2d res_embedding(res_shape);
            auto adj4 = get_4_adjacency_implicit_graph(res_embedding);
            auto flat_res = xt::flatten(result);

            for (long y = 1; y < shape[0] * 2 - 2; y += 2) {
                for (long x = 1; x < shape[1] * 2 - 2; x += 2) {
                    auto v = res_embedding.grid2lin({y, x});
                    index_t max_v = std::numeric_limits<index_t>::lowest();
                    for (auto av: adjacent_vertex_iterator(v, adj4)) {
                        max_v = std::max(max_v, flat_res(av));
                    }
                    result(y, x) = max_v;
                }
            }
        }

        return result;
    }

    BOOST_AUTO_TEST_CASE(fit_contour_2d_empty) {

        auto g = get_4_adjacency_graph({4, 5});

        xt::xarray<int> data{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0};

        auto r = fit_contour_2d(g, {4, 5}, data);

        BOOST_CHECK(r.size() == 0);
    }

    BOOST_AUTO_TEST_CASE(fit_contour_2d_simple) {

        std::array<size_t, 2> shape{4, 5};
        auto g = get_4_adjacency_graph(shape);

        xt::xarray<int> data{0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 1, 1, 1, 2, 0, 3, 0, 0, 0, 0, 2,
                             3};


        xt::xarray<int> ref{{0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 2, 0, 3, 0},
                            {1, 0, 1, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 2, 0, 3, 0}};

        auto contours = fit_contour_2d(g, shape, data);

        auto contours_khalimsky = contour_2_khalimsky(g, shape, contours);
        BOOST_CHECK(is_in_bijection(ref, contours_khalimsky));
    }

    BOOST_AUTO_TEST_CASE(fit_contour_2d_more_complex) {

        std::array<size_t, 2> shape{4, 5};
        auto g = get_4_adjacency_graph(shape);

        xt::xarray<int> data{0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 1, 0, 2, 4, 3, 0, 0, 0, 1, 1, 1, 2, 0, 3, 0, 0, 0, 1, 2,
                             3};


        xt::xarray<int> ref{{0, 0, 0, 1, 0, 5, 0, 7, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 5, 0, 7, 0},
                            {0, 0, 0, 0, 4, 0, 0, 0, 0},
                            {0, 0, 0, 8, 0, 6, 0, 7, 0},
                            {2, 0, 2, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 3, 0, 6, 0, 7, 0}};

        auto contours = fit_contour_2d(g, shape, data);
        auto contours_khalimsky = contour_2_khalimsky(g, shape, contours);

        BOOST_CHECK(is_in_bijection(ref, contours_khalimsky));
    }

    BOOST_AUTO_TEST_CASE(contour_2d_subdivide_nothing) {

        std::array<size_t, 2> shape{4, 5};
        auto g = get_4_adjacency_graph(shape);

        xt::xarray<int> data{0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 1, 0, 2, 4, 3, 0, 0, 0, 1, 1, 1, 2, 0, 3, 0, 0, 0, 1, 2,
                             3};


        xt::xarray<int> ref{{0, 0, 0, 1, 0, 5, 0, 7, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 5, 0, 7, 0},
                            {0, 0, 0, 0, 4, 0, 0, 0, 0},
                            {0, 0, 0, 8, 0, 6, 0, 7, 0},
                            {2, 0, 2, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 3, 0, 6, 0, 7, 0}};

        auto contours = fit_contour_2d(g, shape, data);
        auto contours_subdivison = subdivide_contour(contours, g, embedding_grid_2d(shape));
        auto contours_khalimsky = contour_2_khalimsky(g, shape, contours_subdivison);


        BOOST_CHECK(is_in_bijection(ref, contours_khalimsky));
    }

    BOOST_AUTO_TEST_CASE(contour_2d_subdivide_simple) {

        std::array<size_t, 2> shape{4, 5};
        auto g = get_4_adjacency_graph(shape);

        xt::xarray<int> data{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0};

        xt::xarray<int> ref{{0, 0, 0, 1, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 0, 0, 0, 0},
                            {4, 0, 4, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0}};

        auto contours = fit_contour_2d(g, shape, data);
        auto contours_subdivison = subdivide_contour(contours, g, embedding_grid_2d(shape), 0.000001, false, 0);
        auto contours_khalimsky = contour_2_khalimsky(g, shape, contours_subdivison);

        BOOST_CHECK(is_in_bijection(ref, contours_khalimsky));
    }

    BOOST_AUTO_TEST_CASE(contour_2d_subdivide_simple2) {

        std::array<size_t, 2> shape{4, 5};
        auto g = get_4_adjacency_graph(shape);

        xt::xarray<int> data{0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0};

        auto c = contour2d_2_khalimsky(g, {shape}, data);

        xt::xarray<int> ref{{0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {1, 0, 0, 0, 2, 0, 0, 0, 4},
                            {0, 1, 0, 2, 0, 3, 0, 4, 0},
                            {0, 0, 1, 0, 0, 0, 3, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0}};

        auto contours = fit_contour_2d(g, shape, data);
        auto contours_subdivison = subdivide_contour(contours, g, embedding_grid_2d(shape), 0.000001, false, 0);
        auto contours_khalimsky = contour_2_khalimsky(g, shape, contours_subdivison);

        BOOST_CHECK(is_in_bijection(ref, contours_khalimsky));
    }

BOOST_AUTO_TEST_SUITE_END();