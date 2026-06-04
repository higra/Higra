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
#include "higra/algo/watershed.hpp"
#include "higra/algo/graph_weights.hpp"
#include "higra/image/graph_image.hpp"
#include <random>
#include <unordered_map>
#include <algorithm>

using namespace hg;

namespace test_watershed {

    TEST_CASE("watershed cut simple", "[watershed_cut]") {

        // Fig 4 of Watershed Cuts: Minimum Spanning Forests and the
        // Drop of Water Principle
        // Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 5, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 3, 5, 4, 0, 7, 0, 3, 4, 0};

        auto labels = hg::labelisation_watershed(g, edge_weights);

        array_1d<index_t> expected{1, 1, 1, 2,
                                   1, 1, 2, 2,
                                   1, 1, 3, 3,
                                   1, 1, 3, 3};
        REQUIRE((labels == expected));
    }

    TEST_CASE("watershed cut simple 2", "[watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({3, 3});
        array_1d<int> edge_weights{1, 1, 0, 0, 0, 1, 0, 0, 2, 2, 0, 2};

        auto labels = hg::labelisation_watershed(g, edge_weights);

        array_1d<index_t> expected{1, 1, 1,
                                   2, 1, 1,
                                   2, 2, 1};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed 1", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0};
        array_1d<int> seeds{1, 1, 9, 9,
                            1, 9, 9, 9,
                            9, 9, 9, 9,
                            1, 1, 2, 2};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds, 9);

        array_1d<int> expected{1, 1, 2, 2,
                               1, 1, 2, 2,
                               1, 1, 2, 2,
                               1, 1, 2, 2};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed 2", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0};
        array_1d<int> seeds{1, 1, 0, 0,
                            1, 0, 0, 0,
                            0, 0, 0, 0,
                            2, 2, 3, 3};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);

        array_1d<int> expected{1, 1, 3, 3,
                               1, 1, 3, 3,
                               2, 2, 3, 3,
                               2, 2, 3, 3};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed split minima", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 4});
        array_1d<int> edge_weights{0, 1, 0, 2, 0, 2, 0, 1, 2, 1};
        /* x0x0x0x
         * 1 2 2 0
         * x1x2x1x */
        array_1d<int> seeds{1, 0, 0, 2,
                            0, 0, 0, 0};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
        /*
        * other possible results:
        * ((1, 1, 2, 2),
        * (1, 1, 2, 2))
        * or
        * ((1, 2, 2, 2),
        * (1, 1, 2, 2)) */
        array_1d<int> expected{1, 1, 1, 2,
                               1, 1, 2, 2};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed disconnected seed", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 3});
        array_1d<int> edge_weights{1, 0, 2, 0, 0, 1, 2};
        /* x1x2x
         * 0 0 0
         * x1x2x */
        array_1d<int> seeds{5, 7, 5,
                            0, 0, 0};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);

        array_1d<int> expected{5, 7, 5,
                               5, 7, 5};
        REQUIRE((labels == expected));
    }

    TEST_CASE("seeded watersed seed not in minima", "[seeded_watersed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 4});
        array_1d<int> edge_weights{0, 2, 0, 2, 1, 2, 2, 1, 0, 0};
        /* x0x0x1x
         * 2 2 2 2
         * x1x0x0x */
        array_1d<int> seeds{0, 0, 0, 1,
                            2, 0, 0, 0};
        auto labels = hg::labelisation_seeded_watershed(g, edge_weights, seeds);

        array_1d<int> expected{1, 1, 1, 1,
                               2, 2, 2, 2};
        REQUIRE((labels == expected));
    }

    TEST_CASE("incremental watershed cut basic", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        // Add seeds matching seeded watershed test 2
        array_1d<index_t> seed_v1{0, 1, 4};
        array_1d<index_t> seed_l1{1, 1, 1};
        iws.add_seeds(seed_v1, seed_l1);

        array_1d<index_t> seed_v2{12, 13};
        array_1d<index_t> seed_l2{2, 2};
        iws.add_seeds(seed_v2, seed_l2);

        array_1d<index_t> seed_v3{14, 15};
        array_1d<index_t> seed_l3{3, 3};
        iws.add_seeds(seed_v3, seed_l3);

        auto labels = iws.get_labeling();

        // Check that each seed vertex has the correct label
        REQUIRE(labels(0) == 1);
        REQUIRE(labels(1) == 1);
        REQUIRE(labels(4) == 1);
        REQUIRE(labels(12) == 2);
        REQUIRE(labels(13) == 2);
        REQUIRE(labels(14) == 3);
        REQUIRE(labels(15) == 3);

        // All vertices should be labeled (no background)
        for (index_t i = 0; i < 16; i++) {
            REQUIRE(labels(i) != 0);
        }
    }

    TEST_CASE("incremental watershed cut remove seed", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 3});
        array_1d<int> edge_weights{1, 0, 2, 0, 0, 1, 2};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        array_1d<index_t> sv1{0};
        array_1d<index_t> sl1{1};
        iws.add_seeds(sv1, sl1);

        array_1d<index_t> sv2{2};
        array_1d<index_t> sl2{2};
        iws.add_seeds(sv2, sl2);

        array_1d<index_t> sv3{4};
        array_1d<index_t> sl3{3};
        iws.add_seeds(sv3, sl3);

        // Three seeds => three regions
        auto labels3 = iws.get_labeling();
        REQUIRE(labels3(0) == 1);
        REQUIRE(labels3(2) == 2);
        REQUIRE(labels3(4) == 3);

        // Remove seed at vertex 4 => two regions remain
        array_1d<index_t> rm{4};
        iws.remove_seeds(rm);

        auto labels2 = iws.get_labeling();
        REQUIRE(labels2(0) == 1);
        REQUIRE(labels2(2) == 2);
        // Vertex 4 should now be in one of the remaining regions
        REQUIRE((labels2(4) == 1 || labels2(4) == 2));
    }

    TEST_CASE("incremental watershed cut shared labels", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 3});
        array_1d<int> edge_weights{1, 0, 2, 0, 0, 1, 2};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        // Two seeds with the same label (label 5) and one with label 7
        array_1d<index_t> sv{0, 2, 1};
        array_1d<index_t> sl{5, 5, 7};
        iws.add_seeds(sv, sl);

        auto labels = iws.get_labeling();
        REQUIRE(labels(0) == 5);
        REQUIRE(labels(2) == 5);
        REQUIRE(labels(1) == 7);
    }

    TEST_CASE("incremental watershed cut no seeds", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 2});
        array_1d<int> edge_weights{1, 2, 3, 4};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        auto labels = iws.get_labeling();
        for (index_t i = 0; i < 4; i++) {
            REQUIRE(labels(i) == 0);
        }
    }

    TEST_CASE("incremental watershed cut single seed", "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 2});
        array_1d<int> edge_weights{1, 2, 3, 4};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        array_1d<index_t> sv{0};
        array_1d<index_t> sl{1};
        iws.add_seeds(sv, sl);

        auto labels = iws.get_labeling();
        for (index_t i = 0; i < 4; i++) {
            REQUIRE(labels(i) == 1);
        }
    }

    TEST_CASE("incremental watershed cut batch remove equals sequential",
              "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1,
                                   5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0};

        array_1d<index_t> sv{0, 1, 4, 14, 15};
        array_1d<index_t> sl{1, 1, 1, 2, 2};

        auto a = hg::make_incremental_watershed_cut(g, edge_weights);
        a.add_seeds(sv, sl);
        array_1d<index_t> rm{1, 14};
        a.remove_seeds(rm);

        auto b = hg::make_incremental_watershed_cut(g, edge_weights);
        b.add_seeds(sv, sl);
        array_1d<index_t> rm1{1};
        array_1d<index_t> rm2{14};
        b.remove_seeds(rm1);
        b.remove_seeds(rm2);

        REQUIRE((a.get_labeling() == b.get_labeling()));
    }

    TEST_CASE("incremental watershed cut batch remove both sides of edge",
              "[incremental_watershed_cut]") {
        // Two seeds -> exactly one cut edge; remove both at once -> background everywhere.
        auto g = hg::get_4_adjacency_graph({2, 2});
        array_1d<int> edge_weights{1, 2, 3, 4};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);
        array_1d<index_t> sv{0, 3};
        array_1d<index_t> sl{1, 2};
        iws.add_seeds(sv, sl);
        iws.remove_seeds(sv);

        const auto &labels = iws.get_labeling();
        for (index_t i = 0; i < 4; i++) {
            REQUIRE(labels(i) == 0);
        }
    }

    TEST_CASE("incremental watershed cut interactive churn",
              "[incremental_watershed_cut]") {
        // Regression test for scenario 07: interactive single seed churn with label changes.
        auto g = hg::get_4_adjacency_graph({100, 100});
        const index_t n_edges = g.num_edges();
        array_1d<double> edge_weights = xt::empty<double>({n_edges});
        for (index_t i = 0; i < n_edges; ++i) {
            // Deterministic pseudo-random in [0, 1)
            edge_weights(i) = static_cast<double>(i % 997) / 997.0;
        }

        // Deterministic vertices matching scenario 07
        array_1d<index_t> sv{4098, 8671, 7466, 737, 5621,
                             5954, 3197, 7184, 7657, 3043};
        array_1d<index_t> initial_labels{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        const index_t changed_label = 99;

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

        // Add all 10 seeds
        iws.add_seeds(sv, initial_labels);
        auto labels = iws.get_labeling();

        // Baseline via full seeded watershed
        array_1d<index_t> seeds = xt::zeros<index_t>({g.num_vertices()});
        for (index_t i = 0; i < 10; ++i) {
            seeds(sv(i)) = initial_labels(i);
        }
        auto baseline = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
        REQUIRE((labels == baseline));

        // Churn first 5 seeds: remove then re-add with changed label
        for (index_t i = 0; i < 5; ++i) {
            index_t v = sv(i);
            array_1d<index_t> rm{v};
            array_1d<index_t> add_v{v};
            array_1d<index_t> add_l{changed_label};

            // Remove
            iws.remove_seeds(rm);
            seeds(v) = 0;
            labels = iws.get_labeling();
            baseline = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
            REQUIRE((labels == baseline));

            // Re-add with changed label
            iws.add_seeds(add_v, add_l);
            seeds(v) = changed_label;
            labels = iws.get_labeling();
            baseline = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
            REQUIRE((labels == baseline));
        }
    }

    TEST_CASE("incremental watershed cut decuts ancestor descendant",
              "[incremental_watershed_cut]") {
        // Path graph 1x8 with weights producing a balanced canonical BPT.
        // Seeds at 0, 1, 3, 4 then batch remove of 0, 3 produces two de-cuts
        // where one BPT node is a descendant of the other. Compare to the
        // full seeded watershed to verify Pass 2a handles cross-level de-cuts.
        auto g = hg::get_4_adjacency_graph({1, 8});
        array_1d<double> edge_weights{1, 5, 2, 7, 3, 6, 4};

        array_1d<index_t> sv{0, 1, 3, 4};
        array_1d<index_t> sl{10, 20, 30, 40};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);
        iws.add_seeds(sv, sl);
        array_1d<index_t> rm{0, 3};
        iws.remove_seeds(rm);

        array_1d<index_t> seeds = xt::zeros<index_t>({g.num_vertices()});
        seeds(1) = 20;
        seeds(4) = 40;
        auto expected = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
        REQUIRE((iws.get_labeling() == expected));
    }

    TEST_CASE("incremental watershed cut change seed label",
              "[incremental_watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({2, 2});
        array_1d<int> edge_weights{1, 2, 3, 4};

        auto iws = hg::make_incremental_watershed_cut(g, edge_weights);
        array_1d<index_t> sv{0};
        array_1d<index_t> sl{1};
        iws.add_seeds(sv, sl);
        array_1d<index_t> expected_1{1, 1, 1, 1};
        REQUIRE((iws.get_labeling() == expected_1));

        array_1d<index_t> sv2{0};
        array_1d<index_t> sl2{2};
        iws.add_seeds(sv2, sl2);
        array_1d<index_t> expected_2{2, 2, 2, 2};
        REQUIRE((iws.get_labeling() == expected_2));
    }

    TEST_CASE("incremental watershed cut randomized hardened", "[incremental_watershed_cut]") {
        const index_t size = 50;
        const int num_seed_removed = 4;
        const int marker_radius = 1;
        const index_t num_vertices_total = size * size;

        std::mt19937 rng(0);
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        std::uniform_int_distribution<int> dist_label(1, 9);
        std::uniform_int_distribution<int> dist_pos(0, (int)size - 1);

        auto g = hg::get_4_adjacency_graph({size, size});

        for (int img_idx = 0; img_idx < 10; ++img_idx) {
            // Generate a random flat image (size*size pixel values in [0,1))
            array_1d<double> image = xt::empty<double>({(size_t)num_vertices_total});
            for (index_t k = 0; k < num_vertices_total; ++k)
                image(k) = dist_real(rng);

            auto edge_weights = hg::weight_graph(g, image, hg::weight_functions::L1);

            // seeds array (flat, 0 = background)
            array_1d<index_t> seeds = xt::zeros<index_t>({(size_t)num_vertices_total});

            std::unordered_map<index_t, index_t> seed_dic;

            auto iws = hg::make_incremental_watershed_cut(g, edge_weights);

            for (int step = 0; step < 40; ++step) {
                bool maybe_add = (seed_dic.size() == 0) || (dist_real(rng) >= 0.3);

                if (maybe_add) {
                    index_t label = (index_t)dist_label(rng);
                    int x = dist_pos(rng);
                    int y = dist_pos(rng);

                    std::vector<index_t> new_seed_positions;
                    std::vector<index_t> new_seed_labels;

                    for (int dy = -marker_radius; dy <= marker_radius; ++dy) {
                        for (int dx = -marker_radius; dx <= marker_radius; ++dx) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < (int)size && ny >= 0 && ny < (int)size) {
                                index_t v = (index_t)(ny * size + nx);
                                if (seed_dic.find(v) == seed_dic.end()) {
                                    seeds(v) = label;
                                    seed_dic[v] = label;
                                    new_seed_positions.push_back(v);
                                    new_seed_labels.push_back(label);
                                }
                            }
                        }
                    }

                    if (!new_seed_positions.empty()) {
                        array_1d<index_t> sv = array_1d<index_t>::from_shape({new_seed_positions.size()});
                        array_1d<index_t> sl = array_1d<index_t>::from_shape({new_seed_labels.size()});
                        for (size_t k = 0; k < new_seed_positions.size(); ++k) {
                            sv(k) = new_seed_positions[k];
                            sl(k) = new_seed_labels[k];
                        }
                        iws.add_seeds(sv, sl);
                        auto ref = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
                        REQUIRE((iws.get_labeling() == ref));
                    }
                } else {
                    bool nuke = (dist_real(rng) >= 0.8);
                    std::vector<index_t> removed_seeds;
                    if (nuke)
                    {
                        // remove all
                        for (auto &kv : seed_dic) {
                            seeds(kv.first) = 0;
                            removed_seeds.push_back(kv.first);
                        }
                        seed_dic.clear();
                    } else
                    {
                        // Shuffle the current seed vertices and remove up to num_seed_removed of them
                        std::vector<index_t> keys;
                        keys.reserve(seed_dic.size());
                        for (auto &kv : seed_dic) keys.push_back(kv.first);
                        std::shuffle(keys.begin(), keys.end(), rng);
                        int n_remove = std::min((int)keys.size(), num_seed_removed);
                        keys.resize(n_remove);

                        for (auto v : keys) {
                            seeds(v) = 0;
                            seed_dic.erase(v);
                            removed_seeds.push_back(v);
                        }
                    }

                    array_1d<index_t> sv = array_1d<index_t>::from_shape({removed_seeds.size()});
                    for (size_t k = 0; k < removed_seeds.size(); ++k)
                        sv(k) = removed_seeds[k];
                    iws.remove_seeds(sv);
                    auto ref = hg::labelisation_seeded_watershed(g, edge_weights, seeds);
                    REQUIRE((iws.get_labeling() == ref));
                }
            }
        }
    }

}
