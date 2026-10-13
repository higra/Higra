/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/graph.hpp"
#include "../test_utils.hpp"


/**
 * graph tests
 */

//typedef boost::mpl::list<hg::ugraph, hg::undirected_graph<hg::hash_setS> > test_types;
namespace test_undirected_graph {

    using namespace std;
    using namespace hg;

    template<typename T>
    struct data {
        // 0 - 1
        // | /
        // 2   3
        static auto g() {
            T g(4ul);
            add_edge(0, 1, g);
            add_edge(1, 2, g);
            add_edge(0, 2, g);
            return g;
        }

    };

    TEMPLATE_TEST_CASE("undirected graph size", "[undirected_graph]", hg::ugraph, hg::undirected_graph<hg::hash_setS>) {

        SECTION("check size") {
            auto g = data<TestType>::g();

            REQUIRE(num_vertices(g) == 4);
            REQUIRE(num_edges(g) == 3);
            REQUIRE(out_degree(0, g) == 2);
            REQUIRE(in_degree(0, g) == 2);
            REQUIRE(degree(0, g) == 2);
            REQUIRE(out_degree(3, g) == 0);
            REQUIRE(in_degree(3, g) == 0);
            REQUIRE(degree(3, g) == 0);

            array_2d<index_t> indices{{0, 3},
                                      {1, 2}};
            array_2d<size_t> ref{{2, 0},
                                 {2, 2}};

            REQUIRE(xt::allclose(degree(indices, g), ref));
            REQUIRE(xt::allclose(in_degree(indices, g), ref));
            REQUIRE(xt::allclose(out_degree(indices, g), ref));
        }
        SECTION("copy graph specialized") {
            auto g = data<TestType>::g();

            vector<pair<index_t, index_t>> eref{{0, 1},
                                                {1, 2},
                                                {0, 2}};
            vector<pair<index_t, index_t>> etest;
            for (auto e: hg::edge_iterator(g)) {
                etest.push_back(e);
            }
            REQUIRE(vectorSame(eref, etest));
        }
        SECTION("copy graph generic") {
            hg::embedding_grid_2d embedding{2, 3}; // 2 rows, 3 columns
            std::vector<point_2d_i> neighbours{point_2d_i{{-1l, 0l}},
                                               point_2d_i{{0l, -1l}},
                                               point_2d_i{{0l, 1l}},
                                               point_2d_i{{1l, 0l}}}; // 4 adjacency

            auto g0 = hg::regular_grid_graph_2d(embedding, neighbours);
            auto g = hg::copy_graph<TestType>(g0);

            vector<vector<pair<index_t, index_t>>> outListsRef{{{0, 1}, {0, 3}},
                                                               {{1, 0}, {1, 2}, {1, 4}},
                                                               {{2, 1}, {2, 5}},
                                                               {{3, 0}, {3, 4}},
                                                               {{4, 1}, {4, 3}, {4, 5}},
                                                               {{5, 2}, {5, 4}}
            };
            vector<vector<pair<index_t, index_t>>> outListsTest;

            for (size_t v = 0; v < 6; v++) {
                outListsTest.push_back({});
                for (auto e: hg::out_edge_iterator(v, g))
                    outListsTest[v].push_back({source(e, g), target(e, g)});
                REQUIRE(vectorSame(outListsRef[v], outListsTest[v]));
                REQUIRE(out_degree(v, g) == outListsRef[v].size());

            }
        }
        SECTION("vertex iterator") {
            auto g = data<TestType>::g();

            vector<index_t> vref{0, 1, 2, 3};
            vector<index_t> vtest;

            for (auto v:hg::vertex_iterator(g)) {
                vtest.push_back(v);
            }

            REQUIRE(vectorEqual(vref, vtest));
        }
        SECTION("add vertex and edge") {
            TestType g{};
            hg::add_vertices(4, g);
            add_edge(0, 3, g);

            REQUIRE(num_vertices(g) == 4);
        }
        SECTION("edge iterator") {
            auto g = data<TestType>::g();

            vector<pair<index_t, index_t>> eref{{0, 1},
                                                {1, 2},
                                                {0, 2}};
            vector<pair<index_t, index_t>> etest;
            for (auto e: hg::edge_iterator(g)) {
                etest.push_back(e);
            }

            REQUIRE(vectorEqual(eref, etest));
        }
        SECTION("graph out edge iterator") {
            auto g = data<TestType>::g();

            vector<vector<pair<index_t, index_t>>> outListsRef{
                    {{0, 1}, {0, 2}},
                    {{1, 0}, {1, 2}},
                    {{2, 1}, {2, 0}},
                    {}
            };
            vector<vector<pair<index_t, index_t>>> outListsTest;

            for (auto v:hg::vertex_iterator(g)) {
                outListsTest.push_back(vector<pair<index_t, index_t>>());
                for (auto e:hg::out_edge_iterator(v, g)) {
                    outListsTest[v].push_back({source(e, g),
                                               target(e, g)});
                }
                REQUIRE(vectorSame(outListsRef[v], outListsTest[v]));
            }

        }

        SECTION("in edge iterator") {
            auto g = data<TestType>::g();

            vector<vector<pair<index_t, index_t>>> inListsRef{
                    {{1, 0}, {2, 0}},
                    {{0, 1}, {2, 1}},
                    {{1, 2}, {0, 2}},
                    {}};

            for (auto v: hg::vertex_iterator(g)) {
                vector<pair<index_t, index_t>> inListTest;

                for (auto e: hg::in_edge_iterator(v, g))
                    inListTest.push_back(e);
                REQUIRE(vectorSame(inListsRef[v], inListTest));
            }
        }

        SECTION("add edges") {
            auto g = data<TestType>::g();

            ugraph g2(4);

            array_1d<int> sources{0, 1, 0};
            array_1d<int> targets{1, 2, 2};
            add_edges(sources, targets, g2
            );

            REQUIRE(num_edges(g2) == 3);

            for (index_t i = 0; i < (index_t) num_edges(g2); i++) {
                auto e1 = edge_from_index(i, g);
                auto e2 = edge_from_index(i, g2);
                REQUIRE(e1 == e2);
            }
        }

        SECTION("adjacent vertex iterator") {
            auto g = data<TestType>::g();

            vector<vector<index_t>> adjListsRef{{1, 2},
                                                {0, 2},
                                                {1, 0},
                                                {}};
            vector<vector<index_t>> adjListsTest;

            for (auto v: hg::vertex_iterator(g)) {
                adjListsTest.push_back(vector<index_t>());

                for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                    adjListsTest[v].push_back(av);
                }
                REQUIRE(vectorSame(adjListsRef[v], adjListsTest[v]));
            }
        }

        SECTION("edge index iterator") {
            auto g = data<TestType>::g();

            vector<index_t> ref{0, 1, 2};
            vector<index_t> test;

            for (auto v: hg::edge_iterator(g)) {
                test.push_back(index(v, g)
                );
            }
            REQUIRE(vectorSame(ref, test));
        }

        SECTION("out edge index iterator") {
            auto g = data<TestType>::g();

            vector<vector<index_t>> ref{{0, 2},
                                        {0, 1},
                                        {1, 2},
                                        {}};
            vector<vector<index_t>> test;

            for (auto v: hg::vertex_iterator(g)) {
                test.push_back(vector<index_t>());

                for (auto av: hg::out_edge_iterator(v, g)) {
                    test[v].push_back(index(av, g)
                    );
                }
                REQUIRE(vectorSame(ref[v], test[v]));
            }
        }

        SECTION("in edge index iterator") {
            auto g = data<TestType>::g();

            vector<vector<index_t>> ref{{0, 2},
                                        {0, 1},
                                        {1, 2},
                                        {}};
            vector<vector<index_t>> test;

            for (auto v: hg::vertex_iterator(g)) {
                test.push_back(vector<index_t>());

                for (auto av: hg::in_edge_iterator(v, g)) {
                    test[v].push_back(index(av, g)
                    );
                }
                REQUIRE(vectorSame(ref[v], test[v]));
            }
        }

        SECTION("edge index") {
            auto g = data<TestType>::g();

            vector<pair<index_t, index_t>> eref{{0, 1},
                                                {1, 2},
                                                {0, 2}};
            vector<pair<index_t, index_t>> etest;
            for (auto e: hg::edge_iterator(g)) {
                etest.push_back(edge_from_index(index(e, g), g)
                );
            }

            REQUIRE(vectorSame(eref, etest));
        }

        SECTION("remove edge") {
            auto g = data<TestType>::g();

            remove_edge(1, g);

            vector<pair<index_t, index_t>> eref{{0,             1}, // deleted {1,2}
                                                {invalid_index, invalid_index},
                                                {0,             2}};
            vector<pair<index_t, index_t>> etest;
            for (auto e: hg::edge_iterator(g)) {
                etest.push_back(edge_from_index(e, g)
                );
            }

            REQUIRE(vectorSame(eref, etest));

            REQUIRE(degree(0, g) == 2);
            REQUIRE(degree(1, g) == 1);
            REQUIRE(degree(2, g) == 1);

            vector<vector<index_t>> adjListsRef{{1, 2},
                                                {0},
                                                {0},
                                                {}};
            vector<vector<index_t>> adjListsTest;

            for (auto v: hg::vertex_iterator(g)) {
                adjListsTest.push_back(vector<index_t>());

                for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                    adjListsTest[v].push_back(av);
                }
                REQUIRE(vectorSame(adjListsRef[v], adjListsTest[v]));
            }
        }

        SECTION("set edge") {
            auto g = data<TestType>::g();

            set_edge(1, 3, 0, g);

            vector<pair<index_t, index_t>> eref{{0, 1}, // deleted {1,2}
                                                {0, 3},
                                                {0, 2}};
            vector<pair<index_t, index_t>> etest;
            for (auto e: hg::edge_iterator(g)) {
                etest.push_back(e);
            }

            REQUIRE(vectorSame(eref, etest));

            REQUIRE(degree(0, g) == 3);
            REQUIRE(degree(1, g) == 1);
            REQUIRE(degree(2, g) == 1);
            REQUIRE(degree(3, g) == 1);

            vector<vector<index_t>> adjListsRef{{1, 2, 3},
                                                {0},
                                                {0},
                                                {0}};
            vector<vector<index_t>> adjListsTest;

            for (auto v: hg::vertex_iterator(g)) {
                adjListsTest.push_back(vector<index_t>());

                for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                    adjListsTest[v].push_back(av);
                }
                REQUIRE(vectorSame(adjListsRef[v], adjListsTest[v]));
            }
        }

        SECTION("adjacency matrix") {
            TestType g(5);
            add_edge(0, 1, g);
            add_edge(0, 2, g);
            add_edge(0, 3, g);
            add_edge(0, 4, g);
            add_edge(1, 2, g);
            add_edge(2, 3, g);
            add_edge(2, 4, g);

            array_1d<int> edge_weights{1, 2, 3, 4, 5, 6, 7};

            auto adj_mat = undirected_graph_2_adjacency_matrix(g, edge_weights, -1);

            array_2d<int> ref_adj_mat = {{-1, 1,  2,  3,  4},
                                         {1,  -1, 5,  -1, -1},
                                         {2,  5,  -1, 6,  7},
                                         {3,  -1, 6,  -1, -1},
                                         {4,  -1, 7,  -1, -1}};

            REQUIRE((ref_adj_mat == adj_mat));
            auto res = adjacency_matrix_2_undirected_graph(ref_adj_mat, -1);

            auto &g2 = res.first;
            auto &ew2 = res.second;

            REQUIRE((ew2 == edge_weights));
            REQUIRE(num_vertices(g) == num_vertices(g2));
            REQUIRE(num_edges(g) == num_edges(g2));
            auto it1 = edges(g);
            auto it2 = edges(g);
            for (auto i1 = it1.first, i2 = it2.first; i1 != it1.second; i1++, i2++) {
                REQUIRE(*i1 == *i2);
            }
        }

        SECTION("edge lists") {
            auto g = data<TestType>::g();

            array_1d<index_t> sources_ref{0, 1, 0};
            array_1d<index_t> targets_ref{1, 2, 2};

            auto src = sources(g);
            REQUIRE((sources_ref == src));

            auto tgt = targets(g);
            REQUIRE((targets_ref == tgt));

        }
    }

    TEMPLATE_TEST_CASE("repeated edge deletion and slot revival", "[undirected_graph]",
                       hg::ugraph, hg::undirected_graph<hg::hash_setS>) {
        TestType g(4, 4);
        g.add_edge(0, 1);
        g.add_edge(0, 1); // parallel, unrelated slot
        g.add_edge(2, 2); // self-loop
        std::vector<pair<index_t, index_t>> expected{{0, 1}, {0, 1}, {2, 2}};
        const array_2d<int> weights{{10, 11}, {20, 21}, {30, 31}};
        auto retained = g.endpoint_storage();
        auto check = [&]() {
            REQUIRE(g.num_edges() == expected.size());
            REQUIRE(g.endpoint_storage() == retained);
            index_t ei = 0;
            for (auto e: hg::edge_iterator(g)) {
                REQUIRE(e.index == ei);
                REQUIRE(e.source == expected[ei].first);
                REQUIRE(e.target == expected[ei].second);
                REQUIRE(g.sources()(ei) == e.source); // reacquire native adapters
                REQUIRE(g.targets()(ei) == e.target);
                REQUIRE(retained->edges[ei].source == e.source);
                REQUIRE(retained->edges[ei].target == e.target);
                REQUIRE(weights(e.index, 0) == (ei + 1) * 10);
                REQUIRE(weights(e.index, 1) == (ei + 1) * 10 + 1);
                ++ei;
            }
            for (index_t v = 0; v < 4; ++v) {
                vector<index_t> incident, adjacent, outgoing, incoming, actual_adjacent;
                for (index_t i = 0; i < static_cast<index_t>(expected.size()); ++i) {
                    auto [s, t] = expected[i];
                    if (s == v || t == v) {
                        incident.push_back(i);
                        adjacent.push_back(s == v ? t : s);
                    }
                }
                for (auto e: hg::out_edge_iterator(v, g)) {
                    REQUIRE(e.source == v);
                    outgoing.push_back(e.index);
                }
                for (auto e: hg::in_edge_iterator(v, g)) {
                    REQUIRE(e.target == v);
                    incoming.push_back(e.index);
                }
                for (auto a: hg::adjacent_vertex_iterator(v, g)) actual_adjacent.push_back(a);
                REQUIRE(vectorSame(incident, outgoing));
                REQUIRE(vectorSame(incident, incoming));
                REQUIRE(vectorSame(adjacent, actual_adjacent));
                REQUIRE(g.degree(v) == incident.size());
            }
        };
        check();
        for (index_t slot: {0, 2}) { // ordinary edge and self-loop
            for (int repeat = 0; repeat < 3; ++repeat) {
                hg::remove_edge(slot, g);
                expected[slot] = {invalid_index, invalid_index};
                check();
            }
            for (auto endpoints: vector<pair<index_t, index_t>>{{3, 1}, {3, 1}, {1, 1}, {1, 1}, {2, 0}}) {
                hg::set_edge(slot, endpoints.first, endpoints.second, g);
                expected[slot] = {std::min(endpoints.first, endpoints.second),
                                  std::max(endpoints.first, endpoints.second)};
                check();
            }
            g.remove_edge(slot);
            expected[slot] = {invalid_index, invalid_index};
            check();
        }
        // Native members and free adapters reject invalid input before changing topology.
        for (index_t slot: {-1, 3}) {
            REQUIRE_THROWS(g.remove_edge(slot));
            REQUIRE_THROWS(hg::remove_edge(slot, g));
            REQUIRE_THROWS(g.set_edge(slot, 0, 1));
            REQUIRE_THROWS(hg::set_edge(slot, 0, 1, g));
            check();
        }
        for (index_t slot: {0, 1, 2}) { // live and tombstoned slots
            for (index_t vertex: {-1, 4}) {
                REQUIRE_THROWS(g.set_edge(slot, vertex, 1));
                REQUIRE_THROWS(g.set_edge(slot, 1, vertex));
                REQUIRE_THROWS(hg::set_edge(slot, vertex, 1, g));
                REQUIRE_THROWS(hg::set_edge(slot, 1, vertex, g));
                check();
            }
        }
        REQUIRE(g.add_edge(3, 0).index == 3); // insertion never reuses a tombstone
        REQUIRE(g.num_edges() == 4);
        REQUIRE(g.edge_from_index(0).source == invalid_index);
        while (g.num_edges() <= retained->edges.capacity()) g.add_edge(0, 1);
        REQUIRE(g.endpoint_storage() != retained);
        g.set_edge(0, 3, 2);
        g.remove_edge(0);
        g.remove_edge(0);
        g.set_edge(0, 1, 1);
        REQUIRE(retained->edges[0].source == invalid_index);
        REQUIRE(retained->edges[0].target == invalid_index);
        REQUIRE(g.sources()(0) == 1);
        REQUIRE(g.targets()(0) == 1);
        TestType empty;
        REQUIRE_THROWS(empty.remove_edge(0));
        REQUIRE_THROWS(empty.set_edge(0, 0, 0));
    }

    TEMPLATE_TEST_CASE("endpoint storage generations", "[undirected_graph]",
                       hg::ugraph, hg::undirected_graph<hg::hash_setS>) {
        TestType g(4, 4, 2);
        REQUIRE(g.sources().size() == 0);
        REQUIRE(g.targets().size() == 0);
        g.add_edge(0, 1);
        // Native adapters remain borrowed and do not acquire storage ownership.
        {
            auto storage = g.endpoint_storage();
            const auto owners = storage.use_count();
            auto s = g.sources();
            auto t = g.targets();
            REQUIRE(storage.use_count() == owners);
            REQUIRE(s(0) == 0);
            REQUIRE(t(0) == 1);
            REQUIRE(s.data() == &storage->edges.front().source);
            REQUIRE(t.data() == &storage->edges.front().target);
            static_assert(std::is_const<std::remove_pointer_t<decltype(s.data())>>::value,
                          "Native endpoints must remain const");
        }

        auto first = g.endpoint_storage();
        const auto *allocation = first->edges.data();
        const auto capacity = first->edges.capacity();
        g.add_vertex();
        g.set_edge(0, 2, 3);
        REQUIRE(g.endpoint_storage() == first);
        REQUIRE(first->edges[0].source == 2);
        REQUIRE(first->edges[0].target == 3);
        while (g.num_edges() < capacity) {
            g.add_edge(0, 0); // no relocation despite the exported generation
        }
        REQUIRE(g.endpoint_storage() == first);
        REQUIRE(first->edges.data() == allocation);
        g.add_edge(0, 1);
        REQUIRE(g.endpoint_storage() != first);
        REQUIRE(first->edges.size() == capacity);
        REQUIRE(first->edges.data() == allocation);
        REQUIRE(g.endpoint_storage()->edges.capacity() >= capacity * 2);
        g.set_edge(0, 0, 1);
        REQUIRE(first->edges[0].source == 2);
        REQUIRE(first->edges[0].target == 3);

        std::vector<typename TestType::edge_storage_handle> generations{first};
        while (g.num_edges() < 2048) {
            auto retained = g.endpoint_storage();
            const auto size = retained->edges.size();
            const auto cap = retained->edges.capacity();
            while (g.num_edges() <= cap) {
                g.add_edge(1, 2);
            }
            REQUIRE(retained->edges.size() == cap);
            REQUIRE(retained->edges[size - 1].index == static_cast<index_t>(size - 1));
            generations.push_back(std::move(retained));
        }
        auto current = g.endpoint_storage();
        g.remove_edge(0);
        REQUIRE(current == g.endpoint_storage());
        REQUIRE(current->edges[0].source == invalid_index);
        REQUIRE(current->edges[0].target == invalid_index);
        REQUIRE(first->edges[0].source == 2);
        // Reacquire borrowed adapters after mutation; never read invalidated ones.
        REQUIRE(g.sources()(0) == invalid_index);
        REQUIRE(g.targets()(0) == invalid_index);
        REQUIRE(g.degree(0) == capacity); // remaining self-loops plus edge at capacity

        std::weak_ptr<const typename TestType::edge_storage> retired = first;
        first.reset();
        REQUIRE_FALSE(retired.expired());
        generations.clear();
        REQUIRE(retired.expired());

        // Without external owners, ordinary vector growth keeps the storage object.
        current.reset();
        std::weak_ptr<const typename TestType::edge_storage> unexported = g.endpoint_storage();
        auto before = unexported.lock().get();
        const auto cap = g.endpoint_storage()->edges.capacity();
        while (g.num_edges() <= cap) {
            g.add_edge(2, 3);
        }
        REQUIRE(g.endpoint_storage().get() == before);
        g = TestType();
        REQUIRE(unexported.expired());
    }

    TEMPLATE_TEST_CASE("endpoint storage graph value semantics", "[undirected_graph]",
                       hg::ugraph, hg::undirected_graph<hg::hash_setS>) {
        auto original = data<TestType>::g();
        auto exported = original.endpoint_storage();
        TestType copied(original);
        REQUIRE(copied.endpoint_storage() != exported);
        copied.set_edge(0, 2, 3);
        REQUIRE(original.sources()(0) == 0);
        REQUIRE(original.targets()(0) == 1);
        REQUIRE(original.degree(0) == 2);
        REQUIRE(copied.degree(0) == 1);
        copied.remove_edge(1);
        REQUIRE(original.targets()(1) == 2);
        auto generic_copy = hg::copy_graph<TestType>(original);
        generic_copy.set_edge(0, 1, 3);
        REQUIRE(original.targets()(0) == 1);

        TestType assigned(2);
        assigned.add_edge(0, 1);
        auto previous = assigned.endpoint_storage();
        assigned = original;
        REQUIRE(assigned.endpoint_storage() != exported);
        REQUIRE(previous->edges[0].target == 1);
        assigned.set_edge(0, 2, 3);
        REQUIRE(exported->edges[0].source == 0);
        auto self = assigned.endpoint_storage();
        assigned = assigned;
        REQUIRE(assigned.endpoint_storage() == self);
        assigned = std::move(assigned);
        REQUIRE(assigned.endpoint_storage() == self);

        TestType moved(std::move(original));
        REQUIRE(moved.endpoint_storage() == exported);
        REQUIRE(moved.num_vertices() == 4);
        REQUIRE(moved.degree(0) == 2);
        REQUIRE(original.num_vertices() == 0);
        REQUIRE(original.num_edges() == 0);
        REQUIRE(original.sources().size() == 0);
        REQUIRE(original.targets().size() == 0);
        original.add_vertices(2);
        original.add_edge(0, 1);
        REQUIRE(original.targets()(0) == 1);

        auto displaced = assigned.endpoint_storage();
        assigned = std::move(moved);
        REQUIRE(assigned.endpoint_storage() == exported);
        REQUIRE(assigned.degree(0) == 2);
        REQUIRE(displaced->edges[0].source == 2);
        REQUIRE(moved.num_vertices() == 0);
        moved.add_vertex();
        moved.add_edge(0, 0);
        REQUIRE(moved.targets()(0) == 0);
        std::weak_ptr<const typename TestType::edge_storage> lifetime = exported;
        assigned = TestType();
        REQUIRE_FALSE(lifetime.expired());
        exported.reset();
        REQUIRE(lifetime.expired());
    }

}