/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/structure/strided_span.hpp"
#include "higra/graph.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/structure/lca_fast.hpp"
#include "higra/algo/bipartite_graph.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

namespace test_strided_span {

    using namespace hg;

    struct record {
        index_t a;
        index_t b;
        index_t c;
    };

    using span_t = strided_span<index_t>;
    using cspan_t = strided_span<const index_t>;

    static_assert(std::is_trivially_copyable<cspan_t>::value, "spans are trivially copyable");
    static_assert(std::is_same<std::iterator_traits<span_t::iterator>::iterator_category,
            std::random_access_iterator_tag>::value, "random access iterator");
    static_assert(std::is_convertible<span_t, cspan_t>::value, "span<T> converts to span<const T>");
    static_assert(!std::is_convertible<cspan_t, span_t>::value, "span<const T> does not convert to span<T>");

    // 1d indexable concept
    static_assert(is_1d_indexable_v<cspan_t>, "");
    static_assert(is_1d_indexable_v<array_1d<int>>, "");
    static_assert(is_1d_indexable_v<array_view<const int>>, "");
    static_assert(is_1d_indexable_v<decltype(hg::arange<index_t>(3))>, "");
    static_assert(!is_1d_indexable_v<index_t>, "");
    static_assert(!is_1d_indexable_v<std::vector<index_t>>, "");
    static_assert(std::is_same<indexable_1d_value_t<cspan_t>, index_t>::value, "");
    static_assert(std::is_same<indexable_1d_value_t<array_1d<int>>, int>::value, "");

    TEST_CASE("strided span over a structure member", "[strided_span]") {
        std::vector<record> records{{0, 10, 20},
                                    {1, 11, 21},
                                    {2, 12, 22},
                                    {3, 13, 23}};

        auto bs = member_span(records.data(), &record::b, (index_t) records.size());
        static_assert(std::is_same<decltype(bs), span_t>::value, "mutable structure gives a mutable span");
        REQUIRE(bs.size() == 4);
        REQUIRE(bs.stride() == 3);
        REQUIRE(bs.data() == &records[0].b);
        REQUIRE(!bs.empty());
        for (index_t i = 0; i < 4; ++i) {
            REQUIRE(bs(i) == 10 + i);
            REQUIRE(bs[i] == 10 + i);
        }

        // writes go to the structures
        bs(2) = -1;
        REQUIRE(records[2].b == -1);
        bs[2] = 12;

        const record *crecords = records.data();
        auto cs = member_span(crecords, &record::c, (index_t) records.size());
        static_assert(std::is_same<decltype(cs), cspan_t>::value, "const structure gives a const span");
        REQUIRE(cs(3) == 23);

        cspan_t converted = bs;
        REQUIRE(converted.data() == bs.data());
        REQUIRE(converted.stride() == bs.stride());
        REQUIRE(converted.size() == bs.size());
    }

    TEST_CASE("strided span iterators", "[strided_span]") {
        std::vector<record> records{{0, 13, 20},
                                    {1, 10, 21},
                                    {2, 12, 22},
                                    {3, 11, 23}};
        auto bs = member_span(records.data(), &record::b, (index_t) records.size());

        std::vector<index_t> values(bs.begin(), bs.end());
        REQUIRE((values == std::vector<index_t>{13, 10, 12, 11}));
        REQUIRE(std::distance(bs.begin(), bs.end()) == 4);
        REQUIRE(bs.end() - bs.begin() == 4);
        REQUIRE(*(bs.begin() + 2) == 12);
        REQUIRE(*(2 + bs.begin()) == 12);
        REQUIRE(*(bs.end() - 1) == 11);
        REQUIRE(bs.begin()[3] == 11);
        REQUIRE(bs.begin() < bs.end());
        REQUIRE(bs.end() > bs.begin());
        REQUIRE(bs.begin() <= bs.begin());
        REQUIRE(bs.end() >= bs.begin());
        auto it = bs.end();
        --it;
        it--;
        REQUIRE(*it == 12);
        it -= 2;
        REQUIRE(it == bs.begin());
        it += 1;
        REQUIRE(*it++ == 10);
        REQUIRE(*it == 12);
        REQUIRE(std::accumulate(bs.cbegin(), bs.cend(), (index_t) 0) == 46);

        // standard algorithms modify the structures through the span
        std::sort(bs.begin(), bs.end());
        REQUIRE(records[0].b == 10);
        REQUIRE(records[1].b == 11);
        REQUIRE(records[2].b == 12);
        REQUIRE(records[3].b == 13);
        REQUIRE(records[0].a == 0);
        REQUIRE(records[3].c == 23);
    }

    TEST_CASE("strided span edge cases", "[strided_span]") {
        cspan_t empty;
        REQUIRE(empty.size() == 0);
        REQUIRE(empty.empty());
        REQUIRE(empty.begin() == empty.end());

        std::vector<record> none;
        auto s = member_span(none.data(), &record::a, 0);
        REQUIRE(s.size() == 0);
        REQUIRE(s.begin() == s.end());

        // negative stride: reversed sequence
        index_t data[] = {0, 1, 2, 3, 4, 5};
        cspan_t reversed(data + 5, -2, 3);
        std::vector<index_t> values(reversed.begin(), reversed.end());
        REQUIRE((values == std::vector<index_t>{5, 3, 1}));
        REQUIRE(reversed.begin() < reversed.end());
        REQUIRE(reversed.end() - reversed.begin() == 3);
    }

    TEST_CASE("undirected graph endpoints are strided spans", "[strided_span]") {
        ugraph g(4);
        add_edge(0, 1, g);
        add_edge(1, 2, g);
        add_edge(0, 3, g);
        auto s = sources(g);
        auto t = targets(g);
        static_assert(std::is_same<decltype(s), cspan_t>::value, "");
        REQUIRE(s.size() == 3);
        REQUIRE(t.size() == 3);
        REQUIRE((std::vector<index_t>(s.begin(), s.end()) == std::vector<index_t>{0, 1, 0}));
        REQUIRE((std::vector<index_t>(t.begin(), t.end()) == std::vector<index_t>{1, 2, 3}));

        // tree endpoints: materialized sources, contiguous view of the parents
        tree tr(array_1d<index_t>{3, 3, 4, 4, 4});
        array_1d<index_t> ts = sources(tr);
        REQUIRE((ts == array_1d<index_t>{0, 1, 2, 3}));
        auto tt = targets(tr);
        REQUIRE((tt == array_1d<index_t>{3, 3, 4, 4}));
        REQUIRE(&tt(0) == &parents(tr)(0));
    }

    TEST_CASE("endpoint consumers accept spans and arrays", "[strided_span]") {
        ugraph g(4);
        add_edge(0, 1, g);
        add_edge(1, 2, g);
        add_edge(2, 3, g);
        add_edge(0, 3, g);
        array_1d<index_t> sorted_edges{3, 0, 2, 1};

        array_1d<index_t> src = {0, 1, 2, 0};
        array_1d<index_t> tgt = {1, 2, 3, 3};

        // add_edges: from spans of another graph, from arrays and from contiguous views
        ugraph g2(4);
        add_edges(sources(g), targets(g), g2);
        ugraph g3(4);
        array_view<const index_t> vsrc = src;
        array_view<const index_t> vtgt = tgt;
        add_edges(vsrc, vtgt, g3);
        for (index_t i = 0; i < 4; ++i) {
            REQUIRE(source(edge_from_index(i, g2), g2) == src(i));
            REQUIRE(target(edge_from_index(i, g2), g2) == tgt(i));
            REQUIRE(source(edge_from_index(i, g3), g3) == src(i));
            REQUIRE(target(edge_from_index(i, g3), g3) == tgt(i));
        }
        REQUIRE_THROWS(add_edges(src, array_1d<index_t>{1, 2}, g3));
        REQUIRE_THROWS(add_edges(src, hg::array<index_t>{{1, 2}, {3, 3}}, g3));

        // bpt: strided spans (graph path) and arrays (Python path) give the same result
        auto r1 = hierarchy_core_internal::bpt_canonical_from_sorted_edges(sources(g), targets(g), sorted_edges, 4);
        auto r2 = hierarchy_core_internal::bpt_canonical_from_sorted_edges(src, tgt, sorted_edges, 4);
        REQUIRE((r1.first == r2.first));
        REQUIRE((r1.second == r2.second));
        REQUIRE((r1.second == array_1d<index_t>{3, 0, 2}));

        // lca of the edges
        tree t(r1.first);
        lca_fast lca(t);
        auto l1 = lca.lca(sources(g), targets(g));
        auto l2 = lca.lca(src, tgt);
        REQUIRE((l1 == l2));
        REQUIRE(lca.lca(index_t(0), index_t(3)) == 4);

        // bipartite
        auto b1 = is_bipartite_graph(sources(g), targets(g), 4);
        auto b2 = is_bipartite_graph(src, tgt, 4);
        REQUIRE(b1.first);
        REQUIRE(b2.first);
        REQUIRE((b1.second == b2.second));

        // CSA (bipartite matching) on a graph: strided spans
        ugraph gb(4);
        add_edge(0, 2, gb);
        add_edge(0, 3, gb);
        add_edge(1, 2, gb);
        add_edge(1, 3, gb);
        array_1d<index_t> wb{1, 5, 5, 1};
        auto matching = bipartite_graph_matching(gb, wb);
        std::sort(matching.begin(), matching.end());
        REQUIRE((matching == array_1d<index_t>{0, 3}));
    }
}
