#include <boost/test/unit_test.hpp>

#include "embedding.hpp"


//
// Created by user on 3/9/18.
//




BOOST_AUTO_TEST_SUITE(TestSuitePoint);

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid1d) {
        hg::EmbeddingGrid<> e1 = {10};

        BOOST_CHECK(e1.isInBound({5}));
        BOOST_CHECK(!e1.isInBound({-2}));
        BOOST_CHECK(!e1.isInBound({12}));

        auto p1 = e1.lin2grid(2);
        std::vector<long> p2{2};

        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p2.begin()));

        std::vector<long> p3{15};
        BOOST_CHECK(e1.isInBound(p1));
        BOOST_CHECK(!e1.isInBound(p3));
    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid2d) {
        hg::EmbeddingGrid<> e1 = {10, 5};

        std::vector<long> p1 = {3, 0};
        auto p1t = e1.lin2grid(3);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1t) == 3);

        std::vector<long> p2 = {6, 2};
        auto p2t = e1.lin2grid(26);
        BOOST_CHECK(std::equal(p2.begin(), p2.end(), p2t.begin()));
        BOOST_CHECK(e1.grid2lin(p2) == 26);

        BOOST_CHECK(e1.isInBound(p1t));
        BOOST_CHECK(e1.isInBound(p2t));

        std::vector<long> p3 = {-1, 2};
        std::vector<long> p4 = {6, -1};
        std::vector<long> p5 = {10, 2};
        std::vector<long> p6 = {6, 5};
        BOOST_CHECK(!e1.isInBound(p3));
        BOOST_CHECK(!e1.isInBound(p4));
        BOOST_CHECK(!e1.isInBound(p5));
        BOOST_CHECK(!e1.isInBound(p6));

    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid3d) {
        hg::EmbeddingGrid<> e1 = {10, 5, 2};

        std::vector<long> p1 = {3, 2, 1};
        auto p1t = e1.lin2grid(73);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1) == 73);
    }


BOOST_AUTO_TEST_SUITE_END();
