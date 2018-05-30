//
// Created by perretb on 29/05/18.
//

//
// Created by user on 4/16/18.
//

#include <boost/test/unit_test.hpp>
#include "higra/io/tree_io.hpp"

BOOST_AUTO_TEST_SUITE(treeIO);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(test_saveread) {

        array_1d<int> parent{5, 5, 6, 6, 6, 7, 7, 7};

        array_1d<double> attr1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        array_1d<int> attr2{8, 7, 6, 5, 4, 3, 2, 1};
        tree t(parent);
        ostringstream out;
        save_tree(out, t).add_attribute("attr1", attr1).add_attribute("attr2", attr2).finalize();
        string res = out.str();

        istringstream in(res);
        auto tree_attr = read_tree(in);
        auto t2 = tree_attr.first;
        auto attributes = tree_attr.second;

        BOOST_CHECK(xt::allclose(parent, t2.parents()));

        BOOST_CHECK(attributes.count("attr1") == 1);
        BOOST_CHECK(xt::allclose(attributes["attr1"], attr1));

        BOOST_CHECK(attributes.count("attr2") == 1);
        BOOST_CHECK(xt::allclose(attributes["attr2"], attr2));


    }

BOOST_AUTO_TEST_SUITE_END();