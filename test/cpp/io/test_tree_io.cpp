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
#include "higra/io/tree_io.hpp"

namespace tree_io {

    using namespace hg;
    using namespace std;


    TEST_CASE("read and save tree", "[tree_io]") {
            array_1d<int> parent{ 5, 5, 6, 6, 6, 7, 7, 7 };

            array_1d<double> attr1{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            array_1d<int> attr2{ 8, 7, 6, 5, 4, 3, 2, 1 };
            tree t(parent);
            ostringstream out;
            save_tree(out, t).add_attribute("attr1", attr1).add_attribute("attr2", attr2).finalize();
            string res = out.str();

            istringstream in(res);
            auto tree_attr = read_tree(in);
            auto t2 = tree_attr.first;
            auto attributes = tree_attr.second;

            REQUIRE(xt::allclose(parent, parents(t2)));

            REQUIRE(attributes.count("attr1") == 1);
            REQUIRE(xt::allclose(attributes["attr1"], attr1));

            REQUIRE(attributes.count("attr2") == 1);
            REQUIRE(xt::allclose(attributes["attr2"], attr2));
    }
}