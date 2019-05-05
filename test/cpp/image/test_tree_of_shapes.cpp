/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/image/tree_of_shapes.hpp"
#include "../test_utils.hpp"

namespace tree_of_shapes {

    using namespace hg;
    using namespace std;

    TEST_CASE("test integer_level_multi_queue", "[tree_of_shapes]") {
        using qt = hg::tree_of_shapes_internal::integer_level_multi_queue<int, int>;

        qt q(-2, 7);

        SECTION("empty queue") {
            REQUIRE(q.size() == 0);
            REQUIRE(q.empty());
            REQUIRE(q.num_levels() == 10);
            REQUIRE(q.min_level() == -2);
            REQUIRE(q.max_level() == 7);
            for (int i = -2; i < 8; i++) {
                REQUIRE(q.level_empty(i));
            }
        }
        SECTION("push top pop") {
            q.push(1, 10);
            REQUIRE(!q.level_empty(1));
            REQUIRE(q.size() == 1);
            q.push(1, 7);
            REQUIRE(q.size() == 2);
            REQUIRE(q.top(1) == 7);
            q.pop(1);
            REQUIRE(q.size() == 1);
            REQUIRE(q.top(1) == 10);
            q.pop(1);
            REQUIRE(q.size() == 0);
            REQUIRE(q.level_empty(1));
        }
        SECTION("closest non empty") {
            q.push(0, 4);
            q.push(5, 7);
            std::vector<int> res{0, 0, 0, 0, 0, 5, 5, 5, 5, 5};
            for (int i = -2; i < 8; i++) {
                REQUIRE(q.find_closest_non_empty_level(i) == res[i + 2]);
            }
        }
    }

}
