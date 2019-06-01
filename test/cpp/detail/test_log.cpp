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
#include "higra/detail/log.hpp"
#include <string>

namespace test_log {

    using namespace hg;
    using namespace std;

    TEST_CASE("test logger", "[logger]") {
        string ref_msg("This is a test");
        auto & loggers = logger::callbacks();
        auto save = loggers.front();
        loggers.clear();
        SECTION( "do test" ) {
            loggers.push_back([&ref_msg](const std::string &msg) {
                REQUIRE(msg.find(ref_msg) != std::string::npos);
            });

            HG_LOG_INFO("%s", ref_msg.c_str());
        }
        loggers.clear();
        loggers.push_back(save);
    }
}