/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#define HG_DEBUG

#define CATCH_CONFIG_MAIN


#include "test_utils.hpp"

#include <array>
#include "higra/graph.hpp"
#include "higra/algo/graph_weights.hpp"
#include "higra/algo/tree.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/hierarchy/watershed_hierarchy.hpp"
#include "xtensor/xadapt.hpp"
using namespace hg;
using namespace xt;
using namespace std;

TEST_CASE("unused", "[experimental]") {


}


