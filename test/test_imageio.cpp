
#include <boost/test/unit_test.hpp>

#include "io/image_io.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"


using namespace std;

/**
 * opencv io tests
 */


BOOST_AUTO_TEST_SUITE(imageio_opencv);

    xt::xarray<uchar> referenceImage = {{{255,255,255},{127,127,127},{0,0,0}},{{255,0,0},{0,255,0},{0,0,255}}};


    BOOST_AUTO_TEST_CASE(readpng)
    {
        auto image = hg::imageRead<uchar>("../../resources/test/test.png");
        BOOST_CHECK(image == referenceImage);
    }

    BOOST_AUTO_TEST_CASE(readbmp)
    {
        auto image = hg::imageRead<uchar>("../../resources/test/test.bmp");
        BOOST_CHECK(image == referenceImage);
    }

    BOOST_AUTO_TEST_CASE(tiff)
    {
        auto image = hg::imageRead<uchar>("../../resources/test/test.tif");
        BOOST_CHECK(image == referenceImage);
    }


    BOOST_AUTO_TEST_CASE(readFail)
    {
        BOOST_REQUIRE_THROW(hg::imageRead<uchar>("iDOnTExist.png"), std::runtime_error);
    }





BOOST_AUTO_TEST_SUITE_END();






