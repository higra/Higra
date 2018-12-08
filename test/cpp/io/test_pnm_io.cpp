/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/
#include <boost/test/unit_test.hpp>
#include "higra/io/pnm_io.hpp"


BOOST_AUTO_TEST_SUITE(pnm_io);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(tokenizer) {
        string test = "#comment\n"
                      "2 abc\n"
                      "####\n"
                      "3\n"
                      "jkl\n"
                      "\n";
        std::stringstream in(test);
        pnm_io_internal::tokenizer tok(in);

        string ref[]{"2", "abc", "3", "jkl"};
        string s;
        int i = 0;
        while ((s = tok.next_token()) != "") {
            BOOST_CHECK(ref[i++] == s);
        }
        BOOST_CHECK(i == 4);
    }
    /*
    * Type	Magic number	Extension	Colors
    *               ASCII	Binary
    * Portable BitMap[1]	P1	P4	.pbm	0–1 (white & black)
    * Portable GrayMap[2]	P2	P5	.pgm	0–255 (gray scale)
    * Portable PixMap[3]	P3	P6	.ppm	0–255 (RGB)
    */
    BOOST_AUTO_TEST_CASE(read_P1) {
        string test = "P1\n"
                      "# w h\n"
                      "5 3\n"
                      "# data\n"
                      "0 1 1 0 1\n"
                      "1 1 0 1 1\n"
                      "1 0 1 1 1";
        std::stringstream in(test);
        auto res = pnm_io_internal::read_image_pnm(in);

        array_nd<unsigned char> ref = {{0, 1, 1, 0, 1},
                                       {1, 1, 0, 1, 1},
                                       {1, 0, 1, 1, 1}};
        BOOST_CHECK(res == ref);
    }

    BOOST_AUTO_TEST_CASE(read_P2) {
        string test = "P2\n"
                      "# w h\n"
                      "5 3 255\n"
                      "# data\n"
                      "0 10 1 0 1\n"
                      "1 1 0 255 1\n"
                      "1 0 1 1 1";
        std::stringstream in(test);
        auto res = pnm_io_internal::read_image_pnm(in);

        array_nd<unsigned char> ref = {{0, 10, 1, 0,   1},
                                       {1, 1,  0, 255, 1},
                                       {1, 0,  1, 1,   1}};
        BOOST_CHECK(res == ref);
    }

    BOOST_AUTO_TEST_CASE(read_P3) {
        string test = "P3\n"
                      "# w h\n"
                      "2 3 255\n"
                      "# data\n"
                      "0 10 1      0 1 25\n"
                      "1  1 0    255 1 12\n"
                      "1  0 1     34 1  1";
        std::stringstream in(test);
        auto res = pnm_io_internal::read_image_pnm(in);

        array_nd<unsigned char> ref = {{{0, 10, 1}, {0,   1, 25}},
                                       {{1, 1,  0}, {255, 1, 12}},
                                       {{1, 0,  1}, {34,  1, 1}}};
        BOOST_CHECK(res == ref);
    }

    BOOST_AUTO_TEST_CASE(read_P5) {
        string test = "P5\n"
                      "# w h\n"
                      "5 3 255\n"
                      "# data\n"
                      "aaaaabbbbbccccc";
        std::stringstream in(test);
        auto res = pnm_io_internal::read_image_pnm(in);
        union {
            char c;
            unsigned char uc;
        } converter;

        converter.c = 'a';
        unsigned char a = converter.uc;
        converter.c = 'b';
        unsigned char b = converter.uc;
        converter.c = 'c';
        unsigned char c = converter.uc;
        array_nd<unsigned char> ref = {{a, a, a, a, a},
                                       {b, b, b, b, b},
                                       {c, c, c, c, c}};
        BOOST_CHECK(res == ref);
    }

    BOOST_AUTO_TEST_CASE(read_P6) {
        string test = "P6\n"
                      "# w h\n"
                      "2 3 255\n"
                      "# data\n"
                      "abccbaaabbcabbccac";

        std::stringstream in(test);
        auto res = pnm_io_internal::read_image_pnm(in);

        union {
            char c;
            unsigned char uc;
        } converter;
        converter.c = 'a';
        unsigned char a = converter.uc;
        converter.c = 'b';
        unsigned char b = converter.uc;
        converter.c = 'c';
        unsigned char c = converter.uc;
        array_nd<unsigned char> ref = {{{a, b, c}, {c, b, a}},
                                       {{a, a, b}, {b, c, a}},
                                       {{b, b, c}, {c, a, c}}};
        BOOST_CHECK(res == ref);
    }

    BOOST_AUTO_TEST_CASE(save_1band) {
        array_nd<unsigned char> ref = {{0, 10, 1, 0,   1},
                                       {1, 1,  0, 255, 1},
                                       {1, 0,  1, 1,   1}};
        ostringstream out;
        pnm_io_internal::save_image_pnm(out, ref);
        string res = out.str();

        istringstream in(res);
        auto res2 = pnm_io_internal::read_image_pnm(in);
        BOOST_CHECK(res2 == ref);
    }

    BOOST_AUTO_TEST_CASE(save_3band) {
        array_nd<unsigned char> ref = {{{0, 10, 1}, {0,   1, 25}},
                                       {{1, 1,  0}, {255, 1, 12}},
                                       {{1, 0,  1}, {34,  1, 1}}};
        ostringstream out;
        pnm_io_internal::save_image_pnm(out, ref);
        string res = out.str();

        istringstream in(res);
        auto res2 = pnm_io_internal::read_image_pnm(in);
        BOOST_CHECK(res2 == ref);
    }

BOOST_AUTO_TEST_SUITE_END();