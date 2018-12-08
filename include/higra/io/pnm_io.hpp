/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * TODO: support max value > 255 (muti-byte components)
 * TODO: support P3 format (raw binary data)
 */
#pragma once

#include "../utils.hpp"
#include "../structure/array.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

namespace hg {

    namespace pnm_io_internal {

        class tokenizer {

        public:
            inline
            tokenizer(std::istream &in) : m_in(in), m_stream("") {

            }

            inline
            std::string next_token() {
                if (m_stream.eof()) {
                    do {
                        if (m_in.eof())
                            return "";
                        std::getline(m_in, m_current_line);
                    } while (m_current_line.size() == 0 || m_current_line[0] == '#');
                    m_stream.clear();
                    m_stream.str(m_current_line);
                }
                std::string token;
                m_stream >> token;
                if (token == "") { // empty stream does not indicate eof on first read even if empty
                    return next_token();
                }
                return token;
            }

        private:
            std::istream &m_in;
            std::string m_current_line;
            std::stringstream m_stream;
        };

        inline
        array_nd<unsigned char> read_image_pnm(std::istream &in) {
            std::string tmp;
            bool ascii;
            bool binary;
            int max_value = 1;
            size_t bands;
            tokenizer tok(in);
            tmp = tok.next_token();
            hg_assert(tmp.size() == 2 && tmp[0] == 'P',
                      "Incorrect file format (magic number).");

            /*
             * Type	Magic number	Extension	Colors
             *               ASCII	Binary
             * Portable BitMap[1]	P1	P4	.pbm	0–1 (white & black)
             * Portable GrayMap[2]	P2	P5	.pgm	0–255 (gray scale)
             * Portable PixMap[3]	P3	P6	.ppm	0–255 (RGB)
             */
            switch (tmp[1]) {
                case '1':
                    ascii = true;
                    binary = true;
                    bands = 1;
                    break;
                case '2':
                    ascii = true;
                    binary = false;
                    bands = 1;
                    break;
                case '3':
                    ascii = true;
                    binary = false;
                    bands = 3;
                    break;
                case '4':
                    ascii = false;
                    binary = true;
                    bands = 1;
                    break;
                case '5':
                    ascii = false;
                    binary = false;
                    bands = 1;
                    break;
                case '6':
                    ascii = false;
                    binary = false;
                    bands = 3;
                    break;
                default:
                    hg_assert(false, "Unknown file format (magic number): " + tmp);
            }
            tmp = tok.next_token();
            hg_assert(tmp != "", "End of header reached too soon.");
            size_t width = std::stoi(tmp);

            tmp = tok.next_token();
            hg_assert(tmp != "", "End of header reached too soon.");
            size_t height = std::stoi(tmp);

            if (!binary) {
                tmp = tok.next_token();
                hg_assert(tmp != "", "End of header reached too soon.");
                max_value = std::stoi(tmp);
            }

            hg_assert(width > 0 && height > 0, "Incorrect dimensions.");

            hg_assert(max_value <= 255, "Multi-byte values not supported.");

            index_t num_values = width * height * bands;
            std::vector<size_t> shape{height, width};
            if (bands != 1) {
                shape.push_back(bands);
            }
            array_nd<unsigned char> result = xt::empty<unsigned char>(shape);
            auto r = xt::flatten(result);
            if (ascii) {
                for (index_t i = 0; i < num_values; i++) {
                    tmp = tok.next_token();
                    hg_assert(tmp != "", "End of data reached too soon.");
                    r(i) = std::stoi(tmp);
                }
            } else {
                //consume remaining comments
                while(in.peek() == '#'){
                    std::string tmp;
                    std::getline(in, tmp);
                }
                if (binary) {
                    hg_assert(false, "Binary raw data not supported.");
                } else {
                    in.read((char*)result.data(), num_values);
                }
            }
            return result;

        }

        template<typename T>
        void save_image_pnm(std::ostream &out, const xt::xexpression<T> &ximage) {
            static_assert(std::is_same<typename T::value_type, unsigned char>::value, "Can only save unsigned char values");
            auto &image = ximage.derived_cast();
            hg_assert(image.dimension() == 2 || image.dimension() == 3, "Array must have 2 or 3 dimensions.");
            hg_assert(image.dimension() == 2 || image.shape()[2] == 3,
                      "The size of the 3rd dimension must be 3 (RGB value).");

            // magic number
            if (image.dimension() == 3)
                out << "P6" << std::endl;
            else out << "P5" << std::endl;

            // size
            out << image.shape()[1] << " " << image.shape()[0] << std::endl;

            // max value
            out << "255" << std::endl;

            out.write((char*)image.data(), image.size());
        }

    }

    /**
     * Read the given pnm image (pbm, pgm or ppm formats).
     * Current the following pnm specification are supported
     * P1 binary ascii: supported
     * P2 byte ascii: supported (max value <= 255)
     * P3 RGB ascii: supported (max value <= 255)
     * P4 binary raw: NOT supported
     * P5 byte raw: supported (max value <= 255)
     * P6 RGB rax: supported (max value <= 255)
     *
     * @param filename Path to the file to read
     * @return an array with pixel data
     */
    inline
    array_nd<unsigned char> read_image_pnm(const char *filename) {
        std::ifstream s(filename);
        return pnm_io_internal::read_image_pnm(s);
    }

    /**
     * Save an array as a pnm file (pgm or ppm).
     * The array value_type MUST be unsigned char.
     * If the array has 2 dimensions it is saved as a pgm raw file (format P5).
     * If the array has 3 dimensions, the size of the third dimension must be 3
     * and it is saved as a ppm raw file (format P6).
     *
     * If the provided filename already exists, it will be overwritten !
     *
     * @tparam T xexpression derived type for ximage
     * @param filename Path to the file to write.
     * @param ximage The array containing the data to save.
     */
    template<typename T>
    void save_image_pnm(const char *filename, const xt::xexpression<T> &ximage) {
        std::ofstream s(filename);
        return pnm_io_internal::save_image_pnm(s, ximage);
    }
}