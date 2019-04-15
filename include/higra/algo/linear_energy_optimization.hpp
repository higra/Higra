/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include <deque>
#include <limits>
#include <algorithm>

namespace hg {

    namespace linear_energy_optimization_internal {

        /**
         * One piece of a piecewise energy linear function.
         * A piece represents the line with the given slope and passing through the point (origin_x, origin_y).
         */
        template<typename value_type=double>
        class piecewise_linear_energy_function_piece {
        public:

            piecewise_linear_energy_function_piece(value_type origin_x, value_type origin_y, value_type slope) :
                    m_origin_x(origin_x), m_origin_y(origin_y), m_slope(slope) {}

            double operator()(value_type x) const {
                return m_origin_y + m_slope * (x - m_origin_x);
            }

            value_type &origin_x() {
                return m_origin_x;
            }

            const value_type &origin_x() const {
                return m_origin_x;
            }

            value_type &origin_y() {
                return m_origin_y;
            }

            const value_type &origin_y() const {
                return m_origin_y;
            }

            value_type &slope() {
                return m_slope;
            }

            const value_type &slope() const {
                return m_slope;
            }

            bool
            operator==(const piecewise_linear_energy_function_piece<value_type> &rhs) const {
                return almost_equal(m_origin_x, rhs.m_origin_x) &&
                       almost_equal(m_origin_y, rhs.m_origin_y) &&
                       almost_equal(m_slope, rhs.m_slope);
            }


            bool
            operator!=(const piecewise_linear_energy_function_piece<value_type> &rhs) const {
                return !(*this == rhs);
            }


        private:

            static
            bool almost_equal(value_type a, value_type b) {
                double epsilon = 1e-5;
                return fabs(a - b) < epsilon;
            }

            value_type m_origin_x;
            value_type m_origin_y;
            value_type m_slope;
        };

        /**
         * Piecewise linear energy function as modelled in:
         * 
         *  Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men. Scale-sets Image Analysis. International
         *  Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317
         *  
         * An energy function is a concave non decreasing piecewise linear positive function.
         */
        template<typename value_type=double>
        class piecewise_linear_energy_function {

        public:
            using lp_t = piecewise_linear_energy_function_piece<value_type>;
            using self_type = piecewise_linear_energy_function<value_type>;

            piecewise_linear_energy_function() {};

            piecewise_linear_energy_function(const lp_t &piece) {
                pieces.push_back(piece);
            }

            piecewise_linear_energy_function(lp_t &&piece) {
                pieces.push_back(std::forward<lp_t>(piece));
            }

            piecewise_linear_energy_function(std::initializer_list<lp_t> pieces_l) : pieces(pieces_l) {}

            /**
             * Computes the sum between two piecewise_linear_energy_function
             * The computation is by default limited to the max_pieces largest pieces (right most)
             * 
             * @param other 
             * @return 
             */
            self_type sum(const self_type &other, int max_pieces = 10) const {

                if (other.size() == 0) {
                    return self_type(*this);
                } else if (size() == 0) {
                    return self_type(other);
                }

                self_type result = self_type();
                int count = 0;
                int i1 = pieces.size() - 1;
                int i2 = other.pieces.size() - 1;
                while (i1 >= 0 && i2 >= 0 && count < max_pieces) {
                    const auto &piece1 = pieces[i1];
                    const auto &piece2 = other.pieces[i2];
                    auto new_slope = piece1.slope() + piece2.slope();
                    value_type new_origin_x, new_origin_y;
                    if (piece1.origin_x() >= piece2.origin_x()) {
                        new_origin_x = piece1.origin_x();
                        new_origin_y = piece1.origin_y() + piece2(piece1.origin_x());
                        if (piece1.origin_x() == piece2.origin_x()) {
                            i2--;
                        }
                        i1--;
                    } else {
                        new_origin_x = piece2.origin_x();
                        new_origin_y = piece2.origin_y() + piece1(piece2.origin_x());
                        i2--;
                    }

                    result.pieces.push_front(
                            lp_t(new_origin_x, new_origin_y, new_slope));
                    count++;
                }

                if (result.pieces.size() > 0) {
                    auto &first_piece = result.pieces[0];
                    if (first_piece.origin_x() > 0) {
                        first_piece.origin_y() -= first_piece.slope() * first_piece.origin_x();
                        first_piece.origin_x() = 0;
                    }
                }
                return result;
            }

            /**
             * Infimum between the current piecewise linear energy function and the given linear piece.
             *
             * Returns the abscissa of the intersection between the two functions and infinity if no intersection exists
             *
             * PRECONDITION :
             *   this->size() > 0
             *   (*this)[0].origin_x() == 0
             *   linear_piece.origin_x() == 0
             *   (*this)[this->size() - 1].slope() >= linear_piece.slope()
             *
             *
             * Warning: Modification is done in place
             */
            double infimum(const lp_t &linear_piece) {
                int i = pieces.size() - 1;

                auto &last_piece = pieces[i];
                if (linear_piece.slope() == last_piece.slope()) {
                    auto y = linear_piece(last_piece.origin_x());
                    if (y > last_piece.origin_y()) {
                        return std::numeric_limits<value_type>::infinity();
                    } else if (y == last_piece.origin_y()) {
                        return last_piece.origin_x();
                    } else {
                        pieces.pop_back();
                        i--;
                    }
                }

                value_type xi = std::numeric_limits<double>::infinity();
                bool flag = true;
                while (i >= 0 && flag) {
                    auto &piece = pieces[i];
                    xi = (linear_piece.origin_x() * linear_piece.slope() - piece.origin_x() * piece.slope() -
                          (linear_piece.origin_y() - piece.origin_y())) / (linear_piece.slope() - piece.slope());
                    if (xi > piece.origin_x()) {
                        flag = false;
                    } else {
                        pieces.pop_back();
                    }
                    i--;
                }
                pieces.push_back(lp_t(xi, linear_piece(xi), linear_piece.slope()));
                return xi;
            }

            bool
            operator==(const self_type &rhs) const {
                if (pieces.size() != rhs.pieces.size())
                    return false;
                return std::equal(pieces.begin(), pieces.end(), rhs.pieces.begin());
            }


            bool
            operator!=(const self_type &rhs) const {
                return !(*this == rhs);
            }

            size_t size() const {
                return pieces.size();
            }

            const auto begin() const {
                return pieces.cbegin();
            }

            const auto end() const {
                return pieces.cend();
            }

            auto begin() {
                return pieces.begin();
            }

            auto end() {
                return pieces.end();
            }

            const auto operator[](index_t i) const {
                return pieces[i];
            }

            auto operator[](index_t i) {
                return pieces[i];
            }

        private:
            std::deque<lp_t> pieces;
        };
    }

}