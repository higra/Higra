/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/
#include "../../utils.hpp"

#pragma once

namespace hg {

    /**
     * Facade to ease the declaration of new forward iterators (quick reimplementation of boost facade iterator).
     *
     * The iterator must inherit of the facade and declare the following 3 functions:
     *  - deference: obtain the element at the current position of the iterator
     *  - increment: move the iterator to next element
     *  - equal: test if two iterators are equal
     *
     * @tparam derived_type
     * @tparam value_t
     * @tparam reference_t
     */
    template<typename derived_type, typename value_t, typename reference_t = value_t>
    struct forward_iterator_facade {
        using self_type = forward_iterator_facade<derived_type, value_t, reference_t>;
        using value_type =  value_t;
        using difference_type = std::ptrdiff_t;
        using reference = reference_t;
        using pointer = value_t *;
        using iterator_category = std::forward_iterator_tag;

        friend reference operator*(const derived_type &lhs) {
            return lhs.dereference();
        }

        friend const reference operator*(derived_type &lhs) {
            return lhs.dereference();
        }

        friend derived_type operator++(derived_type &lhs) {
            lhs.increment();
            return lhs;
        }

        friend derived_type operator++(derived_type &lhs, int) {
            derived_type tmp(lhs);
            lhs.increment();
            return tmp;
        }

        friend bool operator==(const derived_type &lhs, const derived_type &rhs) {
            return lhs.equal(rhs);
        }

        friend bool operator!=(const derived_type &lhs, const derived_type &rhs) {
            return !lhs.equal(rhs);
        }
    };

    /**
     * Facade to ease the declaration of new random iterators (quick reimplementation of boost facade iterator).
     *
     * The iterator must inherit of the facade and declare the following 3 functions:
     *  - deference: obtain the element at the current position of the iterator
     *  - increment: move the iterator to next element
     *  - decrement: move the iterator to previous element
     *  - advance(n):	Advance by n positions
     *  - distance_to(j)	Measure the distance to j
     *  - equal: test if two iterators are equal
     *
     * @tparam derived_type
     * @tparam value_t
     * @tparam reference_t
     */
    template<typename derived_type, typename value_t, typename reference_t = value_t>
    struct random_iterator_facade {
        using self_type = random_iterator_facade<derived_type, value_t, reference_t>;
        using value_type =  value_t;
        using difference_type = std::ptrdiff_t;
        using reference = reference_t;
        using pointer = value_t *;
        using iterator_category = std::random_access_iterator_tag;

        friend reference operator*(const derived_type &lhs) {
            return lhs.dereference();
        }

        friend const reference operator*(derived_type &lhs) {
            return lhs.dereference();
        }

        friend derived_type operator++(derived_type &lhs) {
            lhs.increment();
            return lhs;
        }

        friend derived_type operator++(derived_type &lhs, int) {
            derived_type tmp(lhs);
            lhs.increment();
            return tmp;
        }

        friend derived_type operator--(derived_type &lhs) {
            lhs.decrement();
            return lhs;
        }

        friend derived_type operator--(derived_type &lhs, int) {
            derived_type tmp(lhs);
            lhs.decrement();
            return tmp;
        }

        friend bool operator==(const derived_type &lhs, const derived_type &rhs) {
            return lhs.equal(rhs);
        }

        friend bool operator!=(const derived_type &lhs, const derived_type &rhs) {
            return !lhs.equal(rhs);
        }

        friend bool operator<=(const derived_type &lhs, const derived_type &rhs) {
            return rhs.distance_to(lhs) <= 0;
        }

        friend bool operator<(const derived_type &lhs, const derived_type &rhs) {
            return rhs.distance_to(lhs) < 0;
        }

        friend bool operator>=(const derived_type &lhs, const derived_type &rhs) {
            return rhs.distance_to(lhs) >= 0;
        }

        friend bool operator>(const derived_type &lhs, const derived_type &rhs) {
            return rhs.distance_to(lhs) > 0;
        }

        friend auto operator-(const derived_type &lhs, const derived_type &rhs) {
            return rhs.distance_to(lhs);
        }

        friend derived_type operator+(const derived_type &lhs, const difference_type &n) {
            derived_type tmp(lhs);
            tmp.advance(n);
            return tmp;
        }

        friend derived_type operator+(const difference_type &n, const derived_type &rhs) {
            derived_type tmp(rhs);
            tmp.advance(n);
            return tmp;
        }

        auto operator[](const difference_type &n) {
            derived_type tmp(*(derived_type*)(this));
            tmp.advance(n);
            return *tmp;
        }
    };


    /**
     * Facade to ease the declaration of new forward iterators as a transformation of an existing forward iterator
     * (quick reimplementation of boost transform iterator).
     *
     * @tparam transform_fun_type
     * @tparam base_iterator_t
     * @tparam value_t
     * @tparam reference_t
     */
    template<typename transform_fun_type, typename base_iterator_t, typename value_t, typename reference_t = value_t>
    struct transform_forward_iterator :
            public forward_iterator_facade<transform_forward_iterator<transform_fun_type, base_iterator_t, value_t, reference_t>, value_t, reference_t> {

        using self_type = transform_forward_iterator<transform_fun_type, base_iterator_t, value_t, reference_t>;

        void increment() {
            m_base++;
        };

        bool equal(const self_type &other) const {
            return m_base == other.m_base;
        }

        value_t dereference() const {
            return m_fun(*m_base);
        }

        transform_forward_iterator(base_iterator_t &&base, transform_fun_type &&fun) :
                m_base(std::forward<base_iterator_t>(base)),
                m_fun(std::forward<transform_fun_type>(fun)) {}

        transform_forward_iterator(const base_iterator_t &base, const transform_fun_type &fun) :
                m_base(base),
                m_fun(fun) {}

        transform_forward_iterator() : m_base(), m_fun() {}

    private:
        base_iterator_t m_base;
        transform_fun_type m_fun;
    };


    /**
     * Quick implementation of a counting iterator.
     *
     * Counting iterator has an integer value and a step size.
     * Incrementing the iterator simply add the step size to the current value of the iterator.
     *
     * @tparam value_type
     */
    template<typename value_type = index_t>
    struct counting_iterator :
            public random_iterator_facade<counting_iterator<value_type>,
                    value_type> {
        using self_type = counting_iterator<value_type>;
        counting_iterator() : m_position(0), m_step(1) {}

        counting_iterator(value_type position, value_type step = 1) :
                m_position(position),
                m_step(step) {}

        void increment() {
            m_position += m_step;
        }

        void decrement() {
            m_position -= m_step;
        }

        void advance(size_t n) {
            m_position += n * m_step;
        }

        auto distance_to(const self_type& rhs) const {
            return (rhs.m_position - m_position) / m_step;
        }

        bool equal(const counting_iterator<value_type> &other) const {
            return m_position == other.m_position;
        }

        value_type dereference() const {
            return m_position;
        }

    private:
        value_type m_position;
        value_type m_step;
    };


    /**
     * A range of integer value with a start value, a step size, and a maximum value (excluded).
     *
     * irange(start, stop, step) is the equivalent to
     * - python range(start, stop, step)
     * - matlab start:step:stop
     *
     * @tparam value_type
     */
    template<typename value_type = index_t>
    struct irange {

        irange() : m_start(0), m_stop(0), m_step(1) {}

        irange(value_type start, value_type stop, value_type step = 1) :
                m_start(start),
                m_step(step) {
            auto num_steps = std::abs(m_start - stop) / std::abs(m_step);
            m_stop = m_start + num_steps * m_step;
        }

        auto begin() const {
            return counting_iterator<value_type>(m_start, m_step);
        }

        auto end() const {
            return counting_iterator<value_type>(m_stop, m_step);
        }

    private:

        value_type m_start;
        value_type m_step;
        value_type m_stop;

    };
}


