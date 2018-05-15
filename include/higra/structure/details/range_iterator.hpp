//
// Created by perretb on 15/05/18.
//

#pragma once

#include <boost/iterator/iterator_facade.hpp>

namespace hg {

    template<typename value_type = long>
    struct counting_iterator :
            public boost::iterator_facade<counting_iterator<value_type>,
                    value_type,
                    boost::bidirectional_traversal_tag,
                    value_type> {

        counting_iterator() : m_position(0), m_step(1) {}

        counting_iterator(value_type position, value_type step = 1) :
                m_position(position),
                m_step(step) {}

    private:
        friend class boost::iterator_core_access;

        value_type m_position;
        value_type m_step;

        void increment() {
            m_position += m_step;
        }

        void decrement() {
            m_position -= m_step;
        }

        bool equal(const counting_iterator<value_type> &other) const {
            return m_position == other.m_position;
        }

        value_type dereference() const {
            return m_position;
        }
    };

    template<typename value_type = long>
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
        friend class boost::iterator_core_access;

        value_type m_start;
        value_type m_step;
        value_type m_stop;

    };
}


