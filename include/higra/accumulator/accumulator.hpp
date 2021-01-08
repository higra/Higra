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

#include "../utils.hpp"
#include <functional>
#include <limits>
#include <vector>

namespace hg {

#define HG_ACCUMULATORS (min)(max)(mean)(counter)(sum)(prod)(first)(last)

    enum class accumulators {
        first,
        last,
        mean,
        min,
        max,
        counter,
        sum,
        prod,
        argmin,
        argmax
    };

    namespace accumulator_detail {

        /**
        * Marginal processing accumulator
        * @tparam S the storage type
        * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
        */
        template<typename S, typename initializer, typename operation, bool vectorial>
        struct acc_marginal_impl {
        };

        template<typename S, typename initializer, typename operation>
        struct acc_marginal_impl<S, initializer, operation, true> {

            static const bool is_vectorial = true;

            using self_type = acc_marginal_impl<S, initializer, operation, is_vectorial>;
            using value_type = typename std::iterator_traits<S>::value_type;

            acc_marginal_impl(const S &storage_begin, const S &storage_end) :
                    m_storage_begin(storage_begin),
                    m_storage_end(storage_end) {
            }

            template<typename ...Args>
            void
            initialize(Args &&...) {
                std::fill(m_storage_begin, m_storage_end, initializer::template init_value<value_type>());
            }

            template<typename T, typename ...Args>
            void
            accumulate(T value_begin, Args &&...) {
                auto s = m_storage_begin;
                for (; s != m_storage_end; s++, value_begin++) {
                    *s = operation::template reduce<value_type>(*value_begin, *s);
                }
            };

            template<typename ...Args>
            void finalize(Args &&...) const {}

            void
            set_storage(S &storage_begin, S &storage_end) {
                m_storage_begin = storage_begin;
                m_storage_end = storage_end;
            }

            template<typename T>
            void
            set_storage(T &range) {
                m_storage_begin = range.begin();
                m_storage_end = range.end();
            }

        private:
            S m_storage_begin;
            S m_storage_end;
        };

        template<typename S, typename initializer, typename operation>
        struct acc_marginal_impl<S, initializer, operation, false> {

            static const bool is_vectorial = false;

            using self_type = acc_marginal_impl<S, initializer, operation, is_vectorial>;
            using value_type = typename std::iterator_traits<S>::value_type;

            acc_marginal_impl(const S &storage_begin, const S &) :
                    m_storage_begin(storage_begin) {
            }

            template<typename ...Args>
            void
            initialize(Args &&...) {
                *m_storage_begin = initializer::template init_value<value_type>();
            }

            template<typename T, typename ...Args>
            void
            accumulate(const T value_begin, Args &&...) {
                *m_storage_begin = operation::template reduce<value_type>(*value_begin, *m_storage_begin);
            };

            template<typename ...Args>
            void finalize(Args &&...) const {}

            void
            set_storage(S &storage_begin, S &) {
                m_storage_begin = storage_begin;
            }

            template<typename T>
            void
            set_storage(T &range) {
                m_storage_begin = range.begin();
            }

        private:
            S m_storage_begin;
        };


        /**
         * Mean accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_mean_impl {

            using self_type = acc_mean_impl<S, vectorial>;
            using value_type = typename std::iterator_traits<S>::value_type;
            static const bool is_vectorial = vectorial;

            acc_mean_impl(S storage_begin, S storage_end) :
                    m_counter(0),
                    m_storage_begin(storage_begin),
                    m_storage_end(storage_end) {
            }

            template<typename T = self_type, typename ...Args>
            typename std::enable_if_t<T::is_vectorial>
            initialize(Args &&...) {
                std::fill(m_storage_begin, m_storage_end, 0);
            }

            template<typename T = self_type, typename ...Args>
            typename std::enable_if_t<!T::is_vectorial>
            initialize(Args &&...) {
                *m_storage_begin = 0;
            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<T1::is_vectorial>
            accumulate(T value_begin, Args &&...) {
                m_counter++;
                auto s = m_storage_begin;
                for (; s != m_storage_end; s++, value_begin++) {
                    *s += *value_begin;
                }
            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<!T1::is_vectorial>
            accumulate(const T value_begin, Args &&...) {
                m_counter++;
                *m_storage_begin += *value_begin;
            }

            template<typename T = self_type, typename ...Args>
            typename std::enable_if_t<T::is_vectorial>
            finalize(Args &&...) const {
                if (m_counter != 0) {
                    for (auto s = m_storage_begin; s != m_storage_end; s++) {
                        *s /= (value_type) m_counter;
                    }
                }
            }

            template<typename T = self_type, typename ...Args>
            typename std::enable_if_t<!T::is_vectorial>
            finalize(Args &&...) const {
                if (m_counter != 0) {
                    *m_storage_begin /= (value_type) m_counter;
                }
            }

            void set_storage(S storage_begin, S storage_end) {
                m_storage_begin = storage_begin;
                m_storage_end = storage_end;
            }

            template<typename T>
            void set_storage(T &range) {
                m_storage_begin = range.begin();
                m_storage_end = range.end();
            }

        private:
            std::size_t m_counter;
            S m_storage_begin;
            S m_storage_end;
        };

        /**
         * Counter accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_counter_impl {
            using value_type = typename std::iterator_traits<S>::value_type;
            using self_type = acc_counter_impl<S, vectorial>;
            static const bool is_vectorial = vectorial;

            acc_counter_impl(S storage_begin, S storage_end) :
                    m_storage_begin(storage_begin),
                    m_storage_end(storage_end) {
            }

            template<typename ...Args>
            void initialize(Args &&...) {
                *m_storage_begin = 0;
            }

            template<typename T, typename ...Args>
            void accumulate(const T &, Args &&...) {
                (*m_storage_begin)++;
            }

            template<typename ...Args>
            void finalize(Args &&...) const {}


            void set_storage(S storage_begin, S storage_end) {
                m_storage_begin = storage_begin;
                m_storage_end = storage_end;
            }

            template<typename T>
            void set_storage(T &range) {
                m_storage_begin = range.begin();
                m_storage_end = range.end();
            }

        private:
            S m_storage_begin;
            S m_storage_end;
        };

        /**
         * Argmin accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_argmin_impl {
            using value_type = hg::index_t;
            using self_type = acc_argmin_impl<S, vectorial>;
            static const bool is_vectorial = vectorial;
            using storage_value_type = typename std::iterator_traits<S>::value_type;


            acc_argmin_impl(S storage_begin, S storage_end) :
                    m_storage_begin(storage_begin),
                    m_storage_end(storage_end) {
                m_temp.resize(storage_end - storage_begin);
            }

            template<typename ...Args>
            void initialize(Args &&...) {
                *m_storage_begin = -1;
                cur_index = 0;
                std::fill(m_temp.begin(), m_temp.end(), (std::numeric_limits<storage_value_type>::max)());
            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<T1::is_vectorial>
            accumulate(T value_begin, Args &&...) {

                auto t = m_temp.begin();
                auto v = value_begin;
                bool flag = true;
                for (; flag && t != m_temp.end(); t++, v++) {
                    if (*t < *v)
                        flag = false;
                }
                if (flag) {
                    t = m_temp.begin();
                    v = value_begin;
                    for (; flag && t != m_temp.end(); t++, v++) {
                        *t = *v;
                    }
                    *m_storage_begin = cur_index;
                }

                cur_index++;
            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<!T1::is_vectorial>
            accumulate(T value_begin, Args &&...) {

                if (*value_begin < *m_temp.begin()) {
                    *m_temp.begin() = *value_begin;
                    *m_storage_begin = cur_index;
                }

                cur_index++;
            }

            template<typename ...Args>
            void finalize(Args &&...) const {}


            void set_storage(S storage_begin, S storage_end) {
                m_storage_begin = storage_begin;
                m_storage_end = storage_end;
            }

            template<typename T>
            void set_storage(T &range) {
                m_storage_begin = range.begin();
                m_storage_end = range.end();
            }

        private:
            std::vector<storage_value_type> m_temp;
            index_t cur_index;
            S m_storage_begin;
            S m_storage_end;
        };


        /**
         * Argmax accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_argmax_impl {
            using value_type = hg::index_t;
            using self_type = acc_argmin_impl<S, vectorial>;
            static const bool is_vectorial = vectorial;
            using storage_value_type = typename std::iterator_traits<S>::value_type;


            acc_argmax_impl(S storage_begin, S storage_end) :
                    m_storage_begin(storage_begin),
                    m_storage_end(storage_end) {
                m_temp.resize(storage_end - storage_begin);
            }

            template<typename ...Args>
            void initialize(Args &&...) {
                *m_storage_begin = -1;
                cur_index = 0;
                std::fill(m_temp.begin(), m_temp.end(), std::numeric_limits<storage_value_type>::lowest());
            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<T1::is_vectorial>
            accumulate(T value_begin, Args &&...) {

                auto t = m_temp.begin();
                auto v = value_begin;
                bool flag = true;
                for (; flag && t != m_temp.end(); t++, v++) {
                    if (*t > *v)
                        flag = false;
                }
                if (flag) {
                    t = m_temp.begin();
                    v = value_begin;
                    for (; flag && t != m_temp.end(); t++, v++) {
                        *t = *v;
                    }
                    *m_storage_begin = cur_index;
                }

                cur_index++;
            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<!T1::is_vectorial>
            accumulate(T value_begin, Args &&...) {

                if (*value_begin > *m_temp.begin()) {
                    *m_temp.begin() = *value_begin;
                    *m_storage_begin = cur_index;
                }

                cur_index++;
            }

            template<typename ...Args>
            void finalize(Args &&...) const {}


            void set_storage(S storage_begin, S storage_end) {
                m_storage_begin = storage_begin;
                m_storage_end = storage_end;
            }

            template<typename T>
            void set_storage(T &range) {
                m_storage_begin = range.begin();
                m_storage_end = range.end();
            }

        private:
            std::vector<storage_value_type> m_temp;
            index_t cur_index;
            S m_storage_begin;
            S m_storage_end;
        };


        /**
         * First accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_first_impl {
            using value_type = typename std::iterator_traits<S>::value_type;
            using self_type = acc_first_impl<S, vectorial>;
            static const bool is_vectorial = vectorial;

            acc_first_impl(S storage_begin, S storage_end) :
                    m_storage_begin(storage_begin),
                    m_storage_end(storage_end) {
            }

            template<typename ...Args>
            void initialize(Args &&...) {
                m_first = true;
            }

            template<typename T, typename ...Args>
            void accumulate(T value_begin, Args &&...) {
                if (m_first) {
                    m_first = false;
                    auto s = m_storage_begin;
                    for (; s != m_storage_end; s++, value_begin++) {
                        *s = *value_begin;
                    }
                }
            }

            template<typename ...Args>
            void finalize(Args &&...) const {}


            void set_storage(S storage_begin, S storage_end) {
                m_storage_begin = storage_begin;
                m_storage_end = storage_end;
            }

            template<typename T>
            void set_storage(T &range) {
                m_storage_begin = range.begin();
                m_storage_end = range.end();
            }

        private:
            bool m_first;
            S m_storage_begin;
            S m_storage_end;
        };

        /**
         * Last accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_last_impl {
            using value_type = typename std::iterator_traits<S>::value_type;
            using self_type = acc_last_impl<S, vectorial>;
            static const bool is_vectorial = vectorial;

            acc_last_impl(S storage_begin, S storage_end) :
                    m_storage_begin(storage_begin),
                    m_storage_end(storage_end) {
            }

            template<typename ...Args>
            void initialize(Args &&...) {

            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<T1::is_vectorial>
            accumulate(T value_begin, Args &&...) {
                auto s = m_storage_begin;
                for (; s != m_storage_end; s++, value_begin++) {
                    *s = *value_begin;
                }
            }

            template<typename T1 = self_type, typename T, typename ...Args>
            std::enable_if_t<!T1::is_vectorial>
            accumulate(const T value_begin, Args &&...) {
                *m_storage_begin = *value_begin;
            }

            template<typename ...Args>
            void finalize(Args &&...) const {}


            void set_storage(S storage_begin, S storage_end) {
                m_storage_begin = storage_begin;
                m_storage_end = storage_end;
            }

            template<typename T>
            void set_storage(T &range) {
                m_storage_begin = range.begin();
                m_storage_end = range.end();
            }

        private:
            S m_storage_begin;
            S m_storage_end;
        };

    }

    struct accumulator_sum {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_marginal_impl<iterator_type, accumulator_sum, accumulator_sum, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }

        template <typename value_type>
        constexpr static value_type init_value(){
            return 0;
        }

        template <typename value_type>
        static value_type reduce(const value_type & v1, const value_type & v2){
            return v1 + v2;
        }
    };

    struct accumulator_min {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_marginal_impl<iterator_type, accumulator_min, accumulator_min, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }

        template <typename value_type>
        static value_type init_value(){
            return (std::numeric_limits<value_type>::max)();
        }

        template <typename value_type>
        static value_type reduce(const value_type & v1, const value_type & v2){
            return (std::min)(v1, v2);
        }
    };

    struct accumulator_max {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_marginal_impl<iterator_type, accumulator_max, accumulator_max, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }

        template <typename value_type>
        static value_type init_value(){
            return std::numeric_limits<value_type>::lowest();
        }

        template <typename value_type>
        static value_type reduce(const value_type & v1, const value_type & v2){
            return (std::max)(v1, v2);
        }
    };

    struct accumulator_prod {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_marginal_impl<iterator_type, accumulator_prod, accumulator_prod, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }

        template <typename value_type>
        static value_type init_value(){
            return 1;
        }

        template <typename value_type>
        static value_type reduce(const value_type & v1, const value_type & v2){
            return v1 * v2;
        }
    };

    struct accumulator_mean {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_mean_impl<iterator_type, vectorial>(storage.begin(), storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_counter {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = index_t;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_counter_impl<iterator_type, vectorial>(storage.begin(), storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &) {
            return std::vector<std::size_t>();
        }
    };

    struct accumulator_first {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_first_impl<iterator_type, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_last {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_last_impl<iterator_type, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_argmin {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = index_t;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_argmin_impl<iterator_type, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_argmax {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = index_t;
            using iterator_type = decltype(storage.begin());
            return accumulator_detail::acc_argmax_impl<iterator_type, vectorial>(
                    storage.begin(),
                    storage.end());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };
}
