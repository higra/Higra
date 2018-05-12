//
// Created by user on 4/18/18.
//

#pragma once


namespace hg {

#define HG_ACCUMULATORS (min)(max)(mean)(counter)(sum)(prod)

    enum class accumulators {
        mean,
        min,
        max,
        counter,
        sum,
        prod
    };

    namespace accumulator_detail {

        /**
        * Marginal processing accumulator
        * @tparam S the storage type
        * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
        */
        template<typename S, bool vectorial = true>
        struct acc_marginal_impl {
            using value_type = typename S::value_type;
            using reducer_type = std::function<value_type(value_type, value_type)>;

            acc_marginal_impl(S &storage, reducer_type reducer, value_type init_value) :
                    m_init_value(init_value),
                    m_reducer(reducer),
                    m_storage(storage) {
            }

            template<typename ...Args>
            typename std::enable_if_t<vectorial>
            initialize(Args &&...) {
                std::fill(m_storage.begin(), m_storage.end(), m_init_value);
            }

            template<typename ...Args>
            typename std::enable_if_t<!vectorial>
            initialize(Args &&...) {
                *m_storage.begin() = m_init_value;
            }

            template<typename T, typename ...Args>
            std::enable_if_t<vectorial>
            accumulate(T &value, Args &&...) {
                for (auto s = m_storage.begin(), v = value.begin(); s != m_storage.end(); s++, v++) {
                    *s = m_reducer(*v, *s);
                }
            };

            template<typename T, typename ...Args>
            std::enable_if_t<!vectorial>
            accumulate(T &value, Args &&...) {
                *m_storage.begin() = m_reducer(*value.begin(), *m_storage.begin());
            };

            template<typename ...Args>
            void finalize(Args &&...) const {}

        private:

            value_type m_init_value;
            reducer_type m_reducer;
            S &m_storage;
        };


        /**
         * Mean accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_mean_impl {

            acc_mean_impl(S &storage) :
                    m_counter(0),
                    m_storage(storage) {
            }

            template<typename ...Args>
            typename std::enable_if_t<vectorial>
            initialize(Args &&...) {
                std::fill(m_storage.begin(), m_storage.end(), 0);
            }

            template<typename ...Args>
            typename std::enable_if_t<!vectorial>
            initialize(Args &&...) {
                *m_storage.begin() = 0;
            }

            template<typename T, typename ...Args>
            std::enable_if_t<vectorial>
            accumulate(T &value, Args &&...) {
                m_counter++;
                for (auto s = m_storage.begin(), v = value.begin(); s != m_storage.end(); s++, v++) {
                    *s += *v;
                }
            }

            template<typename T, typename ...Args>
            std::enable_if_t<!vectorial>
            accumulate(T &value, Args &&...) {
                m_counter++;
                *m_storage.begin() += *value.begin();
            }

            template<typename ...Args>
            typename std::enable_if_t<!vectorial>
            finalize(Args &&...) const {
                *m_storage.begin() /= m_counter;
            }

            template<typename ...Args>
            typename std::enable_if_t<vectorial>
            finalize(Args &&...) const {
                for (auto s = m_storage.begin(); s != m_storage.end(); s++) {
                    *s /= m_counter;
                }
            }

        private:
            std::size_t m_counter;
            S &m_storage;
        };

        /**
         * Counter accumulator
         * @tparam S the storage type
         * @tparam vectorial bool: is dimension of storage > 0 (different from scalar)
         */
        template<typename S, bool vectorial = true>
        struct acc_counter_impl {

            acc_counter_impl(S &storage) :
                    m_storage(storage) {
            }

            template<typename ...Args>
            void initialize(Args &&...) {
                *m_storage.begin() = 0;
            }

            template<typename T, typename ...Args>
            void accumulate(T &, Args &&...) {
                (*m_storage.begin())++;
            }

            template<typename ...Args>
            void finalize(Args &&...) const {}


        private:
            S &m_storage;
        };

    }

    struct accumulator_sum {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            return accumulator_detail::acc_marginal_impl<S, vectorial>(
                    storage,
                    std::plus<value_type>(),
                    0);
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_min {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            return accumulator_detail::acc_marginal_impl<S, vectorial>(
                    storage,
                    [](value_type v1, value_type v2) { return std::min(v1, v2); },
                    std::numeric_limits<value_type>::max());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_max {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            return accumulator_detail::acc_marginal_impl<S, vectorial>(
                    storage,
                    [](value_type v1, value_type v2) { return std::max(v1, v2); },
                    std::numeric_limits<value_type>::lowest());
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_prod {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            return accumulator_detail::acc_marginal_impl<S, vectorial>(
                    storage,
                    std::multiplies<value_type>(),
                    1);
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return input_shape;
        }
    };

    struct accumulator_mean {

        template<bool vectorial = true, typename S>
        auto make_accumulator(S &storage) const {
            using value_type = typename S::value_type;
            return accumulator_detail::acc_mean_impl<S, vectorial>(storage);
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
            using value_type = typename S::value_type;
            return accumulator_detail::acc_counter_impl<S, vectorial>(storage);
        }

        template<typename shape_t>
        static
        auto get_output_shape(const shape_t &input_shape) {
            return std::vector<std::size_t>();
        }
    };
}
