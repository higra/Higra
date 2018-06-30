//
// Created by user on 4/2/18.
//

#include <boost/test/unit_test.hpp>

#include "higra/structure/fibonacci_heap.hpp"
#include "test_utils.hpp"
#include <random>

BOOST_AUTO_TEST_SUITE(fibonacci_heap);

    using namespace hg;
    using namespace std;

    template<typename T>
    struct node {

        auto get_value() {
            return m_value;
        }

        node(T value) : m_value(value) {

        }

        T m_value;
    };

    template<typename T>
    struct trivial_heap {


        bool empty() const {
            return elements.empty();
        }

        auto &push(T value) {
            elements.emplace_back(value);
            return elements[elements.size() - 1];
        }

        void merge(trivial_heap &other) {
            for (auto n: other.elements)
                elements.push_back(n);
            other.elements.clear();
        }

        auto &top() {
            return elements[imin()];
        }

        void pop() {
            if (elements.size() > 0)
                elements.erase(elements.begin() + imin());
        }

        void update(node<T> &node, const T &value) {
            node.m_value = value;
        }

        void clear() {
            elements.clear();
        }

        auto size() const {
            return elements.size();
        }

    private:
        vector<node<T>> elements;

        index_t imin() {
            T minv = elements[0].get_value();
            index_t imini = 0;
            for (index_t i = 1; i < (index_t) elements.size(); i++) {
                if (elements[i].get_value() < minv) {
                    imini = i;
                    minv = elements[i].get_value();
                }
            }
            return imini;
        }
    };

    BOOST_AUTO_TEST_CASE(test_pool_one_block) {
        fibonacci_heap_internal::object_pool<long> pool;
        long *i1 = pool.allocate();

        long *i2 = pool.allocate();
        BOOST_CHECK(i2 - i1 == 1);
        long *i3 = pool.allocate();
        BOOST_CHECK(i3 - i1 == 2);
        long *i4 = pool.allocate();
        BOOST_CHECK(i4 - i1 == 3);

        pool.free(i3);

        long *i5 = pool.allocate();
        BOOST_CHECK(i5 - i1 == 2);
        long *i6 = pool.allocate();
        BOOST_CHECK(i6 - i1 == 4);

        pool.free(i5);
        pool.free(i4);

        long *i7 = pool.allocate();
        BOOST_CHECK(i7 - i1 == 3);
        long *i8 = pool.allocate();
        BOOST_CHECK(i8 - i1 == 2);
        long *i9 = pool.allocate();
        BOOST_CHECK(i9 - i1 == 5);
        long *i10 = pool.allocate();
        BOOST_CHECK(i10 - i1 == 6);
    }

    BOOST_AUTO_TEST_CASE(test_pool_several_blocks) {
        fibonacci_heap_internal::object_pool<long> pool(3);
        long *i1 = pool.allocate();
        long *i2 = pool.allocate();
        BOOST_CHECK(i2 - i1 == 1);
        long *i3 = pool.allocate();
        BOOST_CHECK(i3 - i1 == 2);

        long *i4 = pool.allocate();
        long *i5 = pool.allocate();
        BOOST_CHECK(i5 - i4 == 1);
        long *i6 = pool.allocate();
        BOOST_CHECK(i6 - i4 == 2);

        long *i7 = pool.allocate();
        long *i8 = pool.allocate();
        BOOST_CHECK(i8 - i7 == 1);

        pool.free(i6);
        pool.free(i2);
        pool.free(i4);

        long *i9 = pool.allocate();
        BOOST_CHECK(i9 - i4 == 0);
        long *i10 = pool.allocate();
        BOOST_CHECK(i10 - i1 == 1);
        long *i11 = pool.allocate();
        BOOST_CHECK(i11 - i4 == 2);

        long *i12 = pool.allocate();
        BOOST_CHECK(i12 - i7 == 2);

        long *i13 = pool.allocate();
        long *i14 = pool.allocate();
        BOOST_CHECK(i14 - i13 == 1);
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap1) {
        fibonacci_heap_internal::fibonacci_heap<long> heap;
        heap.push(10);
        BOOST_CHECK(heap.size() == 1);
        BOOST_CHECK(!heap.empty());
        BOOST_CHECK(heap.top().get_value() == 10);
        heap.push(15);
        BOOST_CHECK(heap.size() == 2);
        BOOST_CHECK(heap.top().get_value() == 10);
        heap.push(8);
        BOOST_CHECK(heap.size() == 3);
        BOOST_CHECK(heap.top().get_value() == 8);

        heap.clear();
        BOOST_CHECK(heap.size() == 0);
        BOOST_CHECK(heap.empty());
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap2) {
        fibonacci_heap_internal::fibonacci_heap<long> heap;
        heap.push(10);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);
        BOOST_CHECK(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        heap.push(22);
        heap.push(17);

        BOOST_CHECK(heap.top().get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);


        BOOST_CHECK(heap.top().get_value() == 2);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 5);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 10);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 15);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 17);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 19);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 22);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);
    }


    BOOST_AUTO_TEST_CASE(test_fibonacci_heap3) {
        fibonacci_heap_internal::fibonacci_heap<long> heap;
        heap.push(10);
        heap.push(15);
        heap.push(8);

        fibonacci_heap_internal::fibonacci_heap<long> heap2;
        heap.push(9);
        heap.push(7);

        heap.merge(heap2);
        BOOST_CHECK(heap.top().get_value() == 7);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 8);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 9);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 10);
        heap.pop();
        BOOST_CHECK(heap.top().get_value() == 15);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);

    }

    struct rnd {
        static auto &get_rng() {
            static std::mt19937 rng(150000); //std::random_device()()
            return rng;
        }

    };

    auto random_heaps(int nbop) {
        std::uniform_int_distribution<std::mt19937::result_type> dist100(1, 100);
        std::uniform_int_distribution<std::mt19937::result_type> weights(1, 100000);

        auto &rng = rnd::get_rng();


        fibonacci_heap_internal::fibonacci_heap<long> heap;
        trivial_heap<long> theap;

        for (int i = 0; i < nbop; i++) {
            int op = dist100(rng);
            if (op < 80) {
                auto w = weights(rng);
                heap.push(w);
                theap.push(w);
            } else {
                if (theap.size() > 0) {
                    BOOST_CHECK(heap.top().get_value() == theap.top().get_value());
                    BOOST_CHECK(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    if (theap.size() > 0) {
                        BOOST_CHECK(heap.top().get_value() == theap.top().get_value());
                    }
                    BOOST_CHECK(heap.size() == theap.size());
                }
            }

        }
        return std::make_pair(std::move(heap), std::move(theap));
    }


    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_stress_test_push_pop) {
        random_heaps(10000);
        random_heaps(10000);
        random_heaps(10000);
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_stress_test_push_pop_merge) {

        std::uniform_int_distribution<std::mt19937::result_type> dist100(1, 100);
        std::uniform_int_distribution<std::mt19937::result_type> weights(1, 100000);

        auto &rng = rnd::get_rng();
        int nbop = 1000;

        fibonacci_heap_internal::fibonacci_heap<long> heap;
        trivial_heap<long> theap;

        for (int i = 0; i < nbop; i++) {
            int op = dist100(rng);
            if (op < 50) {
                auto w = weights(rng);
                heap.push(w);
                theap.push(w);
            } else if (op < 80) {
                if (theap.size() > 0) {
                    BOOST_CHECK(heap.top().get_value() == theap.top().get_value());
                    BOOST_CHECK(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    if (theap.size() > 0) {
                        BOOST_CHECK(heap.top().get_value() == theap.top().get_value());
                    }
                    BOOST_CHECK(heap.size() == theap.size());
                }

            } else {
                auto hh = random_heaps(100);
                heap.merge(hh.first);
                theap.merge(hh.second);
            }
        }
    }


BOOST_AUTO_TEST_SUITE_END();