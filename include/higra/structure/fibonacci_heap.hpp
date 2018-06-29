//
// Created by perretb on 27/06/18.
//

#pragma once

#include <vector>
#include <stack>
#include <array>
#include "../utils.hpp"

namespace hg {

    namespace fibonacci_heap_internal {

        template<typename T>
        struct object_pool {


            union object {
                T t;
                object *next;

                object() {
                    next = nullptr;
                }

                ~object() {

                }
            };


            void free(const T *element) {
                object *new_first = (object *) element;
                new_first->next = first_free;
                first_free = new_first;
            }

            T *allocate() {
                if (first_free == nullptr) {
                    pool.emplace_back(m_blocksize);
                    auto &block = pool[pool.size() - 1];
                    first_free = &block[0];
                    for (index_t i = 0; i < block.size() - 1; i++)
                        block[i].next = first_free + i + 1;
                    block[block.size() - 1].next = nullptr;
                }

                object *tmp = first_free;
                first_free = first_free->next;
                return (T *) tmp;
            }

            object_pool(size_t blocksize = 4096) : m_blocksize(blocksize) {

            }

        private:
            std::vector<std::vector<object>> pool;
            size_t m_blocksize;
            object *first_free = nullptr;
        };


        template<typename T>
        struct fibonacci_heap;

        template<typename T>
        struct node {
            using self_t = node<T>;
            using pointer_t = self_t *;
        private:
            pointer_t m_previous;
            pointer_t m_next;
            pointer_t m_child;
            pointer_t m_parent;

            T m_value;

            int m_degree;
            bool m_marked;

            node() {}

            void init(T value) {
                m_value = value;
                m_previous = this;
                m_next = this;
                m_child = nullptr;
                m_parent = nullptr;
                m_degree = 0;
                m_marked = false;
            }

        public:
            friend class fibonacci_heap<T>;

            T get_value() { return m_value; }
        };


        template<typename T>
        struct fibonacci_heap final {
            using node_t = node<T>;

        private:
            static object_pool<node_t> &s_pool() {
                static object_pool<node_t> pool{};
                return pool;
            }

            node_t *m_heap = nullptr;
            size_t m_size = 0;

        public:
            fibonacci_heap() {

            }

            ~fibonacci_heap() {
                clear();
            }

            bool empty() const {
                return m_heap == nullptr;
            }

            auto &push(T value) {
                m_size++;
                node_t *new_node = s_pool().allocate();
                new_node->init(value);
                m_heap = m_merge(m_heap, new_node);
                return *new_node;
            }

            void merge(fibonacci_heap &other) {
                m_heap = m_merge(m_heap, other.m_heap);
                m_size += other.size();
                other.m_size = 0;
                other.m_heap = nullptr;
            }

            auto &top() {
                return *m_heap;
            }

            void pop() {
                m_extract_min();
            }

            void update(node_t &node, const T &value) {

            }

            void clear() {
                if (m_heap == nullptr)
                    return;

                std::stack<node_t *> s;
                s.push(m_heap);
                for (node_t *tmp = m_heap->m_next; tmp != m_heap; tmp = tmp->m_next) {
                    s.push(tmp);
                }
                while (!s.empty()) {
                    node_t *n = s.top();
                    s.pop();
                    node_t *c = n->m_child;
                    if (c != nullptr) {
                        s.push(c);
                        for (node_t *tmp = c->m_next; tmp != c; tmp = tmp->m_next) {
                            s.push(tmp);
                        }
                    }
                    s_pool().free(n);
                }
                m_heap = nullptr;
                m_size = 0;
            }

            auto size() const {
                return m_size;
            }

        private:
            auto m_merge(node_t *root1, node_t *root2) {
                if (root1 == nullptr)
                    return root2;
                if (root2 == nullptr)
                    return root1;

                if (root1->m_value < root2->m_value) {
                    std::swap(root1, root2);
                }

                node_t *root1_previous = root1->m_previous;
                node_t *root2_next = root2->m_next;

                root2->m_next = root1;
                root1->m_previous = root2;

                root1_previous->m_next = root2_next;
                root2_next->m_previous = root1_previous;

                return root2;
            }

            void m_link_heap(node_t *y, node_t *x) {
                y->m_previous->m_next = y->m_next;
                y->m_next->m_previous = y->m_previous;

                y->m_parent = x;
                auto *child = x->m_child;
                if(child != nullptr) {
                    y->m_next = child->m_next;
                    y->m_previous = child;
                    child->m_next->m_previous = y;
                    child->m_next = y;
                }else{
                    y->m_next = y;
                    y->m_previous = y;
                    x->m_child = y;
                }
                x->m_degree++;
                y->m_marked = false;
            }

            void m_consolidate() {
                if (m_heap == m_heap->m_next)
                    return;

                std::array<node_t *, 64> A;
                std::fill(A.begin(), A.end(), nullptr);

                auto *x = m_heap;
                do {
                    auto d = x->m_degree;
                    while (A[d] != nullptr) {
                        auto *y = A[d];
                        if (y->m_value < x->m_value) {
                            std::swap(x, y);
                        }
                        m_link_heap(y, x);
                        A[d] = nullptr;
                        d++;
                    }
                    A[d] = x;

                    x = x->m_next;
                } while (x != m_heap);

                node_t *minh = nullptr;
                int i = 0;
                while (minh == nullptr) {
                    if (A[i] != nullptr) {
                        minh = A[i];
                    }
                    i++;
                }

                auto *start = minh;
                auto *pos = minh->m_next;

                while (pos != start) {
                    if (pos->m_value < minh->m_value)
                        minh = pos;
                    pos = pos->m_next;
                }
                m_heap = minh;

            }

            void m_extract_min() {
                if (m_heap != nullptr) {
                    auto *child = m_heap->m_child;
                    if(child != nullptr) {
                        do {
                            child->m_parent = nullptr;
                            child = child->m_next;
                        } while (child != m_heap->m_child);
                    }

                    m_merge(m_heap, m_heap->m_child);

                    m_heap->m_previous->m_next = m_heap->m_next;
                    m_heap->m_next->m_previous = m_heap->m_previous;

                    auto *old_heap = m_heap;
                    if (old_heap->m_next == old_heap) {
                        m_heap = nullptr;
                    } else {
                        m_heap = old_heap->m_next;
                        m_consolidate();
                    }
                    s_pool().free(old_heap);
                    m_size--;
                }
            }

        };
    }


}
