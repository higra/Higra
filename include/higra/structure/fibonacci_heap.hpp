//
// Created by perretb on 27/06/18.
//

#pragma once

#include <vector>
#include <stack>
#include "../utils.hpp"

namespace hg {

    namespace fibonacci_heap_internal {

        template<typename T>
        struct object_pool{

            void free(const T * object){
                auto i = object - &pool[0];
                free_chain[i] = first_free;
                first_free = i;
            }

            T* allocate(){
                if(first_free != -1){
                    index_t tmp = first_free;
                    first_free = free_chain[first_free];
                    return &pool[tmp];
                }
                pool.emplace_back();
                free_chain.push_back(invalid_index);
                return &pool[pool.size() - 1];
            }

        private:
            std::vector<T> pool;
            std::vector<T> free_chain;
            index_t first_free = invalid_index;
        };


        template<typename T>
        struct fibonacci_heap;

        template<typename T>
        struct node {
            using self_t = node<T>;
            using pointer_t = self_t*;
        private:
            pointer_t m_previous;
            pointer_t m_next;
            pointer_t m_child;
            pointer_t m_parent;

            T m_value;

            int m_degree;
            bool m_marked;

            node(){}

            void init(T value){
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

            T getValue() { return m_value; }
        };


        template<typename T>
        struct fibonacci_heap final {
            using node_t = node<T>;

        private:
            static object_pool<node_t> s_pool;
            node_t * m_heap = nullptr;
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

            auto & push(T value) {
                m_size++;
                node_t * new_node = s_pool.allocate();
                new_node->init(value);
                m_heap = m_merge(m_heap, new_node);
                return *new_node;
            }

            void merge(fibonacci_heap & other){
                m_heap = m_merge(m_heap, other.m_heap);
                m_size += other.size();
                other.m_size = 0;
                other.m_heap = nullptr;
            }

            auto & top(){
                return *m_heap;
            }

            void pop(){
                // TODO
            }

            void clear(){
                std::stack<node_t *> s;
                s.push(m_heap);
                for(node_t * tmp = m_heap->next; tmp != m_heap; tmp = tmp->next){
                    s.push(tmp);
                }
                while(!s.empty()){
                    node_t * n = s.top;
                    s.pop();
                    node_t * c = n->child;
                    s.push(c);
                    for(node_t * tmp = c->next; tmp != c; tmp = tmp->next){
                        s.push(tmp);
                    }
                    s_pool.free(n);
                }
                m_heap = nullptr;
            }

            auto size() const{
                return m_size;
            }

        private:
            auto m_merge(node_t * root1, node_t * root2) {
                if(root1 == nullptr)
                    return root2;
                if(root2 == nullptr)
                    return root1;

                if(root1->value < root2->value){
                    std::swap(root1, root2);
                }

                node_t * root1_previous = root1->m_previous;
                node_t * root2_next = root2->m_next;

                root2->m_next = root1;
                root1->m_previous = root2;

                root1_previous->m_next = root2_next;
                root2_next->m_previous = root1_previous;

                return root2;
            }


        };
    }


}
