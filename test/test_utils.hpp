//
// Created by user on 3/9/18.
//

#pragma once

#include <vector>
#include <algorithm>

template <typename T>
bool vectorEqual(std::vector<T> v1, std::vector<T> v2){
    return std::equal(v1.begin(), v1.end(), v2.begin());
}