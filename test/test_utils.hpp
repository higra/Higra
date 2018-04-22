//
// Created by user on 3/9/18.
//

#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <boost/type_index.hpp>
#include "xtensor/xmath.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xio.hpp"

template <typename T>
bool vectorEqual(std::vector<T> v1, std::vector<T> v2){
    if (v1.size() != v2.size())
        return false;
    return std::equal(v1.begin(), v1.end(), v2.begin());
}

template<typename T>
void showTypeName(std::string msg = "") {
    std::cout << msg << boost::typeindex::type_id<T>().pretty_name() << std::endl;
}

template<typename T>
struct COMPILE_ERROR;