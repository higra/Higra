//
// Created by user on 3/22/18.
//

#pragma once

#define HG_DEBUG


//#include <boost/log/trivial.hpp>
#include <exception>
#include <string>

#ifndef __FUNCTION_NAME__
#ifdef WIN32   //WINDOWS
#define __FUNCTION_NAME__   __FUNCTION__
#else          //*NIX
#define __FUNCTION_NAME__   __func__
#endif
#endif

//BOOST_LOG_TRIVIAL(error) << msg << "\n";
#ifdef HG_DEBUG
#define hg_assert(test, msg) do { \
    if(!(test)) {\
    throw std::runtime_error(std::string() + __FUNCTION_NAME__ + " in file " + __FILE__ + "(line:" + std::to_string(__LINE__) + "): "  + msg);} \
  } while (0)
#else
#define hg_assert(test, msg) ((void)0)
#endif
