//
// Created by user on 3/6/18.
//

#pragma once

#include <boost/log/trivial.hpp>
#include "xtensor/xarray.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <array>
#include <stdexcept>


namespace hg{

    template <typename...>
    struct always_false { static constexpr bool value = false; };

    template<typename T>
    struct type2CVType{
        static_assert(always_false<T>::value, "Cannot convert opencv image to given type!");//
    };

    template<>
    struct type2CVType<uchar>{
        static const int CVType = CV_8U;
    };

    template<>
    struct type2CVType<char>{
        static const int CVType = CV_8S;
    };

    template<>
    struct type2CVType<unsigned short>{
        static const int CVType = CV_16S;
    };

    template<>
    struct type2CVType<short>{
        static const int CVType = CV_16U;
    };

    template<>
    struct type2CVType<int>{
        static const int CVType = CV_32S;
    };

    template<>
    struct type2CVType<float>{
        static const int CVType = CV_32F;
    };

    template<>
    struct type2CVType<double>{
        static const int CVType = CV_64F;
    };



    template<typename T>
    xt::xarray<T> imageRead(const std::string & filename)
    {
        cv::Mat image = cv::imread(filename.c_str(), -1);

        if(! image.data )                              // Check for invalid input
        {
            throw  std::runtime_error("Error reading : " + filename);
        }


        size_t nCols = image.cols;
        size_t nRows = image.rows;

        size_t channels = image.channels();
        size_t n = nCols * nRows * channels;

        if(channels==3)
        {
            cv::cvtColor(image, image, CV_BGR2RGB);
        }

        int targetType = CV_MAKETYPE(type2CVType<T>::CVType, channels);

        cv::Mat tmp;

        if(image.type() != targetType)
        {
            image.convertTo(tmp, targetType);
        }else{
            tmp = image;
        }

        xt::xarray<T> result = xt::zeros<T>({n,});

        size_t nColsImage = nCols * channels;
        size_t nRowsImage = nRows;

        if (image.isContinuous())
        {
            nColsImage *= nRowsImage;
            nRowsImage = 1;
        }


        T * p;
        int count = 0;
        for (std::size_t i = 0; i < nRowsImage; ++i)
        {
            p = tmp.ptr<T>(i);
            for (std::size_t j = 0; j < nColsImage; ++j)
            {
                result(count++) = p[j];
            }
        }
        result.reshape({nRows, nCols, channels});
        return result;
    }

}
/*
 * cv::Mat imreadHelper(std::string filename, bool forceFloat, bool forceGrayScale)
{
    cv::Mat image;
    image = cv::imread( filename.c_str(), (forceGrayScale)? 0 : -1 );

    if( !image.data )
    {
        throw std::runtime_error("No Image Data");
    }
    cv::Mat tmp;

    if(forceFloat)
    {
        int channels = image.channels();
        int depth = image.depth();
        int targetType;
        if(channels == 1)
            targetType = CV_32FC1;
        else if(channels == 3)
            targetType = CV_32FC3;
        else
            throw std::runtime_error("Unsupported number of channels");


        image.convertTo(tmp, targetType);

        if(depth<=1)
            tmp /= 255.0;
    } else {
        tmp = image;
    }


    return tmp;

}

void imwriteHelper(cv::Mat image, std::string filename)
{
    int depth = image.depth();
    if(depth<=1)
    {
        cv::imwrite(filename.c_str(), image);
    } else {
        double min, max;
        cv::minMaxLoc(image, &min, &max);
        if(min < -0.000001 || max > 1.000001)
            std::cerr << "!!!  Warning, saved image values not between 0 and 1." << std::endl;

        cv::Mat tmp = image.clone();
        tmp *= 255;

        cv::imwrite(filename.c_str(), tmp);
    }

}
 */