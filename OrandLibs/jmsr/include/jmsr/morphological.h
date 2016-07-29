/*
 * Jmorphological.h
 *
 *  Author: José M. Saavedra
 *  Copyright Orand S.A.
 */

#ifndef MORPHOLOGICAL_H
#define MORPHOLOGICAL_H

#include <opencv2/core/core.hpp>

class Morphological
{
public:
    Morphological();
    /*!
     *
     * @param input input binary image
     * @param iter number ot iterations
     * @param done_it number of iterations done to complete the job
     * @return
     */
    static cv::Mat thinning2(cv::Mat input, int iter, int*  done_it=nullptr);
    static cv::Mat thinning_Zhang_Sue(cv::Mat input, int* done_it=nullptr);

private:
    static bool evaluateGuoHall(cv::Mat input,  int ib, int jb, int iter);
    static void thinSubIteration1(cv::Mat &pSrc, cv::Mat &pDst);
    static void thinSubIteration2(cv::Mat &pSrc, cv::Mat &pDst);


};

#endif // MORPHOLOGICAL_H
