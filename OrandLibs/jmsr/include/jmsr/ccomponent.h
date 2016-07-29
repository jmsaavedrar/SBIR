/*
ccomponent.h
Copyright (C) Jose M. Saavedra
ORAND S.A.
*/
#ifndef CCOMPONENT_H
#define CCOMPONENT_H

#include<opencv2/core/core.hpp>
#include<vector>

#define CC_SORT_ASCEND 1
#define CC_SORT_DESCEND 0
#define SKEW_SIGNED 1
#define SKEW_NON_SIGNED 2

#define CONNECTED_8 8
#define CONNECTED_4 4
class CComponent
{
private:
    cv::Mat image; //image must be a binary image CV_8UC1
    bool cc_ready;
    std::vector<std::vector<cv::Point2i> > cc_points;
    std::vector<cv::Rect> cc_bounding_box;
    static void getMaximo(std::vector<int> A, int &maxi, int &pos);//return by reference maximo y pos del maximo
    static std::vector<int> getAreas(std::vector<std::vector<cv::Point2i> > V);
    void Init(cv::Mat _image, bool sort_by_x=true);//Initialize after having cheched the input image

    static void traceImage(cv::Mat &image, cv::Point2i point, int connection_type=CONNECTED_8);
public:
    CComponent();
    CComponent(cv::Mat _image, bool sort_by_x=true); //sort_by_x only sort bounding boxes

    void setImage(cv::Mat _image, bool sort_by_x=true);
    cv::Mat bwAreaOpen(double area);//eliminates small objects, smaller that area using points
    cv::Mat bwDiscardSmallRegions(double th_area);
    cv::Mat bwDiscardLargeRegions(double th_area);
    cv::Mat bwRectAreaOpen(double area);//eliminates small objects, smaller that area using rects
    cv::Mat bwHeightRectOpen(double u_height);//eliminates small objects, smaller that area
    cv::Mat bwWidthRectOpen(double u_width);//eliminates small objects, smaller that area
    cv::Mat bwBigger();//hold the biggest object
    //Mat bwSmaller();

    cv::Rect getBoundingBox(int n_cc);//returns the bounding box of component n_cc
    cv::Point2i getCentroid(int n_cc);
    int getSize(int n_cc);//returns the number of points
    std::vector<cv::Point2i> getPoints(int n_cc);
    cv::Mat getLabelMatrix();

    int getNumberOfComponents();
    void getMinMaxAllBoundingBoxes(double *min_x, double *max_x, double *min_y, double *max_y);
    cv::Rect getMinMaxAllBoundingBoxes();
    std::vector<cv::Rect> getBoundingBoxes();
    std::vector<int> getAreas();
    std::vector<float> getSkew(int sing=SKEW_SIGNED); //return in math coordinates in radians
    float getMeanHeight(float *dsv=NULL); //get the mean height of rectangles
    //vector<Point2i> getPoints(int n_cc);//returns points of component n_cc

    static void setValues(cv::Mat image, std::vector<cv::Point2i> list_of_points, const uchar value);
    static cv::Mat fillHoles(cv::Mat image);
    static std::vector<cv::Point2i> bwBoundary(cv::Mat image);
    static cv::Mat getBoundaryImage(cv::Mat image);
    static std::vector<cv::Point2i> bwGetPoints(const cv::Mat& im_bin);
    /*--------------------------------------------------------------------------------------*/
    static float intersectX(cv::Rect A, cv::Rect B);
    static void jointRects(cv::Rect &A, cv::Rect B);
    static std::vector<cv::Rect> joinYBoundingBoxes(std::vector<cv::Rect> loBB);
    /*--------------------------------------------------------------------------------------*/
    static  void sortComponentValue(float *values, int *index_cc, int n, int tipo=CC_SORT_ASCEND);
    /*--------------------------------------------------------------------------------------*/
    void drawBoundingBoxes(cv::Mat &image, cv::Scalar color=cv::Scalar(255,255,0));
    static void drawBoundingBoxes(cv::Mat &image, std::vector<cv::Rect> list_bb, cv::Scalar color=cv::Scalar(255,255,0));
    /*--------------------------------------------------------------------------------------*/
    void jointCComponents(cv::Mat &bin_image, int id_cc1, int id_cc2, float TH=0);//this function only affects the an input image
    //JoingCCcomponentsByDT(distance transform) is more robust that the one based on terminal points
    void jointCComponentsByDT(cv::Mat &bin_image, int id_cc1, int id_cc2, float TH=0);//joining ccomponent using distance transform
    //For text descriptors
    static void traceFromThisPoint(cv::Mat &image, cv::Point2i point, int connection_type=CONNECTED_8); //just keep the connected component including the point
    //---------------------------------------------
    static float bwGetCircularity(cv::Mat& mat_bin);
    //---------------------------------------------
    static bool bwCircleEstimation(cv::Mat& mat_bin, float* x_c, float* y_c, float* radius);
    //---------------------------------------------
    static float* bwGetTopProfile(const cv::Mat& mat_bin, unsigned int* size);
};

#endif // CCOMPONENT_H
