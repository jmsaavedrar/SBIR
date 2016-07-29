/*
descriptor.h
Copyright (C) Jose M. Saavedra
ORAND S.A.
*/

#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "preprocessing.h"
#include <vector>
//Generic class--------------------------------------
static cv::Mat JMSR_EMPTY_MAT=cv::Mat();
class Params
{    
public:    
    //Params();
    virtual int getDescriptorSize(cv::Size s);
    virtual int getDescriptorSize();
    virtual std::string toString();
    virtual ~Params();

};
//HOGParams--------------------------------------
class HOGParams : public Params
{
    //supossing square regions
public:
    int cell_size; //pixel per cell
    int block_size; // cell per block
    int n_channels;
    float th_intersection;
    cv::Size image_size;
    int size;//it is not used
    HOGParams();
    HOGParams(int _cell_size, int _block_size, int _n_channels, float _th_intersection);
    HOGParams(int _cell_size, int _block_size, int _n_channels, float _th_intersection, cv::Size _image_size);
    int getDescriptorSize(cv::Size s);
    int getDescriptorSize();
    cv::Size getSizeParam();
    std::string toString();
};
//HOG_SPParams-----HOG+Spatial Distribution
class HOG_SPParams : public HOGParams
{
public:
    HOG_SPParams();
    HOG_SPParams(int _cell_size, int _block_size, int _n_channels, float _th_intersection, cv::Size _image_size);
    using HOGParams::getDescriptorSize;
    int getDescriptorSize();
    std::string toString();
};
//HELOParams--------------------------------------
//used also by shelo
class HELOParams  : public Params
{
public:
    int normalization;
    bool squared_root;
    int n_cells;
    int n_bins;
    int n_blocks;//blocks por row or col
    HELOParams();
    HELOParams(int _ncells, int _b_bins, bool sr=false);
    HELOParams(int _ncells, int _b_bins, int _n_blocks, bool sr=false);
    HELOParams(int _ncells, int _b_bins, int _n_blocks, int _normalization, bool sr=false);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    std::string toString();

};
//SHELO_MSParams--------------------------------------
//SHELO params with multiples scale
class SHELO_MSParams  : public Params
{
public:
    int normalization;
    int n_cells;
    int n_bins;
    int n_blocks;//blocks por row or col
    int n_levels;
    std::vector<int> n_cells_by_level;
    SHELO_MSParams();
    SHELO_MSParams(int _n_cells, int _n_bins, int _n_blocks, int _normalization, int levels);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    int getDescriptorSizeByLevel();
    int geNCells(int level);
    std::string toString();

};
//SHELO_SPParams, for be used in spatial piramyd and RS2T_SHELO
class SHELO_SPParams  : public Params
{
public:
    std::vector<HELOParams> sp_params;
    int n_levels;
    SHELO_SPParams();
    SHELO_SPParams(int _n_cells, int _n_bins, int _normalization, int _n_levels, int _d_factor=2);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    int getDescriptorSizeByLevel(int i);
    std::string toString();

};
//JHOGParams--------------------------------------
//This is a simple Histograms of Gradients
class JHOGParams  : public Params
{
public:
    int n_blocks;//blocks por row or col
    int n_bins;
    JHOGParams();
    JHOGParams(int _n_blocks, int _n_bins);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    std::string toString();
};
//JHOGParams--------------------------------------
//uisng different number of blocks per row and cols
class JHOGParams2  : public Params
{
public:
    int n_blocks_v;//blocks por row or col
    int n_blocks_h;//blocks por row or col
    int n_bins;
    JHOGParams2();
    JHOGParams2(int _n_blocks_v, int _n_blocks_h, int _n_bins);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    std::string toString();
};
//ConcavityParams--------------------------------------
class ConcavityParams  : public Params
{
public:
    int n_blocks;//blocks por row or col
    ConcavityParams();
    ConcavityParams(int _n_blocks);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    std::string toString();
};
class DTParams: public Params
{
public:
    int n_blocks;//blocks por row or col
    cv::Size im_size;
    DTParams();
    DTParams(int _n_blocks, cv::Size _im_size);
    DTParams(int _n_blocks, int width, int height);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    std::string toString();
};
//HELOParams--------------------------------------
class LBPParams  : public Params //This may be used by SLBP and LBP
{
public:
    int num_cells;
    int quantize_value;
    int radius;
    int n_neighbors;
    LBPParams();
    LBPParams(int _num_cells);
    LBPParams(int _num_cells, int _quantize_value, int _radius=1, int _n_neighbors=8);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    std::string toString();

};
//CLDParams Color layout descriptor----------------
class CLDParams : public Params //This may be used por CLD and DCT
{
public:
    int num_cells_x;
    int num_cells_y;
    CLDParams();
    CLDParams(int n_cells_y, int n_cells_x);
    int getDescriptorSize();
    int getDescriptorSize(cv::Size s);
    std::string toString();
};
//Descriptor----------------------------------------
class Descriptor
{
private:
    static float *computerPerim2(int *vector_x, int *vector_y, int vector_size, float &total_perim);
    static float *getCentroidDistance(int *vector_x, int *vector_y, int vector_size);    

public:
    Descriptor();
    static void  sampligByArcLenght(int *&vector_x, int *&vector_y, int vector_size, int number_of_samples);
    static float *fourierDescriptor(int *vector_x, int *vector_y, int vector_size, int number_of_samples);
    static float *fourierDescriptor(std::vector<cv::Point2i> contour, int number_of_samples);
    static float *getHOGDescriptor(cv::Mat image, int number_of_bins);
    static void   getHOGDescriptor(cv::Mat image, int number_of_bins, float *des);
    static float *getHOGDescriptorLocal(cv::Mat image, int number_of_bins);
    static float *getJHOGLocalDescriptor(cv::Mat image, JHOGParams params=JHOGParams());
    static float *getSoftJHOGLocalDescriptor(cv::Mat image, JHOGParams params=JHOGParams());
    static float *getHOGLocal_2x1(cv::Mat image1, int number_of_bins);
    static float *getJLocalConcavity8(cv::Mat image, ConcavityParams params=ConcavityParams());
    static float *getJLocalConcavity4(cv::Mat image, ConcavityParams params=ConcavityParams());
    static float *getDistanceTransformDescriptor(cv::Mat image, cv::Size im_size=cv::Size(30,30));
    static float *getLocalDistanceTransformDescriptor(cv::Mat image, DTParams params=DTParams(2,30,30));

    static float *getConcavidad(cv::Mat image);
    static float *getConcavidad8(cv::Mat image);
    static float *getBuenDescriptor(cv::Mat, int *des_size);
    static float *getBuenDescriptor8(cv::Mat, int *des_size);
    static float *getConcavidadLocal(cv::Mat image); //This is only for 2x2 partition
    static float *getConcavidad8Local(cv::Mat image);//Thi is only fo 2x2 partition
    static float *getHorizontalProfile(cv::Mat image);
    static float *getVerticalProfile(cv::Mat image);
    static float *getConcavityDescriptor(cv::Mat image, int *size_des);//according to the paper of Oliveira
    static float *getLocalConcavityDescriptor(cv::Mat image, int *size_des);//according to the paper of Oliveira
    static float *getBuenDescriptor3(cv::Mat image, int *size_des);
    static float *getHOGDalalTriggs(cv::Mat image, HOGParams params=HOGParams());
    static float *getHOGDalalTriggs_SpatialPyramid(cv::Mat image, HOG_SPParams params=HOG_SPParams());
#ifndef NO_VLFEAT
    static float *getHOGDalalTriggsVL(cv::Mat image, HOGParams params=HOGParams());
    static float *getHOGDalalTriggsFZ(cv::Mat image, HOGParams params=HOGParams());
    static float *getLBP_VL_Descriptor(cv::Mat image, LBPParams params=LBPParams());
    static void HOG_VL_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints,  cv::Mat &descriptors, HOGParams params=HOGParams());
    static void HOG_FZ_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints,  cv::Mat &descriptors, HOGParams params=HOGParams());
    static void LBP_VL_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints,  cv::Mat &descriptors, LBPParams params=LBPParams());
    static int estimate_HOG_VL_size(cv::Size image_size, HOGParams params);
    static int estimate_HOG_FZ_size(cv::Size image_size, HOGParams params);
#endif
    //--------------Texture based descriptors-----------------------------------------
    static float *getSimpleLBPDescriptor(cv::Mat image, int n_bins, int normed=NORMALIZE_UNIT);
    static float *getGrayLayoutDescriptor(cv::Mat image, CLDParams params=CLDParams());
    static float *getDCTDescriptor(cv::Mat image, CLDParams params=CLDParams());
    //--------------------------------------------------------------------------------
    static void  drawHOGDalalTriggs(cv::Mat image, float *hog, float max_ang, HOGParams params=HOGParams());
    static bool  isValid(int x, int y, int min_x, int max_x, int min_y, int max_y);
    static float *getHELODescriptor(cv::Mat image, HELOParams params=HELOParams());
    static float *getLocalHELODescriptor(cv::Mat image, HELOParams params=HELOParams(), cv::Mat& im_draw=JMSR_EMPTY_MAT);
    static float *getLocalHELODescriptor_with_mask(cv::Mat image, HELOParams params, cv::Mat mask, bool draw=false, cv::Mat* im_draw=NULL, cv::Scalar color=cv::Scalar(0,255,255));//S_HELO
    static float* getSHELO_MS(cv::Mat image, SHELO_MSParams params=SHELO_MSParams()); //deprecated
    static float* getSHELO_SP(cv::Mat image, SHELO_SPParams params=SHELO_SPParams());
    static void HOG_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints,  cv::Mat &descriptors, HOGParams params=HOGParams());
    static void LBP_COLOR_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints,  cv::Mat &descriptors, LBPParams params=LBPParams());
    static void LBP_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints,  cv::Mat &descriptors, LBPParams params=LBPParams());
    static void SHELO_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, HELOParams params=HELOParams());
    static void computeDenseKeypoints(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, int stepx=50, int stepy=50, int npoints=-1);

    /*-------------Functions to compute lbp-------------------------------------------------*/
    //  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
    //  patterns: Application to face recognition." IEEE Transactions on Pattern
    //  Analysis and Machine Intelligence, 28(12):2037-2041.
    static void lbp_extendend(cv::InputArray src, cv::OutputArray dst, int radius=1, int neighbors=8);
    static float *lbp_spatial_histogram(cv::InputArray src, int numPatterns, int grid_x=8, int grid_y=8, bool normed=true);
    static float *getLBP_Face(cv::Mat image, LBPParams params); //this was used for face recognition
    static float *getExtendedLBPDescriptor(cv::Mat image, LBPParams params);
    static unsigned char bin_lbp_8bits(unsigned char val);
    static float *get_lbp_8bits(cv::Mat image, int radius=1, int normed=NORMALIZE_UNIT);
    static float *get_lbp_8bits_grid(cv::Mat image, LBPParams params=LBPParams(4,59,1,8), int normed=NO_NORMALIZATION);
    static float *get_color_lbp_8bits_grid(cv::Mat image, LBPParams params=LBPParams(4,59,1,8), int normed=NO_NORMALIZATION);
    //Color descriptors
    static float* getRGBColorHistogram(cv::Mat image, int k, int *size);
    static float* getRGBColorHistogram_grid3(cv::Mat query, int k, int *size);
    //Color descriptors
    static float* get_SHELO_2_for_image(cv::Mat image, int *size_des, cv::Mat mask=cv::Mat());
    static float* get_SHELO_2_for_image(const char* image_file, int *size_des,cv::Mat mask=cv::Mat());
    static float* get_HOG_Dalal_for_image(const char* image_file, int *size_des);
    static float* get_HOG_Dalal_for_image(cv::Mat image_file, int *size_des);    
};




#endif // DESCRIPTOR_H
