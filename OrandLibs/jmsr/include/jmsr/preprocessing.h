/*
 * preprocessing.h
 *  Author: José M. Saavedra
 *  Copyright Orand S.A.
 *  Definition of static functions for image processing tasks based on OpenCV primitives.
 *  2013
*/

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<vector>
#ifndef PI
    #define PI 3.14159
#endif
#ifndef P_EPS
    #define P_EPS 10e-6
#endif
#define EDGE_SOBEL 0
#define EDGE_PREWITT 1
#define EDGE_DERIVADA 2

#define BKP_TERMINATION 1//BKP from breakpoint
#define BKP_BIFURCATION 2
#define BKP_ALL 3

#define JANG_DEGREE 1
#define JANG_RADIANS 2
#define JLINE_DEGREE JANG_DEGREE
#define JLINE_RADIANS JANG_RADIANS

#define NORMALIZE_UNIT 1
#define NORMALIZE_MAX 2
#define NORMALIZE_SUM 3
#define NORMALIZE_ROOT 4
#define NORMALIZE_ROOT_UNIT 5
#define NO_NORMALIZATION -1

#define JDIST_L1 1
#define JDIST_L2 2

#define ABSOLUTE_VAL 1
#define RELATIVE_VAL 2

#define J_ROI_PADDING_ZEROS 1
#define J_ROI_COMPLETE 2
/*---------------namespace jmsr for enum--------------------------------------------------------*/
namespace jmsr{
	enum NormMethod{
		UNIT=1,
		MAX=2,
		SUM=3,
		ROOT=4,
		ROOT_UNIT=5,
		NONE=0
	};
	//!< return NONE if str_norm does not match any NormMethod
	NormMethod string2norm_method(std::string str_norm);
}
/*---------------JLine class--------------------------------------------------------*/
class JLine
{
private:
    cv::Point2i point0; //!< initial point
    cv::Point2i point1;
    float ang;
    float angRad;
    float length;
    int idx_edge_link; // to be used only for edgelinks
public:
    JLine();
    JLine(cv::Point2i p0, cv::Point2i p1);
    JLine(cv::Point2i p0, cv::Point2i p1, int idx_el);
    void setPoints(cv::Point2i p0, cv::Point2i p1);
    void setInitialPoint(cv::Point2i p);
    void setEndPoint(cv::Point2i p);
    cv::Point2i getInitialPoint();
    cv::Point2i getEndPoint();
    float getAngle(int tipo=JLINE_DEGREE);
    float getLength();
    void setEdgeLinkIdx(int idx);
    int getEdgeLinkIdx();
};
/*-----------------------------------------------------------------------------------*/
class JEllipse
{
    public:
        float ang_deg;//[0..PI]
        float max_rad;
        float min_rad;
        cv::Point2i center;
    public:
        JEllipse();
        JEllipse(float _ang_deg, float _max_rad, float _min_rad, cv::Point2i _center);
        cv::Point2i getCenter();
        float getMaxRad();
        float getMinRad();
        float getAngle(int tipo=JANG_DEGREE);
        float getArea();
        float getPerimeter();
        float getRadiiRatio();
};
/*-----------------------------------------------------------------------------------*/
class Preprocessing
{
private:
    // to be used only by edgeLink
    static std::vector<cv::Point2i> getEdgeLink(const cv::Mat input,  cv::Mat &traceM, cv::Mat breakpoints, int pos_i, int pos_j);    
public:
    Preprocessing();
    static cv::Mat preprocessDigit(cv::Mat image);
    static cv::Mat preprocessDigit_1(cv::Mat image);
    static cv::Mat gammaCorrection(cv::Mat image, float gamma, bool FLAG_UCHAR=true);
    static int umbralOtsu(cv::Mat imagen, float *var=NULL);
    //to implement Sauvola method
    static int umbralByPercentil(cv::Mat imagen, float percentil=50);
    static cv::Mat histograma(cv::Mat input);
    static void smoothVector(float *vector, int n);
    static std::vector<cv::Point2i> getLinePoints_Bresenham(cv::Point2i p, cv::Point2i q);
    static void setImageValue(cv::Mat &im, std::vector<cv::Point2i> sop, uchar value, int offset_x=0, int offset_y=0);
    static void setImageValue(cv::Mat &im, std::vector<JLine> list_lines, uchar value, int offset_x=0, int offset_y=0, std::vector<int> mask=std::vector<int>());
    static void setImageColorValue(cv::Mat &im, std::vector<cv::Point2i> sop, cv::Scalar value, int offset_x=0, int offset_y=0);
    static std::vector<uchar>  getImageValue(cv::Mat im, std::vector<cv::Point2i>);

    static cv::Mat anisotropicDiffusion(const cv::Mat& input, double lambda, double K, int max_iter);
    static cv::Mat computeThermalDiffusion(cv::Mat input, double K);
    static float computeC1(int x, double K);
    static float computeC2(int x, double K);
    static cv::Mat edge_derivadas(cv::Mat input, int TIPO);

    static cv::Mat detectBreakpoints(cv::Mat input, int tipo=BKP_ALL);//Detect termination points, bifurcation or both
	static std::vector<cv::Point2i> detectBreakpoints2(cv::Mat input, int tipo=BKP_ALL);//Detect termination points, bifurcation or both
    static std::vector<std::vector<cv::Point2i> > edgeLink(cv::Mat input, int TH_MIN_SIZE=0);//compute a set of edgelinks of a binary image
    static bool isIsolated(cv::Mat input, int i, int j);
    static bool isValidPoint(int i, int j, cv::Size im_size);
    static void drawEdgeLinks(cv::Mat &image, std::vector<std::vector<cv::Point2i> > edgeLinks, cv::Scalar color=cv::Scalar(-1,-1,-1));//draw edgelinks
    static bool isNearBreakPoint(cv::Mat bkps, cv::Mat visited, int *i, int *j);
    static float getPerpendicularDistance(cv::Point2f point, cv::Point2f point0, cv::Point2f point1);
    static void findMaxDeviation(std::vector<cv::Point2i> set_of_points, int *pos, float *dist);
     static void findMaxDeviation(std::vector<cv::Point2i> set_of_points, int *pos, float *dist, int inip, int endp);
    static std::vector<int> findInflectionPoints(std::vector<cv::Point2i> sop, float TH);
    static std::vector<int> findInflectionPoints(std::vector<cv::Point2i> sop, float TH, int inip, int endp);
    static std::vector<JLine> edgeLinks2Lines(std::vector<std::vector<cv::Point2i> > edgelinks, float TH=5);//TH=MAX_DEVIATION
    static std::vector<JLine> edgeLinks2Lines(std::vector<cv::Point2i>  edgelink, float TH=5);//TH=MAX_DEVIATION
    static std::vector<cv::Point2i> getInflectionPoints(std::vector<std::vector<cv::Point2i> > edgelinks, float TH);
    static std::vector<cv::Point2i> getInflectionPoints(std::vector<cv::Point2i> edgelinks, float TH);

    static void drawJLines(cv::Mat &image, std::vector<JLine> lines, cv::Scalar color=cv::Scalar(-1,-1,-1), std::vector<int> mask=std::vector<int>());
    static cv::Mat getLineOrientationHistogram(std::vector<JLine> lines, int NBINS=10);
    static void drawHistogram(std::string name, cv::Mat hist);
    static void plot1D(std::string name, float *values, int n);
    static void plot1D(std::string name, int *values, int n);
    static cv::Mat DOC_skewCorrection(cv::Mat image, float scale=0.25);
    static cv::Mat CHECK_skewCorrection(cv::Mat image, float scale=0.25, float *angle=NULL);
    static float getAngleBySkewCorrection(cv::Mat image, float scale=0.25);
    static cv::Mat skinSegmentationMOG(cv::Mat input, double umbral);//this function implements the skin segmetation of Jones based on MoG
    static float getProb_MOG(float vector[], int DIM, float U[], float COV[], float W[], int N);//computes the prob using MOG given means U, covariance COV and weights W, N=number of MOGs
    static void drawGrid(cv::Mat &image, int stepx=10, int stepy=10, cv::Scalar color=cv::Scalar(0,0,255), bool num_line=false);    
    static void drawContour(cv::Mat &im, std::vector<cv::Point2i> sop, uchar value);
    //-----These functions are used by Curvature Scale space CSS
    static cv::Mat getGaussianFilter1D(float sigma);
    static void smoothContour(std::vector<cv::Point2i> &contour, float sigma);    
    static std::vector<int> getZeroCrossingCurvature(std::vector<cv::Point2i> contour, float sigma);
    //normaliza vector
    static void normalizeVector(float *vector, int n, int tipo=NORMALIZE_UNIT);
    static void normalizeVector(float *vector, int n, jmsr::NormMethod tipo=jmsr::NONE);
    static void linearInterBIN(float pos, int *izq, int *der, float *w_izq, float *w_der);
    static void cannyThreshold(const cv::Mat& im, float *l_th, float *h_th);
    static cv::Mat canny(const cv::Mat& im, float l_th=-1, float h_th=1, float sigma=1);
    static cv::Mat crop(const cv::Mat& im, int padding=0); // for binary images
    static cv::Mat crop(const cv::Mat& im, unsigned char bk_color, int padding=0); // for gray images
    static cv::Mat crop_rgb(cv::Mat im, cv::Scalar color, int padding=0);
    static cv::Rect get_cropping_rect(cv::Mat im, cv::Scalar color);
    static cv::Rect get_cropping_rect(cv::Mat im, unsigned char bk_color);
    static std::vector<cv::Point2d> getPointsByValue(const cv::Mat& image, unsigned char gray_value);
    static cv::Mat getPatch(const cv::Mat& image, int i, int j, cv::Size patch_size, int boder_type=J_ROI_PADDING_ZEROS);
    //crear la clase lines ylisto
    //Utilities
    /*---------------------------------------------------*/
    static cv::Point2i getCenterOfMass(cv::Mat image);
    static std::vector<std::vector<int> > getSubsets(int N);
    static cv::Mat toUint8(cv::Mat input);
    static cv::Mat toUint8_3(cv::Mat input);
    /*---------------------------------------------------*/
    //----------------------- Function to compute distances
    static float computeDistance(cv::Point p1, cv::Point p2, int type=JDIST_L2);
    static float computeDistance(cv::Point2f p1, cv::Point2f p2, int type=JDIST_L2);
    static float computeDistance(float *v1, float *v2, int size, int type=JDIST_L2);
    /*---------------------------------------------------*/
    static std::vector<int> getHistogramModesOtsu(cv::Mat hist);
    static void estimateBiModalSeparationOtsu(float *hist, int start, int end, int *pos, float *conf_sep);
    static std::vector<int> getHistogramModesOtsuR(float *hist, int start, int end);
    /*---------------------------------------------------*/
    static std::vector<int> getHistogramLocalMaximums(cv::Mat hist_h);
    static std::vector<int> getHistogramLocalMinimums(cv::Mat hist_h);
    static std::vector<int> getHistogramValleys(cv::Mat hist_h, int TH_MIN=INT_MAX);
    static cv::Mat getHorizontalProyection(cv::Mat image);
    static cv::Mat getVerticalProyection(cv::Mat image);
    /*---------------------------------------------------*/
    static JEllipse rotatedRectToEllipse(cv::RotatedRect rect);
    static float getRMSContrast(cv::Mat image);//input is a grayscale image
    static float math2image(float ang, int tipo=JANG_DEGREE);
    static float image2math(float ang, int tipo=JANG_DEGREE);
    /*----------------------------------------------------*/
    //Some functions to process check images
    static float getFocusValue(cv::Mat image, int N=64, int DPI=200);
    static int detectHorizontalBlackLines(cv::Mat bin_image, std::vector<int> &pos_lines, std::vector<int> &height_lines, float per_black_pixels=0.99);
    static float getPercentageAvgBrightness(cv::Mat image, int input_type=ABSOLUTE_VAL, float input=200);
    static float getPercentageAvgContrast(cv::Mat image, int input_type=ABSOLUTE_VAL, float input1=200, float input2=200);
    static cv::Mat alignCheckImage(cv::Mat image);//input is a gray image
    static cv::Mat alignCheckImage(cv::Mat images, float the_angle);//angle in image_cood [0..180]
    // Color conversionf
    static void BGR2oRGB(cv::Mat &input, cv::Mat& output);
    /*----------------------------------------------------*/
    //Operations obtained from text processing
    static cv::Mat textSmoothing(cv::Mat image);
    //----------------------------------------------------*/
    //Extras
    static std::string intToString(int number);
    static std::vector<std::string> splitString(std::string input,char delimiter);
    //----------------------------------------------------*/
    //----------------------------------------------------*/
    //Function for face image processing
    //paper: Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions
    // Tan&Triggs
    static cv::Mat illuminationNormalization_Tan_Triggs(cv::Mat input,float gamma=0.2, float alpha=0.1, float tau=10);
    //----------------------------------------------------*/
    // Segmentation of images
    static void grabCutSegmentation(cv::Mat &input, cv::Mat &outmask, cv::Rect rect, int n_iter);

    static void equalizeHistogram_color(cv::Mat &in_image, cv::Mat &out_image);
    //----------------------------------------------------*/
    static void bwGetExternalBorder(cv::Mat& mat_bin, cv::Mat& mat_border);
    //----------------------------------------------------*/
    static std::vector<cv::Point> bwGetExternalBorder(cv::Mat& mat_bin);
    //----------------------------------------------------*/
    //----------------------------------------------------*/
        static void padImage_8UC1(const cv::Mat& image_in, cv::Mat& image_out, int size_out, int size_in, uchar boder_value=255);
    //----------------------------------------------------*/
    static void preprocess_sketch(const cv::Mat& image_in, cv::Mat& image_out, int size_out, int size_in);
    //----------------------------------------------------*/
    static void getSketchWithLowResolution(cv::Mat& sk_image_in, cv::Mat& sk_image_out,
    		int width, int height, int resolution);
    //----------------------------------------------------*/
    //do resize keeping aspect ratio, the higher dimension is transformed to size
    static void imresize(const cv::Mat& im_in, cv::Mat& mat_out, int tartget_size);
    //----------------------------------------------------*/
    static void normalizeSketch(const cv::Mat& mat_sketch_in, cv::Mat& mat_sketch_out, int width=10);
};


#endif // PREPROCESSING_H
