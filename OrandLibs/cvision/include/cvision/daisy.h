/*
 * Daisy.h
 *
 *  Created on: Apr 24, 2015
 *      Author: Jose M. Saavedra
 *      This is an implementation based on the Matlab code of Engin Tola
 */

#ifndef CVISION_DAISY_H_
#define CVISION_DAISY_H_
#include "opencv2/highgui/highgui.hpp"
#include <cmath>
#include <vector>
#include <string>

const double PI = 3.14159;

class GridPoint{
private:
	float radius;
	float comp_x;
	float comp_y;
public:
	GridPoint();
	GridPoint(float _radius, float _x, float _y);
	void setRadius(float _radius);
	void setComp_X(float _x);
	void setComp_Y(float _y);
	void setValues(float _radius, float _x, float _y);
	float getRadius();
	float getComp_X();
	float getComp_Y();
	std::string toString();
};

class DiasyParams{

};
class Daisy {
public:
	Daisy();
	static std::vector<GridPoint> computeGrid(int R, int RQ, int TQ);
	static std::vector<float> computeLevelSigmas(int R, int RQ);
	static void layeredGradient(cv::Mat& image, int n_layers, std::vector<cv::Mat>& layers_g);
	static cv::Mat gaussian_1d(float sigma, int size);
	static int getFilterSize(float sigma);
	static void smoothLayers(std::vector<cv::Mat>& layers_g, float sigma);
	static void copyLayers(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst);
	static float* computeDaisy(std::vector<GridPoint>& grid, int HQ,
			std::vector<std::vector<cv::Mat> >& cubes,
			int rows, int cols,int pos_x, int pos_y);
	static std::vector<float*> compute(cv::Mat& image, int* size_des, int R, int RQ, int TQ, int HQ);
	static float* compute(cv::Mat& image, int pos_i, int pos_j, int* size_des, int R, int RQ, int TQ, int HQ);
	virtual ~Daisy();
};

#endif /* CVISION_DAISY_H_ */
