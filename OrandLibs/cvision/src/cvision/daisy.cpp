/*
 * Daisy.cpp
 *
 *  Created on: Apr 24, 2015
 *      Author: jsaavedr
 */

#include "daisy.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "jmsr/JUtil.h"
#include "jmsr/preprocessing.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cassert>
static int VERBOSE=false;
GridPoint::GridPoint():radius(0), comp_x(0), comp_y(0){
}
GridPoint::GridPoint(float _radius, float _x, float _y):radius(_radius), comp_x(_x), comp_y(_y){
}
void GridPoint::setRadius(float _radius){
	radius=_radius;
}
void GridPoint::setComp_X(float _x){
	comp_x=_x;
}
void GridPoint::setComp_Y(float _y){
	comp_y=_y;
}
void GridPoint::setValues(float _radius, float _x, float _y){
	radius=_radius; comp_x=_x; comp_y=_y;
}
float GridPoint::getComp_X(){
	return comp_x;
}
float GridPoint::getComp_Y(){
	return comp_y;
}
float GridPoint::getRadius(){
	return radius;
}
std::string  GridPoint::toString(){
	return std::to_string(radius)+" "+
			std::to_string(comp_x)+ " "+
			std::to_string(comp_y);
}
/*--------------------------------------------------------------*/
Daisy::Daisy(){
}
Daisy::~Daisy(){
	// TODO Auto-generated destructor stub
}
/*--------------------------------------------------------------*/
std::vector<GridPoint> Daisy::computeGrid(int R, int RQ, int TQ){
	float rs=((float)R)/RQ;
	float ts=(2.0*PI)/TQ;
	int gs=RQ*TQ+1;
	std::vector<GridPoint> grid(gs);
	int cnt=0;
	float rv=0, tv=0;
	grid[cnt].setValues(0,0,0);
	for(int r=0; r<RQ; r++){
		for(int t=0; t<TQ; t++){
			cnt++;
			rv=(r+1)*rs;
			tv=t*ts;
			grid[cnt].setValues((float)(r), rv*cos(tv), rv*sin(tv));
		}
	}
	return grid;
}
/*--------------------------------------------------------------*/
std::vector<float> Daisy::computeLevelSigmas(int R, int RQ){
	std::vector<float> sigmas(RQ);
	float rs=((float)R)/RQ;
	for(int r=0; r<RQ; r++){
		sigmas[r]=(r+1)*0.5*rs;
	}
	return sigmas;
}
/*--------------------------------------------------------------*/
cv::Mat Daisy::gaussian_1d(float sigma, int size){
	//size is odd
	int sz=floor((float)(size-1)/2);
	float v=sigma*sigma;
	cv::Mat filter(1,size, CV_32F);
	int k=0;
	float val=0;
	float sum=0;
	for(float x=-sz; x<=sz; x++){
		val=exp(-(0.5*x*x)/(float)v);
		sum+=val;
		filter.at<float>(0,k++)=val;
	}
	for(int i=0; i<k;i++){
		filter.at<float>(0,i)=filter.at<float>(0,i)/sum;
	}
	return filter;

}

void Daisy::layeredGradient(cv::Mat& image, int n_layers, std::vector<cv::Mat>& layers_g){
	//first smooth the image
	assert(image.type()==CV_32F);
	layers_g.resize(n_layers);
	//std::cout<<"2"<<std::endl;
	cv::Mat gFilter=gaussian_1d(0.5,5);
	//std::cout<<"1"<<std::endl;
	cv::Mat image_f;
	image.copyTo(image_f);
	cv::Mat image_g;
	cv::filter2D(image_f,image_g, CV_32F,gFilter,cv::Point(-1,-1),0,0);
	//std::cout<<"2"<<std::endl;
	//std::cout<<image_g<<std::endl;
	cv::filter2D(image_g,image_f, CV_32F,gFilter.t(),cv::Point(-1,-1),0,0);
	//std::cout<<image_f<<std::endl;

	cv::Mat dFilter=(cv::Mat_<float>(1,3)<<-0.5, 0, 0.5);
	cv::Mat im_dx;
	cv::Mat im_dy;
	cv::filter2D(image_f,im_dx, CV_32F,dFilter,cv::Point(-1,-1),0,0);
	cv::filter2D(image_f,im_dy, CV_32F,dFilter.t(),cv::Point(-1,-1),0,0);

	int h=image.rows;
	int w=image.cols;
	float th=0;
	float pi2s=2*PI/n_layers;
	float kos, zin;
	float val=0;
	for(int l=0; l<n_layers; l++){
		th=l*pi2s;
		kos=cos(th);
		zin=sin(th);
		layers_g[l].create(h,w,CV_32F);
		for(int i=0; i<h;i++){
			for(int j=0; j<w;j++){
				val=kos*im_dx.at<float>(i,j)+zin*im_dy.at<float>(i,j);
				layers_g[l].at<float>(i,j)=std::max(val,0.0f);
			}
		}
	}
}

int Daisy::getFilterSize(float sigma){
	int fsz = floor(5*sigma);
	if ((fsz%2) == 0) fsz = fsz +1;
	if (fsz < 3)	fsz = 3;
	return fsz;
}
void Daisy::smoothLayers(std::vector<cv::Mat>& layers_g, float sigma){
	int size=getFilterSize(sigma);
	cv::Mat filter=gaussian_1d(sigma,size);
	cv::Mat im_f;
	for(size_t i=0; i<layers_g.size();i++){
		cv::filter2D(layers_g[i],im_f, CV_32F,filter,cv::Point(-1,-1),0,0);
		cv::filter2D(im_f,layers_g[i], CV_32F,filter.t(),cv::Point(-1,-1),0,0);
	}
}
/*--------------------------------------------------------------*/
std::vector<float*>  Daisy::compute(cv::Mat& im, int* size_des, int R, int RQ, int TQ, int HQ){
    //std::chrono::time_point<std::chrono::system_clock> start, end;
	//start = std::chrono::system_clock::now();

	cv::Mat im_float;
	if(JUtil::isRGB(im)){
		std::cout<<"convert"<<std::endl;
		cv::cvtColor(im, im, CV_BGR2GRAY);
	}
	im.convertTo(im_float, CV_32F);
	im_float=im_float*(1.0/255.0);
	if(VERBOSE) std::cout<<"Computing grid"<<std::endl;
	std::vector<GridPoint> grid=computeGrid(R,RQ,TQ);
	if(VERBOSE) std::cout<<"Computing level sigmas"<<std::endl;
	std::vector<float> sigmas=computeLevelSigmas(R,RQ);
	std::vector<cv::Mat> L;
	if(VERBOSE) std::cout<<"1. Computing Layers"<<std::endl;
	layeredGradient(im_float,HQ,L);
	float sig_inc=sqrt((1.6*1.6)-(0.5*0.5));
	smoothLayers(L,sig_inc);
	if(VERBOSE) std::cout<<"2. Compute Cubes"<<std::endl;
	int n_sigmas=sigmas.size();
	float sigma=0;
	std::vector<std::vector<cv::Mat> > cubes(n_sigmas);
	for(int r=0;r<n_sigmas;r++){
		if(r==0){
			sigma=sigmas[r];
			copyLayers(L,cubes[r]);
		}
		else{
			sigma=sqrt(sigmas[r]*sigmas[r]-sigmas[r-1]*sigmas[r-1]);
			copyLayers(cubes[r-1],cubes[r]);
		}
		smoothLayers(cubes[r],sigma);
	}
	*size_des=grid.size()*HQ;
	if(VERBOSE) std::cout<<"3. Compute Daisy"<<std::endl;
	std::vector<float*> vec_des(im.rows*im.cols);
	for(int pos_i=0; pos_i<im.rows;pos_i++ ){
		for(int pos_j=0; pos_j<im.cols;pos_j++ ){
			vec_des[pos_i*im.cols+pos_j]=computeDaisy(grid, HQ, cubes, im.rows, im.cols, pos_j,pos_i);
		}
	}
	//end = std::chrono::system_clock::now();
	//auto elapsed =std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout<<"Time Daisy:"<<elapsed.count()<<std::endl;
	return vec_des;
}
/*--------------------------------------------------------------*/
float* Daisy::compute( cv::Mat& im,  int pos_i, int pos_j, int* size_des, int R, int RQ, int TQ, int HQ){
    /*std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();*/
	cv::Mat im_float;
	if(JUtil::isRGB(im)){
		cv::cvtColor(im, im, CV_BGR2GRAY);
	}
	im.convertTo(im_float, CV_32F);
	im_float=im_float*(1.0/255.0);
	if(VERBOSE) std::cout<<"Computing grid"<<std::endl;
	std::vector<GridPoint> grid=computeGrid(R,RQ,TQ);
	if(VERBOSE) std::cout<<"Computing level sigmas"<<std::endl;
	std::vector<float> sigmas=computeLevelSigmas(R,RQ);
	std::vector<cv::Mat> L;
	if(VERBOSE) std::cout<<"1. Computing Layers"<<std::endl;
	layeredGradient(im_float,HQ,L);
	float sig_inc=sqrt((1.6*1.6)-(0.5*0.5));
	smoothLayers(L,sig_inc);
	if(VERBOSE) std::cout<<"2. Compute Cubes"<<std::endl;
	int n_sigmas=sigmas.size();
	float sigma=0;
	std::vector<std::vector<cv::Mat> > cubes(n_sigmas);
	for(int r=0;r<n_sigmas;r++){
		if(r==0){
			sigma=sigmas[r];
			copyLayers(L,cubes[r]);
		}
		else{
			sigma=sqrt(sigmas[r]*sigmas[r]-sigmas[r-1]*sigmas[r-1]);
			copyLayers(cubes[r-1],cubes[r]);
		}
		smoothLayers(cubes[r],sigma);
	}

	*size_des=grid.size()*HQ;
	if(VERBOSE)std::cout<<"3. Compute Daisy"<<std::endl;
	float* des=computeDaisy(grid, HQ, cubes, im.rows, im.cols, pos_j,pos_i);
	/*end = std::chrono::system_clock::now();
	auto elapsed =std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"Time Daisy:"<<elapsed.count()<<std::endl;*/
	return des;
}
/*--------------------------------------------------------------*/
void Daisy::copyLayers(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst){
	dst.resize(src.size());
	for(size_t i=0; i<dst.size(); i++){
		src[i].copyTo(dst[i]);
	}
}
/*--------------------------------------------------------------*/
float* Daisy::computeDaisy(std::vector<GridPoint>& grid, int HQ,
		std::vector<std::vector<cv::Mat> >& cubes,
		int rows, int cols,int pos_x, int pos_y){
	int size_des=grid.size()*HQ;
	float *des=new float[size_des];
	for(int i=0; i<size_des;i++){
		des[i]=0;
	}
	float *des_g=NULL;

	float radius=0;
	float x, y=0;
	float wx=0, wy=0;
	int ix=0, iy=0, h=0;
	for(int g=0; g<(int)grid.size(); g++){
		radius=grid[g].getRadius();
		x=pos_x+grid[g].getComp_X();
		y=pos_y+grid[g].getComp_Y();
		des_g=des+g*HQ;
		ix=(int)x;
		iy=(int)y;
		wx=x-ix;
		wy=y-iy;
		//std::cout<<cubes.size()<<std::endl;
		//std::cout<<cubes[radius].size()<<std::endl;

	    if( 0 > iy || iy >= rows || 0>ix || ix >= cols ) continue;
		for(h=0; h<HQ;h++){
			des_g[h]=(1-wx)*(1-wy)*(cubes[radius][h]).at<float>(iy,ix);
		}
		if(iy>=0 && iy<rows && ix+1>=0 && ix+1<cols){
			for(h=0; h<HQ;h++){
				des_g[h]+=(wx)*(1-wy)*(cubes[radius][h]).at<float>(iy,ix+1);
			}
		}
		if(iy+1>=0 && iy+1<rows && ix>=0 && ix<cols){
			for(h=0; h<HQ;h++){
				des_g[h]+=(1-wx)*(wy)*(cubes[radius][h]).at<float>(iy+1,ix);
			}
		}
		if(iy+1>=0 && iy+1<rows && ix+1>=0 && ix+1<cols){
			for(h=0; h<HQ;h++){
				des_g[h]+=(wx)*(wy)*(cubes[radius][h]).at<float>(iy+1,ix+1);
			}
		}
		//partial normalization
		Preprocessing::normalizeVector(des_g,HQ,NORMALIZE_UNIT);
	}
	return des;
}
