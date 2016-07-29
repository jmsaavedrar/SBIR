/*
 * sbir_keyshapes.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: jsaavedr
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "sbir_keyshapes.h"
#include "clustering.h"
#include "daisy.h"
#include "jmsr/preprocessing.h"
#include "jmsr/JUtil.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#define sign(x) ( ( 0 < x ) - ( x < 0 ) )


/*------------------------------------------------------------------------------*/
LKS_PARAMS::LKS_PARAMS():R(15), RQ(2), TQ(8), HQ(8), patch_size(31,31),
	SR(true), block_size(2), knn(10), vote("P"), step(1) {

}
/*------------------------------------------------------------------------------*/
LKS_PARAMS::LKS_PARAMS(int _R,
		int _RQ,
		int _TQ,
		int _HQ,
		cv::Size _patch_size,
		int _SR,
		int _block_size,
		int _knn, std::string _vote, int _step):R(_R), RQ(_RQ),
				TQ(_TQ), HQ(_HQ),
				patch_size(_patch_size), n_clusters(-1),
				SR(_SR), block_size(_block_size), knn(_knn), vote(_vote), step(_step) {

}
/*------------------------------------------------------------------------------*/
void LKS_PARAMS::create(char* vargs[], int nargs){
	int i=0;
	while(i<nargs){
		if(strcmp(vargs[i],"-SR")==0){
			i++;
			if(i<nargs && strcmp(vargs[i],"false")==0) SR=false;
			else if(i<nargs && strcmp(vargs[i],"true")==0) SR=true;
		}
		if(strcmp(vargs[i],"-BLOCK")==0){
			i++;
			if(i<nargs) block_size=atoi(vargs[i]);
		}
		if(strcmp(vargs[i],"-KNN")==0){
			i++;
			if(i<nargs) knn=atoi(vargs[i]);
		}
		if(strcmp(vargs[i],"-VOTE")==0){
			i++;
			if(i<nargs) vote=std::string(vargs[i]);
		}
		if(strcmp(vargs[i],"-STEP")==0){
			i++;
			if(i<nargs) step=atoi(vargs[i]);
		}
		i++;
	}
}
/*------------------------------------------------------------------------------*/
void LKS_PARAMS::create(std::string str_input){
	std::string str_sr=JUtil::getInputValue(str_input, "-SR");
	if(!str_sr.empty()){
		if(str_sr.compare("true")==0) SR=true;
		else if(str_sr.compare("false")==0) SR=false;
	}
	std::string str_block=JUtil::getInputValue(str_input, "-BLOCK");
	if(!str_block.empty()) block_size=atoi(str_block.c_str());
	std::string str_knn=JUtil::getInputValue(str_input, "-KNN");
	if(!str_knn.empty()) knn=atoi(str_knn.c_str());
	vote=JUtil::getInputValue(str_input, "-VOTE");
	std::string str_step=JUtil::getInputValue(str_input, "-STEP");
	if(!str_step.empty()) step=atoi(str_step.c_str());
	std::string str_n_clusters=JUtil::getInputValue(str_input, "-N_CLUSTERS");
	if(!str_n_clusters.empty()) n_clusters=atoi(str_n_clusters.c_str());
}

void LKS_PARAMS::setValues(int _block_size,
				int _knn, std::string _vote, int _step){
	block_size=_block_size;
	knn=_knn;
	vote=_vote;
	step=_step;
}
/*------------------------------------------------------------------------------*/
void LKS_PARAMS::setNClusters(int k){
	n_clusters=k;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getDescriptorSize() const{
	return block_size*block_size*(n_clusters);
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getR() const{
	return R;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getRQ() const{
	return RQ;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getTQ() const{
	return TQ;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getHQ() const{
	return HQ;
}
/*------------------------------------------------------------------------------*/
cv::Size LKS_PARAMS::getPatchSize() const{
	return patch_size;
}
/*------------------------------------------------------------------------------*/
bool LKS_PARAMS::getSR() const{
	return SR;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getBlockSize() const{
	return block_size;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getKNN() const{
	return knn;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getNClusters() const{
	return n_clusters;
}
/*------------------------------------------------------------------------------*/
std::string LKS_PARAMS::getVote() const{
	return vote;
}
/*------------------------------------------------------------------------------*/
int LKS_PARAMS::getStep() const{
	return step;
}
/*------------------------------------------------------------------------------*/
std::string LKS_PARAMS::toString() const{
	std::string str("");
	str+="R: "+ std::to_string(R)+" ";
	str+="RQ: "+ std::to_string(RQ)+" ";
	str+="TQ: "+ std::to_string(TQ)+" ";
	str+="HQ: "+ std::to_string(HQ)+" ";
	str+="PatchSize: "+ std::to_string(patch_size.width)+" ";
	str+="SR: "+ std::to_string(SR)+" ";
	str+="BLOCK: "+ std::to_string(block_size)+" ";
	str+="KNN: "+ std::to_string(knn)+" ";
	str+="VOTE: "+ vote+" ";
	str+="STEP: "+ std::to_string(step) + " ";
	str+="K_Clusters: "+ std::to_string(n_clusters);
	return str;
}
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
SBIR_Keyshapes::SBIR_Keyshapes(std::string cluster_file, const LKS_PARAMS& _params):params(_params){
	int n_clusters;
    keyshapes=Clustering::readClusters(cluster_file, &n_clusters, &dim_center);
    int n_std_rows=0, n_std_cols=0;
    stats=JUtil::readMat(cluster_file+".stats", CV_32F, &n_std_rows, &n_std_cols);
    assert(n_std_rows==n_clusters && n_std_cols==5);
    params.setNClusters(n_clusters);
    index=new SimilaritySearch<flann::L2<float> >(keyshapes, params.getNClusters(), dim_center, J_LINEAR);
    size_des=params.getNClusters();
    size_des_VLAD=params.getNClusters()*dim_center;
    std::cout<<"Cluster index was loaded OK"<<std::endl;
}
/*------------------------------------------------------------------------------*/
int SBIR_Keyshapes::getDescriptorSize(){
	return params.getDescriptorSize();
}
int SBIR_Keyshapes::getVLADDescriptorSize(){
	return params.getDescriptorSize()*dim_center;
}
/*------------------------------------------------------------------------------*/
void SBIR_Keyshapes::classify_sketch_patch(float* patch_des, std::vector<int>& classes, std::vector<float>& votes){
	ResultQuery rst;
	//float *descriptor=Descriptor.getDaysi();
	rst=index->searchBySimilarity(patch_des, dim_center, params.getKNN());
	classes.resize(params.getKNN());
	votes.resize(params.getKNN());
	float sum_dists=0;
	for(int i=0; i<params.getKNN();i++){
		classes[i]=rst.idx[i];
		sum_dists+=rst.distances[i];
	}
	if(params.getKNN()==1){
		votes[0]=1;

	}else
	{	if(params.getVote().compare("P")==0){ //using simple probs
			for(int i=0; i<params.getKNN();i++){
				votes[i]=1-rst.distances[i]/sum_dists;
			}
		}
		if(params.getVote().compare("E")==0){ //Using Exponential function with fixe sigma
			float alfa=2.0;
			for(int i=0; i<params.getKNN();i++){
				votes[i]=std::exp(-1*alfa*rst.distances[i]);
			}
		}
		if(params.getVote().compare("K")==0){ ///using gaussian kernel
			float sum_probs=0;
			for(int i=0; i<params.getKNN();i++){
				votes[i]=JUtil::getKernelProbDist(rst.distances[i], 0.0, stats[classes[i]*5+1]);
				sum_probs+=votes[i];
				//std::cout<<classes[i]<<" "<<votes[i]<<" "<<rst.distances[i]<<std::endl;
			}
			for(int i=0; i<params.getKNN();i++){
				votes[i]=votes[i]/sum_probs;
			}
		}
	}
}
/*------------------------------------------------------------------------------*/
float *SBIR_Keyshapes::computeDescriptor(cv::Mat& image){
	assert(image.type()==CV_8UC1);
	cv::Mat bin_image;
	cv::threshold(image,bin_image, 150, 1,cv::THRESH_BINARY_INV);
	int patch_des_size;
	//------pre-processing using morphological
	//cv::Mat se=cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
	//cv::morphologyEx(bin_image, bin_image, cv::MORPH_DILATE, se);
	//descriptor computation-------------------------------
	std::vector<float*> vec_des=Daisy::compute(bin_image,&patch_des_size,
			params.getR(), params.getRQ(), params.getTQ(), params.getHQ());
	//-----------------------------------------------------
	std::vector<cv::Point2d> edge_points=Preprocessing::getPointsByValue(bin_image, 1);
	std::vector<int> id_clusters;
	std::vector<float> votes;
	int pos_patch=0;

	cv::Mat patch;
	int block_size=params.getBlockSize();
	std::vector<float*> block_des(block_size*block_size);
	//------------------------------Initialization of block descriptors
	for(int b=0; b<block_size*block_size; b++){
		block_des[b]=new float[size_des];
		for(int i=0; i<size_des;i++) block_des[b][i]=0;
	}
	//-------------------------------Processing using spatial division
	float x_block=0, y_block=0;
	int x_izq, x_der, y_izq, y_der;
	float w_x_izq, w_x_der, w_y_izq, w_y_der;
	int idx_block=0;
	float weight=0;
	for(size_t i=0; i<edge_points.size();i+=params.getStep()){
		//patch=Preprocessing::getPatch(bin_image, edge_points[i].y, edge_points[i].x, patch_size);
		pos_patch=edge_points[i].y*image.cols+edge_points[i].x;
		classify_sketch_patch(vec_des[pos_patch], id_clusters, votes);

		x_block=(edge_points[i].x/static_cast<float>(bin_image.cols))*block_size;
		y_block=(edge_points[i].y/static_cast<float>(bin_image.rows))*block_size;
		Preprocessing::linearInterBIN(x_block, &x_izq, &x_der, &w_x_izq, &w_x_der);
		Preprocessing::linearInterBIN(y_block, &y_izq, &y_der, &w_y_izq, &w_y_der);

		for(int id_k=0; id_k<params.getKNN(); id_k++){

			if(JUtil::isValid(x_izq, y_izq, 0, block_size-1,0, block_size-1)){
				idx_block=y_izq*block_size+x_izq;
				weight=w_x_izq*w_y_izq;
				block_des[idx_block][id_clusters[id_k]]+=votes[id_k]*weight;
			}
			if(JUtil::isValid(x_izq, y_der, 0, block_size-1,0, block_size-1)){
				idx_block=y_der*block_size+x_izq;
				weight=w_x_izq*w_y_der;
				block_des[idx_block][id_clusters[id_k]]+=votes[id_k]*weight;
			}
			if(JUtil::isValid(x_der, y_izq, 0, block_size-1,0, block_size-1)){
				idx_block=y_izq*block_size+x_der;
				weight=w_x_der*w_y_izq;
				block_des[idx_block][id_clusters[id_k]]+=votes[id_k]*weight;
			}
			if(JUtil::isValid(x_der, y_der, 0, block_size-1,0, block_size-1)){
				idx_block=y_der*block_size+x_der;
				weight=w_x_der*w_y_der;
				block_des[idx_block][id_clusters[id_k]]+=votes[id_k]*weight;
			}
		}
		//cv::imshow("patch",patch*255);
		//cv::waitKey();
	}
	//----------------------Concatenating all descriptors
	float *des=new float[block_size*block_size*size_des];
	for(int b=0; b<block_size*block_size; b++){
		Preprocessing::normalizeVector(block_des[b],size_des,NORMALIZE_UNIT);
		std::copy(block_des[b],block_des[b]+size_des, des+b*size_des);
		delete[] block_des[b];
	}
	//----------------------Squared-Root Normalization
	if(params.getSR()){
		for(int i=0; i<block_size*block_size*size_des; i++){
			des[i]=std::sqrt(des[i]);
		}
	}
	//---------------------- cleaning vec_des
	for(size_t i=0; i<vec_des.size();i++){
		delete[] vec_des[i];
	}
	vec_des.clear();
	return des;
}
/*------------------------------------------------------------------------------*/
float *SBIR_Keyshapes::computeVLADDescriptor(cv::Mat& image){
	assert(image.type()==CV_8UC1);
	cv::Mat bin_image;
	cv::threshold(image,bin_image, 150, 1,cv::THRESH_BINARY_INV);
	int patch_des_size;
	//------pre-processing using morphological
	//cv::Mat se=cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
	//cv::morphologyEx(bin_image, bin_image, cv::MORPH_DILATE, se);
	//descriptor computation-------------------------------
	std::vector<float*> vec_des=Daisy::compute(bin_image,&patch_des_size,
			params.getR(), params.getRQ(), params.getTQ(), params.getHQ());
	//-----------------------------------------------------
	std::vector<cv::Point2d> edge_points=Preprocessing::getPointsByValue(bin_image, 1);
	std::vector<int> id_clusters;
	std::vector<float> votes;
	int pos_patch=0;

	cv::Mat patch;
	int block_size=params.getBlockSize();
	std::vector<float*> block_des(block_size*block_size);
	//------------------------------Initialization of block descriptors
	for(int b=0; b<block_size*block_size; b++){
		block_des[b]=new float[size_des_VLAD];
		for(int i=0; i<size_des_VLAD;i++) block_des[b][i]=0;
	}
	//-------------------------------Processing using spatial division
	float x_block=0, y_block=0;
	int x_izq, x_der, y_izq, y_der;
	float w_x_izq, w_x_der, w_y_izq, w_y_der;
	int idx_block=0;
	float weight=0;
	float* center;
	for(size_t i=0; i<edge_points.size();i+=params.getStep()){
		//patch=Preprocessing::getPatch(bin_image, edge_points[i].y, edge_points[i].x, patch_size);
		pos_patch=edge_points[i].y*image.cols+edge_points[i].x;
		classify_sketch_patch(vec_des[pos_patch], id_clusters, votes);

		x_block=(edge_points[i].x/static_cast<float>(bin_image.cols))*block_size;
		y_block=(edge_points[i].y/static_cast<float>(bin_image.rows))*block_size;
		Preprocessing::linearInterBIN(x_block, &x_izq, &x_der, &w_x_izq, &w_x_der);
		Preprocessing::linearInterBIN(y_block, &y_izq, &y_der, &w_y_izq, &w_y_der);

		for(int id_k=0; id_k<params.getKNN(); id_k++){

			center=keyshapes+id_clusters[id_k]*dim_center;
			if(JUtil::isValid(x_izq, y_izq, 0, block_size-1,0, block_size-1)){
				idx_block=y_izq*block_size+x_izq;
				weight=w_x_izq*w_y_izq;
				for(int d=0; d<dim_center;d++){
					block_des[idx_block][id_clusters[id_k]*dim_center+d]+=(vec_des[pos_patch][d]-center[d])*votes[id_k]*weight;
				}
			}
			if(JUtil::isValid(x_izq, y_der, 0, block_size-1,0, block_size-1)){
				idx_block=y_der*block_size+x_izq;
				weight=w_x_izq*w_y_der;
				for(int d=0; d<dim_center;d++){
					block_des[idx_block][id_clusters[id_k]*dim_center+d]+=(vec_des[pos_patch][d]-center[d])*votes[id_k]*weight;
				}
			}
			if(JUtil::isValid(x_der, y_izq, 0, block_size-1,0, block_size-1)){
				idx_block=y_izq*block_size+x_der;
				weight=w_x_der*w_y_izq;
				for(int d=0; d<dim_center;d++){
					block_des[idx_block][id_clusters[id_k]*dim_center+d]+=(vec_des[pos_patch][d]-center[d])*votes[id_k]*weight;
				}
			}
			if(JUtil::isValid(x_der, y_der, 0, block_size-1,0, block_size-1)){
				idx_block=y_der*block_size+x_der;
				weight=w_x_der*w_y_der;
				for(int d=0; d<dim_center;d++){
					block_des[idx_block][id_clusters[id_k]*dim_center+d]+=(vec_des[pos_patch][d]-center[d])*votes[id_k]*weight;
				}
			}
		}
		//cv::imshow("patch",patch*255);
		//cv::waitKey();
	}
	//----------------------Concatenating all descriptors
	float *des=new float[block_size*block_size*size_des_VLAD];
	for(int b=0; b<block_size*block_size; b++){
		Preprocessing::normalizeVector(block_des[b],size_des_VLAD,NORMALIZE_UNIT);
		std::copy(block_des[b],block_des[b]+size_des_VLAD, des+b*size_des_VLAD);
		delete[] block_des[b];
	}
	//----------------------Squared-Root Normalization
	if(params.getSR()){
		for(int i=0; i<block_size*block_size*size_des_VLAD; i++){
			des[i]=sign(des[i])*std::pow(std::abs(des[i]), 0.5);
			//std::cout<<des[i]<<" ";
		}
		//std::cout<<std::endl;
	}
	//---------------------- cleaning vec_des
	for(size_t i=0; i<vec_des.size();i++){
		delete[] vec_des[i];
	}
	vec_des.clear();
	return des;
}
/*------------------------------------------------------------------------------*/
SBIR_Keyshapes::~SBIR_Keyshapes() {

}
void SBIR_Keyshapes::release(){
	index->releaseIndex();
	delete index;
	delete[] stats;
}
