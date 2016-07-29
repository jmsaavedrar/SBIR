/*
 * sbir_method.cpp
 *
 *  Created on: May 15, 2015
 *      Author: jsaavedr
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jmsr/preprocessing.h"
#include "jmsr/JUtil.h"
#include "sbir_method.h"
#include <iostream>
#define SBIR_METHOD_MSG(msg) std::cout<<"SBIR_METHOD: "<<msg<<std::endl;

SBIR_Method::SBIR_Method() {
	// TODO Auto-generated constructor stub

}

SBIR_Method::~SBIR_Method() {
	// TODO Auto-generated destructor stub
}

/*----------------------------------------------------------------*/
/*! \fn doJLinearSearch(std::string query, int K=-1) using JLinearSearch index
    \brief Searching the database w.r.t. a query image
    \param query The complete name of the query image
    \param K For k-nn search, K=-1 all dataset is retrieved
*/
ResultQuery SBIR_Method::doJLinearSearch(const std::string& query, int K, bool withFlip, std::vector<std::string> v_subset_str){
    cv::Mat image=cv::imread(query);
    return doJLinearSearch(image, K, withFlip,v_subset_str);
}
/*----------------------------------------------------------------*/
/*! \fn doJLinearSearch(cv::Mat query, int K=-1) using JLINEAR index
    \brief Searching the database w.r.t. a query image
    \param query a Mat object
    \param K For k-nn search, K=-1 all dataset is retrieved
*/
ResultQuery SBIR_Method::doJLinearSearch(cv::Mat& query, int K, bool withFlip, std::vector<std::string> v_subset_str){
	return doMainSearch(query,  J_INDEX_LINEAR, K,  withFlip,v_subset_str);
}
/*----------------------------------------------------------------*/
ResultQuery SBIR_Method::doJLinearSearch(float* query, int query_size, int K, std::vector<std::string> v_subset_str){
    return doMainSearch(query, query_size, J_INDEX_LINEAR, K, v_subset_str);
}
/*----------------------------------------------------------------*/
ResultQuery SBIR_Method::doJLinearSearch(float* query, float* query_flip, int query_size, int K, std::vector<std::string> v_subset_str){
    return doMainSearch(query, query_flip, query_size, J_INDEX_LINEAR, K, v_subset_str);
}
/*----------------------------------------------------------------*/
/*! \fn doSearch(std::string query, int K=-1)
    \brief Searching the database w.r.t. a query image using FLANN index
    \param query The complete name of the query image
    \param K For k-nn search, K=-1 all dataset is retrieved
*/
ResultQuery SBIR_Method::doSearch(const std::string& query, int K, bool withFlip, std::vector<std::string> v_subset_str)
{
    cv::Mat image=cv::imread(query);
    return doSearch(image, K, withFlip, v_subset_str);
}
/*----------------------------------------------------------------*/
/*! \fn doSearch(cv::Mat query, int K=-1) using FLANN index
    \brief Searching the database w.r.t. a query image
    \param query a Mat object
    \param K For k-nn search, K=-1 all dataset is retrieved
*/
ResultQuery SBIR_Method::doSearch(cv::Mat& query, int K, bool withFlip, std::vector<std::string> v_subset_str)
{
	return doMainSearch(query, J_INDEX_FLANN, K,  withFlip, v_subset_str);
}
/*----------------------------------------------------------------*/
ResultQuery SBIR_Method::doSearch(float* query, int query_size, int K, std::vector<std::string> v_subset_str)
{
	return doMainSearch(query, query_size, J_INDEX_FLANN, K, v_subset_str);

}
/*----------------------------------------------------------------*/
ResultQuery SBIR_Method::doSearch(float* query, float* query_flip, int query_size, int K, std::vector<std::string> v_subset_str)
{
	return doMainSearch(query, query_flip, query_size, J_INDEX_FLANN, K, v_subset_str);
}
/*----------------------------------------------------------------*/
ResultQuery SBIR_Method::doMainSearch(float* query, int query_size, int index_type, int K, std::vector<std::string> v_subset_str)
{
    //This function requires processing in a non-flipping mode
	if (K==-1) K=getSize();
	if (!v_subset_str.empty()) index_type=J_INDEX_LINEAR;
	ResultQuery result=searchBySimilarity(query,query_size,index_type, K, v_subset_str);
	std::cout<<"SBIR_METHOD: msbir method ready for search OK "<<result.getSize()<<std::endl;
	return result;
}

/*----------------------------------------------------------------*/
ResultQuery SBIR_Method::doMainSearch(float* query, float* query_flip, int query_size, int index_type, int K, std::vector<std::string> v_subset_str)
{
    //This function requires processing in a flipping mode
	std::cout<<"--> doMainSearch"<<std::endl;
	if (K==-1) K=getSize();
	if (!v_subset_str.empty()) index_type=J_INDEX_LINEAR;

	std::cout<<"SBIR_METHOD: searching (1)"<<std::endl;
    ResultQuery result=searchBySimilarity(query,query_size, index_type, -1, v_subset_str);
    std::cout<<"RST="<<result.getSize()<<std::endl;

    std::cout<<"SBIR_METHOD: searching (flip)"<<std::endl;
    ResultQuery result_flip=searchBySimilarity(query_flip,query_size, index_type, -1, v_subset_str);
    std::cout<<"SBIR_METHOD: RST_FLIP="<<result_flip.getSize()<<std::endl;

    std::cout<<"SBIR_METHOD: Merging for flipped image"<<std::endl;
	if(v_subset_str.empty())
		result=ResultQuery::merge(result, result_flip,K);
	else
		result=ResultQuery::mergeWithSubset(result, result_flip,K);
	std::cout<<"OK SBIR::doSearch"<<std::endl;
	return result;
}

/*----------------------------------------------------------------*/
ResultQuery SBIR_Method::doMainSearch(cv::Mat& query_image, int index_type, int K,  bool withFlip, std::vector<std::string> v_subset_str)
{
    if(query_image.empty()){
    	std::cout<<"ERROR in SBIR_METHOD: The query name is incorrect!!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    if (K==-1) K=getSize();
    int query_size=0;
    cv::Mat image_rgb;
    //working in grayscale space
    if(query_image.channels()==3){
        cv::cvtColor(query_image,image_rgb,CV_BGR2GRAY);
    }
    else{
    	query_image.copyTo(image_rgb);
    }
    float* query=getSBIRDescriptor(image_rgb, &query_size);
    ResultQuery result;
    if(!withFlip){
    	result=doMainSearch(query, query_size, index_type, K, v_subset_str);
    	std::cout<<"SBIR_METHOD: releasing memory for query"<<std::endl;
    	freeDescriptor(query);
    	return result;
    }
    else
    { // It requires to retrieve the complete dataset (K=-1)
    	std::cout<<"SBIR_METHOD: flipping image"<<std::endl;
    	cv::Mat image_flip;
        cv::flip(image_rgb,image_flip, 1);
        int query_size_flip;
        float* query_flip=getSBIRDescriptor(image_rgb, &query_size_flip);
        JUtil::jmsr_assert(query_size==query_size_flip, "query_size != query_size_flip");
        result=doMainSearch(query, query_flip, query_size, index_type, K, v_subset_str);
        std::cout<<"SBIR_METHOD: releasing memory for query"<<std::endl;
        freeDescriptor(query);
        std::cout<<"SBIR_METHOD: releasing memory for query_flip"<<std::endl;
        freeDescriptor(query_flip);
        return result;
    }
}
/*----------------------------------------------------------------*/
std::string SBIR_Method::getName(int idx){ return sbir_index->getName(idx);}
std::string SBIR_Method::getClass(int idx){ return sbir_index->getClass(idx);}
int SBIR_Method::getSize() {
	std::cout<<"executing this size"<<std::endl;
	return sbir_index->getDBSize();
}
/*----------------------------------------------------------------*/
std::string SBIR_Method::getIdsByClasses(const std::string& classes_str){
	std::vector<std::string> classes_vec=Preprocessing::splitString(classes_str, ' ');
	std::string ids_str("");
	for(unsigned int i=0; i<classes_vec.size(); i++)
	{
		std::cout<<"clase: "<<classes_vec[i]<<std::endl;
		if(class_idx_map.find(classes_vec[i])!=class_idx_map.end())
		{
			ids_str=ids_str+"\t"+namesByClass[class_idx_map[classes_vec[i]]];
		}
	}
	return ids_str;
}
/*----------------------------------------------------------------*/
bool SBIR_Method::hasEnoughStrokes(cv::Mat& query){
	cv::Mat bin;
	cv::threshold(query, bin, 150,1,cv::THRESH_BINARY_INV);
	if(cv::sum(bin)[0]<10) return false;
	else return true;
}
/*----------------------------------------------------------------*/
void SBIR_Method::freeDescriptor(float* descriptor){
	delete[] descriptor;
}
