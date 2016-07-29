/*
 * lks_sbir.cpp
 *
 *  Created on: May 15, 2015
 *      Author: jsaavedr
 */

#include "lks_sbir.h"
#include "jmsr/JUtil.h"
#include "jmsr/preprocessing.h"
#include <vector>
#include <string>
#include <iostream>

/*-----------------------------------------------------------------------------------------------------*/
LKS_SBIR::LKS_SBIR(std::string _str_cluster_file, std::string _str_db_file, LKS_PARAMS _params):
str_cluster_file(_str_cluster_file),
str_db_file(_str_db_file), lks(NULL),indexLoaded(false), params(_params){
	sbir_index=NULL;
	linear_index=NULL;
}
/*-----------------------------------------------------------------------------------------------------*/
LKS_SBIR::LKS_SBIR(std::string str_input):
str_cluster_file(""),
str_db_file(""), lks(NULL),indexLoaded(false), params(){
	str_db_file=JUtil::getInputValue(str_input, "-lks_db");
	JUtil::jmsr_assert(!str_db_file.empty()," LKS_SBIR requires a lks_db, use -lks_db");
	str_cluster_file=JUtil::getInputValue(str_input, "-cluster");
	JUtil::jmsr_assert(!str_cluster_file.empty()," LKS_SBIR requires a cluster file, use -cluster file");
	params.create(str_input);
	sbir_index=NULL;
	linear_index=NULL;
}

/*-----------------------------------------------------------------------------------------------------*/
void LKS_SBIR::loadIndex(){
	std::vector<std::string> obj_classes;
	std::vector<std::string> obj_ids;
	int n_objects=0, size_des=0;
	float* des=JUtil::readDescriptors(str_db_file, &n_objects, &size_des, obj_ids, obj_classes);
	sbir_index=new SimilaritySearch<flann::L2<float> >(obj_ids, obj_classes, des, n_objects, size_des, J_LINEAR);
	linear_index=new JLinearSearch<JL2>(obj_ids, obj_classes, des, n_objects, size_des);
	lks=new SBIR_Keyshapes(str_cluster_file, params);
	std::cout<<"Index has been loaded!!"<<std::endl;
	std::cout<<"N_objects: "<<n_objects<<" "<<"Size_des:"<<size_des<<std::endl;
	indexLoaded=true;
}
/*-----------------------------------------------------------------------------------------------------*/
float* LKS_SBIR::getSBIRDescriptor(cv::Mat& image, int *size_des){
	float *des=NULL;
	//-------------------------------------------------------------- Preprocessing input image
	cv::Mat  image_out;
	Preprocessing::preprocess_sketch(image, image_out, 256,200);
	//cv::imshow("image_out", image_out);
	//cv::waitKey();
	//--------------------------------------------------------------
	*size_des=lks->getDescriptorSize();
	des=lks->computeDescriptor(image_out);
	return des;
}
/*-----------------------------------------------------------------------------------------------------*/
ResultQuery LKS_SBIR::searchBySimilarity(float* descriptor, int size_des, int INDEX_TYPE, int K, std::string subset_str){
	std::vector<std::string> subset=Preprocessing::splitString(subset_str, '\t');
	return searchBySimilarity(descriptor, size_des, INDEX_TYPE, K, subset);
}
/*-----------------------------------------------------------------------------------------------------*/
ResultQuery LKS_SBIR::searchBySimilarity(float* descriptor, int size_des, int INDEX_TYPE, int K, std::vector<std::string> v_subset_str){
	ResultQuery result;
	if(indexLoaded){
		if(INDEX_TYPE==J_INDEX_FLANN)
		{
			result=sbir_index->searchBySimilarity(descriptor,size_des, K);
		}
		else if(INDEX_TYPE==J_INDEX_LINEAR)
		{
			if(v_subset_str.empty())
				result=linear_index->searchBySimilarity(descriptor,size_des, K);
			else
			{
				result=linear_index->searchBySimilarity(descriptor,size_des, v_subset_str, K);
			}
		}
		else
		{
			std::cerr<<"ERROR: Incorrect index type for querying"<<std::endl;
			exit(EXIT_FAILURE);
		}
	}
	else{
		std::cout<<"WARNING: LKS indexes were not loaded, result_query is empty!!!"<<std::endl;
	}
	return result;
}
/*-----------------------------------------------------------------------------------------------------*/
void LKS_SBIR::release(){
	std::cout<<"releasing LKS "<<std::endl;
	if(indexLoaded){
		sbir_index->releaseIndex();
		linear_index->releaseIndex();
		lks->release();
	}
	std::cout<<"Ok -> releasing LKS "<<std::endl;
}
/*-----------------------------------------------------------------------------------------------------*/
LKS_SBIR::~LKS_SBIR() {
	// TODO Auto-generated destructor stub
}


