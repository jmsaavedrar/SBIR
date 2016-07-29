/*
 * lks_sbir.h
 *
 *  Created on: May 15, 2015
 *      Author: jsaavedr
 */

#ifndef CVISION_LKS_SBIR_H_
#define CVISION_LKS_SBIR_H_

#include "sbir_method.h"
#include "sbir_keyshapes.h"
#include "jmsr/similaritysearch.h"

class LKS_SBIR: public SBIR_Method {
private:
	std::string str_cluster_file;
	std::string str_db_file;
	SBIR_Keyshapes *lks;
	bool indexLoaded;
	LKS_PARAMS params;
public:
	LKS_SBIR(std::string _str_cluster_file, std::string _str_db_file, LKS_PARAMS _params);
	LKS_SBIR(std::string _str_input);
	virtual ~LKS_SBIR();
	// functions from MSBIR_Method that will be re-implemented
	float* getSBIRDescriptor(cv::Mat& image, int *size_des);
	ResultQuery searchBySimilarity(float* descriptor, int size_des, int INDEX_TYPE, int K, std::string subset_str="");
	ResultQuery searchBySimilarity(float* descriptor, int size_des, int INDEX_TYPE, int K, std::vector<std::string> v_subset_str);
	void release();
	void loadIndex();
};

#endif /* CVISION_LKS_SBIR_H_ */
