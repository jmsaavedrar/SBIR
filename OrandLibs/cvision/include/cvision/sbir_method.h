/*
 * sbir_method.h
 *
 *  Created on: May 15, 2015
 *      Author: jsaavedr
 */

#ifndef CVISION_SBIR_METHOD_H_
#define CVISION_SBIR_METHOD_H_

#include "opencv2/highgui/highgui.hpp"
#include "jmsr/similaritysearch.h"
#include <string>


class SBIR_Method {
protected:
	 SimilaritySearch<flann::L2<float> > *sbir_index; /*!< A similaritySearch object. It is the index */
	 JLinearSearch<JL2> *linear_index; /*<! A linear search index >*/
	 std::map<std::string, int> class_idx_map; //map a class name with an int idx
	 std::vector<std::string> namesByClass; //a vector whose slot "i" stores a string with product names belonging to class "i"
public:
	SBIR_Method();
	virtual ~SBIR_Method();
	virtual float* getSBIRDescriptor(cv::Mat& image, int *size_des)=0;
	virtual ResultQuery searchBySimilarity(float* query, int query_size, int INDEX_TYPE, int K, std::vector<std::string> v_subsset_str)=0;
	virtual ResultQuery searchBySimilarity(float* query, int query_size, int INDEX_TYPE, int K, std::string subset_str="")=0;
	virtual void release()=0;
	virtual void loadIndex()=0;
	virtual std::string getName(int idx);
	virtual std::string getClass(int idx);
	virtual int getSize();

	ResultQuery doMainSearch(cv::Mat& query_image, int index_type, int K, bool withFlip, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doMainSearch(float* query, int query_size, int index_type, int K, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doMainSearch(float* query, float* query_flip, int query_size, int index_type, int K, std::vector<std::string> v_subset_str=std::vector<std::string>());

	ResultQuery doSearch(const std::string& query, int K=-1, bool withFlip=false, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doSearch(cv::Mat& query, int K=-1, bool withFlip=false, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doSearch(float* query, int query_size, int K=-1, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doSearch(float* query, float* query_flip, int query_size, int K=-1, std::vector<std::string> v_subset_str=std::vector<std::string>());

	ResultQuery doJLinearSearch(const std::string& query, int K=-1, bool withFlip=false, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doJLinearSearch(cv::Mat& query, int K=-1, bool withFlip=false, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doJLinearSearch(float* query, int query_size, int K=-1, std::vector<std::string> v_subset_str=std::vector<std::string>());
	ResultQuery doJLinearSearch(float* query, float* query_flip, int query_size, int K=-1, std::vector<std::string> v_subset_str=std::vector<std::string>());

	std::string getIdsByClasses(const std::string& classes_str);
	static bool hasEnoughStrokes(cv::Mat& query);
	static void freeDescriptor(float* descriptor);
};

#endif /* CVISION_SBIR_METHOD_H_ */
