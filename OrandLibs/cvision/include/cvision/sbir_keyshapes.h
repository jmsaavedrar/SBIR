/*
 * sbir_keyshapes.h
 *
 *  Created on: Mar 27, 2015
 *      Author: jsaavedr
 */

#ifndef CVISION_SBIR_KEYSHAPES_H_
#define CVISION_SBIR_KEYSHAPES_H_

#include "opencv2/highgui/highgui.hpp"
#include "jmsr/similaritysearch.h"
#include "flann/flann.hpp"
#include <vector>

class LKS_PARAMS{
private:
	int R; //radius
	int RQ; //radius  quantization
	int TQ; //angular quantization
	int HQ; //histogram quantization
	cv::Size patch_size; //patch size for computing daisy
	int n_clusters;//number of clusters
	bool SR; //squared normalization
	int block_size; //block_size;
	int knn; //k nearest-neighbors
	std::string vote; //P probability E exponencia
	int step; //step of edge points

public:
	LKS_PARAMS();
	LKS_PARAMS(int _R,
			int _RQ,
			int _TQ,
			int _HQ,
			cv::Size _patch_size,
			int _SR,
			int _block_size,
			int _knn, std::string _vote, int _step);
	void setValues(int _block_size,
				int _knn, std::string _vote, int _step);
	void create(char* vargs[], int nargs);
	void create(std::string str_input);
	void setNClusters(int k);
	int getDescriptorSize() const;
	int getR() const;
	int getRQ() const;
	int getHQ() const;
	int getTQ() const;
	cv::Size getPatchSize() const;
	bool getSR() const;
	int getBlockSize() const;
	int getKNN() const;
	int getNClusters() const;
	int getStep() const;
	std::string getVote() const;
	std::string toString() const;
};
class SBIR_Keyshapes {
private:
	SimilaritySearch<flann::L2<float> > *index;
	int dim_center;
	int size_des;
	int size_des_VLAD;
	float *keyshapes;
	float *stats; // min_dist, max_dist, mean_dist, std_dist, count
	LKS_PARAMS params;
public:
	SBIR_Keyshapes(std::string cluster_file, const LKS_PARAMS& params);
	//SBIR_Keyshapes(std::string cluster_file, cv::Size _patch_size, int _R, int _RQ, int _TQ, int _HQ);
	int getDescriptorSize();
	float *computeDescriptor(cv::Mat& image);
	int getVLADDescriptorSize();
	float *computeVLADDescriptor(cv::Mat& image);
	void classify_sketch_patch(float* des,std::vector<int>& classes, std::vector<float>& votes);
	void release();
	virtual ~SBIR_Keyshapes();

};

#endif /* CVISION_SBIR_KEYSHAPES_H_ */


