/*
 * getLKSDescriptor.cpp
 *
 *  Created on: Oct 7, 2015
 *  Author: Jose Saavedra
 *  Copyright Orand S.A.
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvision/sbir_keyshapes.h"
#include "jmsr/JUtil.h"
#include <cstdio>
#include <iostream>

int main(int nargs, char* vargs[] ){
	if (nargs < 2 ){
		std::cout<<"Usage "<<std::endl;
		std::cout<<"      "<<vargs[0]<<" -cluster=<cluster> -f=<image>"<<std::endl;
		exit(EXIT_FAILURE);
	}
	std::string str_input=JUtil::lineArgumentsToString(nargs, vargs);
	std::string str_cluster=JUtil::getInputValue(str_input, "-cluster");
	JUtil::jmsr_assert(!str_cluster.empty()," cluster file is required, use -cluster=<cluster>");
	std::string str_image=JUtil::getInputValue(str_input, "-f");
	JUtil::jmsr_assert(!str_image.empty()," an image is required, use -f=<image>!");
	std::cout<<str_image<<std::endl;
	std::cout<<str_cluster<<std::endl;
	LKS_PARAMS params(15,2,8,8,cv::Size(31,31), true, 4,10,"K",1);
	SBIR_Keyshapes lks(str_cluster, params);
	std::cout<<"SBIR_Keyshapes OK"<<std::endl;
	cv::Mat mat_image=cv::imread(str_image);
	JUtil::jmsr_assert(!mat_image.empty(), str_image + " does not exist!");
	if(JUtil::isRGB(mat_image)){
		cv::cvtColor(mat_image,mat_image, CV_BGR2GRAY);
	}
	float *lks_des=lks.computeDescriptor(mat_image);
	std::cout<<"LKS Descriptor>>"<<std::endl;
	for(int i=0; i<lks.getDescriptorSize(); i++){
		std::cout<<lks_des[i]<<" ";
	}
	std::cout<<"<<"<<std::endl;
	delete[] lks_des;
	return 0;
}





