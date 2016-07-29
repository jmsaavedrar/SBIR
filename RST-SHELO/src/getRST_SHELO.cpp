/*
 * getRST_SHELO.cpp
 * Author: Jose Saavedra
 * Copyright Orand S.A.
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jmsr/preprocessing.h"
#include "jmsr/descriptor.h"
#include "jmsr/JUtil.h"
#include <cstdio>
#include <iostream>


float* getSBIRDescriptor(cv::Mat& image, int *size_des, bool draw){	
	HELOParams params(25,36,6,NORMALIZE_ROOT_UNIT,true);	
	*size_des=params.getDescriptorSize();
	float* descriptor=NULL;
	if (!draw){
		descriptor=Descriptor::getLocalHELODescriptor(image,params);
	}
	else{
		cv::Mat mat_rst;
		std::string str_output("shelo_out.png");
		descriptor=Descriptor::getLocalHELODescriptor(image, params, mat_rst);
		cv::imshow("shelo", mat_rst);
		cv::imwrite(str_output,mat_rst);
	}
	return descriptor;
}

int main(int nargs, char* vargs[] ){
	if (nargs < 2 ){
		std::cout<<"Usage "<<std::endl;
		std::cout<<"      "<<vargs[0]<<" -f=<filename> -draw=[true|false]"
		<<std::endl;
		exit(EXIT_FAILURE);
	}
	
	std::string str_input=JUtil::lineArgumentsToString(nargs, vargs);
	std::string str_file=JUtil::getInputValue(str_input, "-f");
	JUtil::jmsr_assert(!str_file.empty(), " file is required use -f=<filename>");
	std::string str_draw=JUtil::getInputValue(str_input, "-draw");
	bool draw=false;
	if( str_draw == "1" || str_draw == "true") draw=true;
	cv::Mat mat_image=cv::imread(str_file);
	if(mat_image.channels()==3){
		cv::cvtColor(mat_image, mat_image, CV_BGR2GRAY);
	}
	int des_size;
	float* des=getSBIRDescriptor(mat_image, &des_size, draw);
	for(int i=0; i<des_size-2; i++){
		std::cout<<des[i]<<" ";
	}
	std::cout<<des[des_size]<<std::endl;
	delete[] des;
	return 0;
}





