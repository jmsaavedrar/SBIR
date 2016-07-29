/*
ccomponent.cpp
Copyright (C) Jose M. Saavedra
ORAND S.A.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the
Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
Boston, MA  02110-1301, USA.
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ccomponent.h"
#include "morphological.h"
#include "preprocessing.h"
#include <iostream>
#include <algorithm>
#include <vector>

CComponent::CComponent()
{
    cc_ready=false;
}
/*-------------------------------------------------------------------------------------*/
CComponent::CComponent(cv::Mat _image, bool sort_by_x)
{
    if((_image.type()==CV_8UC1)&&(_image.channels()==1))
    {
        Init(_image, sort_by_x);
    }
}
/*-------------------------------------------------------------------------------------*/
bool compare_rects(cv::Rect a, cv::Rect b){ return a.x<b.x;}


void CComponent::Init(cv::Mat _image, bool sort_by_x)
{    
    int i=0,j=0, max_x=0, max_y=0, min_x=0, min_y=0;
    int x=0, y=0;
    std::vector<std::vector<cv::Point2i> > vector_points;
    std::vector<cv::Point2i> points;
    std::vector<cv::Vec4i> hierarchy;

    //Padding with two extra cols and two extra rows to get the findcountours works    
    image=cv::Mat::zeros(_image.rows+2, _image.cols+2, _image.type());
    _image.copyTo(image(cv::Rect(1,1,_image.cols, _image.rows)));
    cc_points.clear();
    cc_bounding_box.clear();

    if(sum(_image)[0]>0)
    {
        //std::cout<<"1"<<std::endl;
        cv::Mat bin_cc=cv::Mat::zeros(image.size(), CV_8UC1);
        //std::cout<<"2"<<image<<std::endl;
        findContours(image, vector_points, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
        //rstd::cout<<"3"<<std::endl;
        //std::cout<<" ncc: "<<vector_points.size()<<std::endl;
        i=0;
        /*------------------------------------------------------------------------------*/
        int n_cc=0;
        while(i>=0)
        {
            cc_points.push_back(std::vector<cv::Point2i>());
            n_cc++;
            //std::cout<<"Nro CC :"<<cc_points.size()<<std::endl;
            /*--Drawing a connected component------------------------------------------*/
            bin_cc.setTo(cv::Scalar(0));
            drawContours(bin_cc, vector_points,i,cv::Scalar(1),-1,8,hierarchy);
            /*--------Looking for max and min of vector points-------------------------*/
            points=vector_points[i];
            min_x=points[0].x;
            min_y=points[0].y;
            max_x=points[0].x;
            max_y=points[0].y;

            for(j=0;j<(int)points.size();j++)
            {
                if(min_x>points[j].x) min_x=points[j].x;
                if(min_y>points[j].y) min_y=points[j].y;
                if(max_x<points[j].x) max_x=points[j].x;
                if(max_y<points[j].y) max_y=points[j].y;
            }
            /*------------------------------------------------------------------------------*/
            /*----------Capturing points----------------------------------------------------*/            
            cc_bounding_box.push_back(cv::Rect(min_x-1, min_y-1, max_x-min_x+1, max_y-min_y+1));
            for (x=min_x;x<=max_x;x++)
            {
                for (y=min_y;y<=max_y;y++)
                {
                    if(bin_cc.at<uchar>(y,x)==1)
                    {
                        cc_points[n_cc-1].push_back(cv::Point2i(x-1,y-1));//because of the padding
                    }
                }
            }
            /*------------------------------------------------------------------------------*/
    //            std::cout<<"Area :"<<cc_points[n_cc-1].size()<<std::endl;
    //            imshow("cc",bin_cc*255);
    //            waitKey();
            i=hierarchy[i][0];//next contour at top level
        }
        //Be careful, if sort_by_x is true only the rects will be sort according to the x position
        //The order with respect to the components are lost
        if(sort_by_x) sort(cc_bounding_box.begin(), cc_bounding_box.end(), compare_rects);
        cc_ready=true;
        //std::cout<<"Connected Components computed succsesfully"<<std::endl;
    }    
    image=image(cv::Rect(1,1,_image.cols, _image.rows));
}
/*-------------------------------------------------------------------------------------*/
void CComponent::setImage(cv::Mat _image, bool sort_by_x)//Image is a binary image,
{    
    if((_image.type()==CV_8UC1)&&(_image.channels()==1))
    {        
        Init(_image, sort_by_x);
    }
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::getLabelMatrix()
{
    cv::Mat labels=cv::Mat(image.size(), CV_8UC1);
    for(int i=0; i<getNumberOfComponents();i++)
    {
        setValues(labels, cc_points[i], i+1);
    }
    return labels;
}
/*-----------------------------------------------------------------------------------*/
void CComponent::getMaximo(std::vector<int> V, int &maxi, int &pos)
{
    int i=0, n=V.size();
    maxi=V[0];
    pos=0;
    if(n>1)
    {
        maxi=V[0];
        pos=0;
        for(i=1; i<n;i++)
        {
            if(V[i]>maxi)
            {
                maxi=V[i];
                pos=i;
            }
        }
    }
}
/*-----------------------------------------------------------------------------------*/
std::vector<int> CComponent::getAreas(std::vector<std::vector<cv::Point2i> > V)
{
    int n=V.size();
    int i=0;
    std::vector<int> areas(n);
    for(i=0;i<n;i++)
    {
        areas[i]=V[i].size();
        //std::cout<<i<<" "<<areas[i]<<std::endl;
    }
    return areas;
}
/*-----------------------------------------------------------------------------------*/
std::vector<int> CComponent::getAreas()
{
    int n=cc_points.size();
    int i=0;
    std::vector<int> areas(n);
    for(i=0;i<n;i++)
    {
        areas[i]=cc_points[i].size();
        //std::cout<<i<<" "<<areas[i]<<std::endl;
    }
    return areas;
}
/*-----------------------------------------------------------------------------------*/
std::vector<float> CComponent::getSkew(int sign)
{
    int n=cc_points.size();
    cv::RotatedRect rect;
    JEllipse jEllipse;
    int i=0;
    std::vector<float> skews(n);
    for(i=0;i<n;i++)
    {
        rect=minAreaRect(cc_points[i]);
        jEllipse=Preprocessing::rotatedRectToEllipse(rect);
        //mathematical coordinates
        skews[i]=Preprocessing::image2math(jEllipse.getAngle(JANG_DEGREE),JANG_DEGREE);
        if(sign==SKEW_NON_SIGNED)
        {
            skews[i]=(skews[i]>180)?skews[i]-180:skews[i];
        }
    }
    return skews;
}
/*-----------------------------------------------------------------------------------*/
/*Set the pixels of list_of_points with value */
void CComponent::setValues(cv::Mat image, std::vector<cv::Point2i> list_of_points, const uchar value)
{
    int i=0;
    for(i=0;i<(int)list_of_points.size();i++)
    {
        image.at<uchar>(list_of_points[i].y, list_of_points[i].x)=value;
    }
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::bwAreaOpen(double area)
{
    cv::Mat bin_cc(image.size(),CV_8UC1);
    bin_cc.setTo(cv::Scalar(0));
    int i=0;
    std::vector<int> areas;
    if(cc_ready)
    {
        areas=getAreas(cc_points);
        for(i=0; i<(int)areas.size(); i++)
        {
            if(areas[i]>area)
            {
                setValues(bin_cc,cc_points[i],1);
            }
        }

    }
    return bin_cc;
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::bwRectAreaOpen(double area)
{
    cv::Mat bin_cc(image.size(),CV_8UC1);
    bin_cc.setTo(cv::Scalar(0));
    int i=0;
    if(cc_ready)
    {
        for(i=0; i<(int)getNumberOfComponents(); i++)
        {
            if(getBoundingBox(i).area()>area)
            {
                setValues(bin_cc,cc_points[i],1);
            }
        }

    }
    return bin_cc;
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::bwDiscardSmallRegions(double th_area)
{
    return bwAreaOpen(th_area);
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::bwDiscardLargeRegions(double th_area)
{
    cv::Mat bin_cc(image.size(),CV_8UC1);
    bin_cc.setTo(cv::Scalar(0));
    int i=0;
    if(cc_ready)
    {
        for(i=0; i<(int)cc_points.size(); i++)
        {
            if(cc_points[i].size()<th_area)
            {
                setValues(bin_cc,cc_points[i],1);
            }
        }

    }
    return bin_cc;
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::bwHeightRectOpen(double u_height)
{
    cv::Mat bin_cc(image.size(),CV_8UC1);
    bin_cc.setTo(cv::Scalar(0));
    int i=0;
    if(cc_ready)
    {
        for(i=0; i<(int)getNumberOfComponents(); i++)
        {
            if(getBoundingBox(i).height>u_height)
            {
                setValues(bin_cc,cc_points[i],1);
            }
        }

    }
    return bin_cc;
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::bwWidthRectOpen(double u_width)
{
    cv::Mat bin_cc(image.size(),CV_8UC1);
    bin_cc.setTo(cv::Scalar(0));
    int i=0;
    if(cc_ready)
    {
        for(i=0; i<(int)getNumberOfComponents(); i++)
        {
            if(getBoundingBox(i).width>u_width)
            {
                setValues(bin_cc,cc_points[i],1);
            }
        }

    }
    return bin_cc;
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::bwBigger()
{
    cv::Mat bin_cc(image.size(),CV_8UC1);
    bin_cc.setTo(cv::Scalar(0));
    if (cc_bounding_box.size()>0)
    {
        int maximo=0, posM=0;
        std::vector<int> areas;
        areas=getAreas(cc_points);
        getMaximo(areas,maximo, posM);
        setValues(bin_cc,cc_points[posM],1);
    }
    return bin_cc;
}
/*-----------------------------------------------------------------------------------*/
cv::Rect CComponent::getBoundingBox(int n_cc)
{
    return cc_bounding_box[n_cc];
}
/*-----------------------------------------------------------------------------------*/
std::vector<cv::Rect> CComponent::getBoundingBoxes(){
	std::vector<cv::Rect> bbs;
	for(int i=0; i<getNumberOfComponents(); i++){
		bbs.push_back(getBoundingBox(i));
	}
	return bbs;
}
/*-----------------------------------------------------------------------------------*/
cv::Point2i CComponent::getCentroid(int n_cc)
{
    cv::Point2i centroid;
    std::vector<cv::Point2i> lop = cc_points[n_cc];
    int sum_x=0;
    int sum_y=0;
    int i=0;

    for(i=0;i<(int)lop.size();i++)
    {
        sum_x=sum_x+lop[i].x;
        sum_y=sum_y+lop[i].y;
    }
    sum_x=(int)round(sum_x/(float)lop.size());
    sum_y=(int)round(sum_y/(float)lop.size());
    centroid.x=sum_x;
    centroid.y=sum_y;
    return centroid;
}
/*-----------------------------------------------------------------------------------*/
int CComponent::getSize(int n_cc)
{
    return (cc_points[n_cc]).size();
}
/*-----------------------------------------------------------------------------------*/
std::vector<cv::Point2i> CComponent::getPoints(int n_cc)
{
    return cc_points[n_cc];
}
/*-----------------------------------------------------------------------------------*/
int CComponent::getNumberOfComponents()
{
    return cc_bounding_box.size();
}
/*-----------------------------------------------------------------------------------*/
cv::Mat CComponent::fillHoles(cv::Mat _image)
{
    cv::Mat bin_cc_rec=cv::Mat(_image.size(), CV_8UC1);

    if((_image.type()==CV_8UC1)&&(_image.channels()==1))
    {
        //---------------We used a 1-padded image to avoid border problems
        cv::Mat image=cv::Mat::zeros(_image.rows+2, _image.cols+2, CV_8UC1);
        _image.copyTo(image(cv::Rect(1,1,_image.cols, _image.rows)));
        cv::Mat bin_cc=cv::Mat::zeros(image.size(),CV_8UC1);
        bin_cc.setTo(cv::Scalar(0));
        int i=0;
        std::vector<std::vector<cv::Point2i> > vector_points;        
        std::vector<cv::Point2i> cc_points;
        std::vector<cv::Vec4i> hierarchy;
        findContours(image, vector_points, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
        i=0;

        while(i>=0)
        {

            cc_points.clear();
            /*--Drawing a connected component------------------------------------------*/
            drawContours(bin_cc, vector_points,i,cv::Scalar(1),-1,8,hierarchy);
            /*------------------------------------------------------------------------------*/
            i=hierarchy[i][0];//next contour at top level
        }
        //---------------Taking the original size
        bin_cc(cv::Rect(1,1,_image.cols, _image.rows)).copyTo(bin_cc_rec);
    }
    return bin_cc_rec;
}
/*-----------------------------------------------------------------------------------*/
std::vector<cv::Point2i> CComponent::bwBoundary(cv::Mat image)
{
    //This computes the boundary points for an one connected component image
    //discarding holes
    std::vector<std::vector<cv::Point2i> > vector_points;
    std::vector<cv::Vec4i> hierarchy;

    findContours(image, vector_points, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    //There should be one connected component
    if(vector_points.size() == 0)
        return std::vector<cv::Point2i>(1, cv::Point2i(0,0));
    return vector_points[0];
}
/*-----------------------------------------------------------------------------------*/
void CComponent::getMinMaxAllBoundingBoxes(double *min_x, double *max_x, double *min_y, double *max_y)
{
    int n=cc_bounding_box.size();
    cv::Mat mat_x=cv::Mat(n*2, 1, CV_32S);//int
    cv::Mat mat_y=cv::Mat(n*2, 1, CV_32S);//int
    /*--------------------------------------------------------------*/
    int k=0;
    for(int i=0;i<n;i++)
    {

            mat_x.at<int>(k,0)=cc_bounding_box[i].x;
            mat_x.at<int>(k+1,0)=cc_bounding_box[i].x+cc_bounding_box[i].width-1;


            mat_y.at<int>(k,0)=cc_bounding_box[i].y;
            mat_y.at<int>(k+1,0)=cc_bounding_box[i].y+cc_bounding_box[i].height-1;

            k=k+2;
    }
    /*--------------------------------------------------------------*/
    minMaxLoc(mat_x(cv::Range(0,k),cv::Range::all()), min_x, max_x);
    minMaxLoc(mat_y(cv::Range(0,k),cv::Range::all()),min_y, max_y);
}
/*-----------------------------------------------------------------------------------*/
cv::Rect CComponent::getMinMaxAllBoundingBoxes()
{
    double min_x=0, max_x=0, min_y=0, max_y=0;
    int n=cc_bounding_box.size();
    cv::Mat mat_x=cv::Mat(n*2, 1, CV_32S);//int
    cv::Mat mat_y=cv::Mat(n*2, 1, CV_32S);//int
    /*--------------------------------------------------------------*/
    int k=0;
    for(int i=0;i<n;i++)
    {

            mat_x.at<int>(k,0)=cc_bounding_box[i].x;
            mat_x.at<int>(k+1,0)=cc_bounding_box[i].x+cc_bounding_box[i].width-1;


            mat_y.at<int>(k,0)=cc_bounding_box[i].y;
            mat_y.at<int>(k+1,0)=cc_bounding_box[i].y+cc_bounding_box[i].height-1;

            k=k+2;
    }
    /*--------------------------------------------------------------*/
    minMaxLoc(mat_x(cv::Range(0,k),cv::Range::all()), &min_x, &max_x);
    minMaxLoc(mat_y(cv::Range(0,k),cv::Range::all()),&min_y, &max_y);

    return cv::Rect((int)min_x, (int)min_y, (int)max_x-(int)min_x+1, (int)max_y-(int)min_y+1);
}
/*-----------------------------------------------------------------------------------*/
//DEvuelve el porcentaje de intersección de A y B con respecto a A
float CComponent::intersectX(cv::Rect A, cv::Rect B)
{
    int x1A=0, x2A=0;
    int x1B=0, x2B=0;

    x1A=A.x;
    x2A=A.x+A.width-1;
    x1B=B.x;
    x2B=B.x+B.width-1;

    int start=std::max(x1A, x1B);
    int end=std::min(x2A, x2B);

    return (std::max(end-start+1,0)/(float)A.width);
}
/*-----------------------------------------------------------------------------------*/
//join A with B, modifying A
void joinRects(cv::Rect &A, cv::Rect B)
{
    cv::Rect C;
    C.x=std::min(A.x, B.x);
    C.y=std::min(A.y, B.y);

    C.width=std::max(A.x+A.width-1, B.x+B.width-1)-C.x+1;
    C.height=std::max(A.y+A.height-1, B.y+B.height-1)-C.y+1;

    A.x=C.x;
    A.y=C.y;
    A.width=C.width;
    A.height=C.height;
}

/*-----------------------------------------------------------------------------------*/
//Joint bounding boxes give an intersection umbral
std::vector<cv::Rect> CComponent::joinYBoundingBoxes(std::vector<cv::Rect> loBB)
{
    std::vector<cv::Rect> loBB2;
    loBB2.clear();

    std::vector<cv::Rect>::iterator curRect;
    std::vector<cv::Rect>::iterator curSig;
    float val_int=0;
    float umbral=0.2;

    for(curRect=loBB.begin(); curRect!=loBB.end();curRect++)
    {
        loBB2.push_back(*curRect);
    }

    curRect=loBB2.begin();
    curSig=curRect+1;
    /*----------------------------------------------------------------------------------*/
    while(curSig!=loBB2.end())
    {
        val_int=std::max(intersectX(*curRect, *curSig),intersectX(*curSig, *curRect));
        if(val_int>umbral)
        {
            joinRects(*curRect,*curSig);
            loBB2.erase(curSig);

        }else
        {
            curRect=curRect+1;
        }
        curSig=curRect+1;
    }
    /*----------------------------------------------------------------------------------*/
    return loBB2;
}
/*-----------------------------------------------------------------------------------------*/
void CComponent::sortComponentValue(float *values, int *index_cc, int n, int tipo)//tipo 1:ascend, 0:descend
{
    //Ordena index_cc, que corresponde a indices de los ccomps, con respecto a valores almacenados en values
    //Algoritmo cuadrático
    int i=0, j=0;
    float aux_f=0;
    int  aux_i=0;
    if(tipo==CC_SORT_ASCEND)
    {
        for(i=0;i<n-1;i++)
        {
            for(j=i+1;j<n;j++)
            {
                if(values[i]>values[j]) //ASCEND
                {
                    aux_f=values[i];
                    values[i]=values[j];
                    values[j]=aux_f;
                    /*------------------------*/
                    aux_i=index_cc[i];
                    index_cc[i]=index_cc[j];
                    index_cc[j]=aux_i;
                    /*------------------------*/
                }
            }
        }
    }
    if(tipo==CC_SORT_DESCEND)
    {
        for(i=0;i<n-1;i++)
        {
            for(j=i+1;j<n;j++)
            {
                if(values[i]<values[j]) //ASCEND
                {
                    aux_f=values[i];
                    values[i]=values[j];
                    values[j]=aux_f;
                    /*------------------------*/
                    aux_i=index_cc[i];
                    index_cc[i]=index_cc[j];
                    index_cc[j]=aux_i;
                    /*------------------------*/
                }
            }
        }
    }
}
/*------------------------------------------------------------------------------------------------------*/
cv::Mat CComponent::getBoundaryImage(cv::Mat image)
{
    //image is a CV_8UC1
    //Esta funcion dibuja el contorno de un objeto, no anula hoyos
    cv::Mat bin_cc(image.size(),CV_8UC1);
    bin_cc.setTo(cv::Scalar(0));

    if((image.type()==CV_8UC1)&&(image.channels()==1))
    {
        int  i=0;
        std::vector<std::vector<cv::Point2i> > vector_points;
        std::vector<cv::Point2i> cc_points;
        std::vector<cv::Vec4i> hierarchy;

        findContours(image.clone(), vector_points, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
        //Usando CV_RETR_CCOMP-> se obtienene tambien los bordes externos, lo que permite representar hoyos
        //Usando CV_RETR_EXTERNAL -> se obtendra solamente el borde externo, lo que eliminare cualquier hoyo
        i=0;
        while(i>=0)
        {
            cc_points.clear();
            /*--Drawing a connected component------------------------------------------*/
            drawContours(bin_cc, vector_points,i,cv::Scalar(1),1,8,hierarchy);
            /*------------------------------------------------------------------------------*/
            i=hierarchy[i][0];//next contour at top level
        }
    }
    return bin_cc;
}

std::vector<cv::Point2i> CComponent::bwGetPoints(const cv::Mat& im_bin){
	std::vector<cv::Point2i> vec_points;
	for(int i=0; i<im_bin.rows; i++){
		for(int j=0; j<im_bin.cols; j++){
			if(im_bin.at<uchar>(i,j)==1){
				vec_points.push_back(cv::Point2i(j,i));
			}
		}
	}
	return vec_points;
}
/*------------------------------------------------------------------------------------------------------*/
void CComponent::drawBoundingBoxes(cv::Mat &image, cv::Scalar color)
{
	if (image.channels()==1)	
	{
		cvtColor(image, image, CV_GRAY2BGR);
	}	
    for(int i=0; i<(int)cc_bounding_box.size();i++)
    {
		rectangle(image, cc_bounding_box[i], color, 2);
    }
}
/*------------------------------------------------------------------------------------------------------*/
void CComponent::drawBoundingBoxes(cv::Mat &image, std::vector<cv::Rect> list_bb, cv::Scalar color)
{
    if (image.channels()==1)
    {
        cvtColor(image, image, CV_GRAY2BGR);
    }
    for(int i=0; i<(int)list_bb.size();i++)
    {
            rectangle(image, list_bb[i], color, 2);
    }
}
/*------------------------------------------------------------------------------------------------------*/
void CComponent::jointCComponents(cv::Mat &bin_image, int id_cc1, int id_cc2, float TH)//this function only affects the an input image
{
    if(bin_image.rows!=image.rows || bin_image.cols!=image.cols )
    {
        std::cout<<"ERROR in jointCComponentes: incompatible image sizes!!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    //---------------------------extract cc1
    cv::Rect rect_cc1=cc_bounding_box[id_cc1];
    cv::Rect rect_cc2=cc_bounding_box[id_cc2];
    cv::Mat image_cc1=cv::Mat::zeros(rect_cc1.height+2, rect_cc1.width+2, CV_8UC1);
    cv::Mat image_cc2=cv::Mat::zeros(rect_cc2.height+2, rect_cc2.width+2, CV_8UC1);
    cv::Mat thin_cc1;
    cv::Mat thin_cc2;
    std::vector<cv::Point2i> bkp_cc1;
    std::vector<cv::Point2i> bkp_cc2;
    Preprocessing::setImageValue(image_cc1, cc_points[id_cc1], 1, -rect_cc1.x+1, -rect_cc1.y+1);
    Preprocessing::setImageValue(image_cc2, cc_points[id_cc2], 1, -rect_cc2.x+1, -rect_cc2.y+1);
    thin_cc1=Morphological::thinning_Zhang_Sue(image_cc1);
    thin_cc2=Morphological::thinning_Zhang_Sue(image_cc2);
    //std::cout<<"list"<<std::endl;
    //imshow("thin1", thin_cc1*255);
    //imshow("thin2", thin_cc2*255);
    //waitKey();
    bkp_cc1=Preprocessing::detectBreakpoints2(thin_cc1, BKP_TERMINATION);
    bkp_cc2=Preprocessing::detectBreakpoints2(thin_cc2, BKP_TERMINATION);

    float d=0, dmin=999999999;
    cv::Point2i p1;
    cv::Point2i p2;
    cv::Point2i best_p1;
    cv::Point2i best_p2;
    //------------------------search the two closest bkp
    for (int i_cc1=0;i_cc1<(int)bkp_cc1.size(); i_cc1++)
    {
        p1=bkp_cc1[i_cc1];
        p1.x+=rect_cc1.x-1;
        p1.y+=rect_cc1.y-1;
        for (int i_cc2=0;i_cc2<(int)bkp_cc2.size(); i_cc2++)
        {
            p2=bkp_cc2[i_cc2];
            p2.x+=rect_cc2.x-1;
            p2.y+=rect_cc2.y-1;
            d=Preprocessing::computeDistance(p1, p2 );
            if(d<dmin)
            {
                dmin=d;
                best_p1=p1;
                best_p2=p2;
            }
        }
    }
    std::cout<<Preprocessing::computeDistance(best_p1, best_p2)<<" "<<TH<<std::endl;
    if(Preprocessing::computeDistance(best_p1, best_p2)<TH)
    {
        std::vector<cv::Point2i> ps=Preprocessing::getLinePoints_Bresenham(best_p1, best_p2);
        Preprocessing::setImageValue(bin_image, ps,1);
    }
    //---------------------------extract cc2
}
/*------------------------------------------------------------------------------------------------------*/
void CComponent::jointCComponentsByDT(cv::Mat &bin_image, int id_cc1, int id_cc2, float TH)//this function only affects the an input image
{
    if(bin_image.rows!=image.rows || bin_image.cols!=image.cols )
    {
        std::cout<<"ERROR in jointCComponentes: incompatible image sizes!!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    //---------------------------extract cc1
    cv::Rect rect_cc1=cc_bounding_box[id_cc1];
    cv::Rect rect_cc2=cc_bounding_box[id_cc2];
    cv::Rect rect_both=rect_cc1|rect_cc2;

    cv::Mat image_cc=cv::Mat::zeros(rect_both.height+2, rect_both.width+2, CV_8UC1);
    //------------------------------------- Computing distance transform wiht cc1
    Preprocessing::setImageValue(image_cc, cc_points[id_cc1], 1, -rect_both.x+1, -rect_both.y+1);
    cv::Mat dt;
    distanceTransform(1-image_cc, dt, CV_DIST_L2, 3); //dt is CV_32F float
    double minVal=image_cc.rows*image_cc.cols;
    cv::Point minPoint;
    cv::Point p;
    for(int i=0; i<(int)cc_points[id_cc2].size();i++) //computing the mindist using cc2
    {
        p=cc_points[id_cc2][i];
        p.x=p.x+(-rect_both.x+1);
        p.y=p.y+(-rect_both.y+1);

        if(dt.at<float>(p.y, p.x)<minVal)
        {
            minVal=dt.at<float>(p.y, p.x);
            minPoint=p;
        }
    }
    if(minVal<TH) //getting the joining line
    {
        p=minPoint;
        cv::Point subm_min_point;
        double subm_min=0, subm_max=0;
        std::vector<cv::Point2i> ps;
        ps.push_back(p);
        while(p.y>=0 && p.y<image_cc.rows && p.x>=0 && p.x<image_cc.cols && image_cc.at<uchar>(p.y, p.x)!=1)
        {
            cv::Mat subm=dt(cv::Range(p.y-1,p.y+2), cv::Range(p.x-1, p.x+2));
            subm.at<float>(1,1)=INT_MAX;
            minMaxLoc(subm, &subm_min, &subm_max, &subm_min_point);
            subm_min_point.x-=1;
            subm_min_point.y-=1;
            p.x+=subm_min_point.x;
            p.y+=subm_min_point.y;
            ps.push_back(p);
        }
        //waitKey();
        Preprocessing::setImageValue(bin_image, ps,1, rect_both.x-1, rect_both.y-1);
    }
}
/*----------------------------------------------------------------------------------------------------------*/
float CComponent::getMeanHeight(float *dsv)
{
    float sum=0;
    int n=cc_bounding_box.size();
    for (int i=0;i<n;i++)
    {
        sum+=cc_bounding_box[i].height;
    }
    float mean=sum/n;
    if(dsv!=NULL)
    {
        sum=0;
        for (int i=0;i<n;i++)
        {
            sum+=(mean-cc_bounding_box[i].height)*(mean-cc_bounding_box[i].height);
        }
        sum=sum/n;
        *dsv=sqrt(sum);
    }
    return mean;
}

/*----------------------------------------------------------------------------*/
void CComponent::traceImage(cv::Mat &image, cv::Point2i point, int connection_type)
{
    //image is a binary image just 1 and 0
    if(point.x>=0 && point.x<image.cols && point.y>=0 && point.y<image.rows)
    {
        if(image.at<uchar>(point.y, point.x)==1)
        {
            image.at<uchar>(point.y, point.x)=2;
            //for 8 and 4 connected
            traceImage(image, point+cv::Point2i(0,-1), connection_type);
            traceImage(image, point+cv::Point2i(0,+1), connection_type);
            traceImage(image, point+cv::Point2i(+1,0), connection_type);
            traceImage(image, point+cv::Point2i(-1,0), connection_type);
            if (connection_type==CONNECTED_8)
            {
                traceImage(image, point+cv::Point2i(-1,-1), connection_type);
                traceImage(image, point+cv::Point2i(+1,-1), connection_type);
                traceImage(image, point+cv::Point2i(-1,+1), connection_type);
                traceImage(image, point+cv::Point2i(+1,+1), connection_type);
            }
        }
    }
}
/*----------------------------------------------------------------------------*/
//This function discard components that are not connected to point
void CComponent::traceFromThisPoint(cv::Mat &image, cv::Point2i point, int connection_type)
{
    //Be care on data, this stores all the data of the origina Mat
    if(image.type()==CV_8UC1)
    {
        traceImage(image, point, connection_type);

        for(int i=0; i<image.rows*image.cols; i++)
        {
            if(image.data[i]==2) image.data[i]=1;
            else image.data[i]=0;
        }
    }
}

/*----------------------------------------------------------------------------*/
//This function estimates the roundness of a binary region
/**
 *
 * @param mat_bin: A binary image fg=1, bg=0
 * @return the value of circularity(roundness) of an object in a binary image [0-1]
 */
float CComponent::bwGetCircularity(cv::Mat& mat_bin){
    float cir=0;
    std::vector<cv::Point2i> contour=bwBoundary(mat_bin);
	int n=contour.size();
    float cx=0, cy=0, r=0;
    bool status_ok=bwCircleEstimation(mat_bin, &cx, &cy, &r);
	if (status_ok){
		float error=0;

		float d=0;
		for(int i=0; i<n;i++){
			d=std::sqrt(std::pow(cx-contour[i].x,2) + std::pow(cy-contour[i].y,2));
			error+=std::pow(r-d,2);
		}
		error=std::sqrt(error);
		float area_circ=PI*r*r;
		//std::cout<<"r: "<<r<<" PI*r*r: "<<area_circ<<std::endl;
		cir=1.0/std::exp(error*100/area_circ);
	}
	return cir;
}

/*----------------------------------------------------------------------------*/
//This function estimates a circle through the general equation
bool CComponent::bwCircleEstimation(cv::Mat& mat_bin, float* x_c, float* y_c, float* radius){
	bool status=false;
	std::vector<cv::Point2i> contour=bwBoundary(mat_bin);
	int pos=0;
	int n=contour.size();
	if(n>10){
		cv::Mat A(5,3,CV_32F);
		cv::Mat B(5,1,CV_32F);
		cv::Mat X;
		pos=0;
		A.at<float>(0,0)=contour[pos].x;
		A.at<float>(0,1)=contour[pos].y;
		A.at<float>(0,2)=1;
		B.at<float>(0)=std::pow(contour[pos].x,2)+std::pow(contour[pos].y,2);

		pos=n*0.2;
		A.at<float>(1,0)=contour[pos].x;
		A.at<float>(1,1)=contour[pos].y;
		A.at<float>(1,2)=1;
		B.at<float>(1)=std::pow(contour[pos].x,2)+std::pow(contour[pos].y,2);

		pos=n*0.4;
		A.at<float>(2,0)=contour[pos].x;
		A.at<float>(2,1)=contour[pos].y;
		A.at<float>(2,2)=1;
		B.at<float>(2)=std::pow(contour[pos].x,2)+std::pow(contour[pos].y,2);

		pos=n*0.6;
		A.at<float>(3,0)=contour[pos].x;
		A.at<float>(3,1)=contour[pos].y;
		A.at<float>(3,2)=1;
		B.at<float>(3)=std::pow(contour[pos].x,2)+std::pow(contour[pos].y,2);

		pos=n*0.8;
		A.at<float>(4,0)=contour[pos].x;
		A.at<float>(4,1)=contour[pos].y;
		A.at<float>(4,2)=1;
		B.at<float>(4)=std::pow(contour[pos].x,2)+std::pow(contour[pos].y,2);

		//std::cout<<A<<std::endl;
		//std::cout<<B<<std::endl;
		cv::solve(A,B,X,cv::DECOMP_NORMAL);

		//std::cout<<X<<std::endl;
		float a=X.at<float>(0);
		float b=X.at<float>(1);
		float c=X.at<float>(2);

		float cx=a*0.5;
		float cy=b*0.5;
		float r=std::sqrt(c+cx*cx+cy*cy);
		*x_c=cx;
		*y_c=cy;
		*radius=r;
		status=true;
	}
	return status;
}
//------------------------------------------------------------------------------------
float* CComponent::bwGetTopProfile(const cv::Mat& mat_bin, unsigned int* size){
	*size=mat_bin.cols;
	float* v=new float[*size];
	int i=0;
	for(int unsigned j=0; j<*size; j++){
		i=0;
		while(i<mat_bin.rows && mat_bin.at<uchar>(i,j)==0){
			i++;
		}
		v[j]=i;
	}
	return v;
}
