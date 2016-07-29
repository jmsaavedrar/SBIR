/*
preprocessing.cpp,
Implementation of static functions devoted to image processing.
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

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include<iostream>
#include<cstdlib>
#include<cassert>
#include<cmath>
#include "preprocessing.h"
#include "JUtil.h"
#include "morphological.h"
#include "ccomponent.h"

namespace jmsr{
	NormMethod string2norm_method(std::string str_norm){
		NormMethod norm=jmsr::NONE;
		if(str_norm == "SUM") norm=jmsr::SUM;
		else if(str_norm == "MAX") norm=jmsr::MAX;
		else if(str_norm == "ROOT_UNIT") norm=jmsr::ROOT_UNIT;
		else if(str_norm == "ROOT") norm=jmsr::ROOT;
		else if(str_norm == "UNIT") norm=jmsr::UNIT;
		else norm=jmsr::NONE;
		return norm;
	}
}
/*----------------------------------------------------------------------------------*/
//! Implementation for JLine class
JLine::JLine()
{
    ang=0; /*!< in math coordinates */
    angRad=0;
    length=0;
}
/*----------------------------------------------------------------------------------*/
JLine::JLine(cv::Point2i p0, cv::Point2i p1)
{
    setPoints(p0,p1);
}
/*----------------------------------------------------------------------------------*/
JLine::JLine(cv::Point2i p0, cv::Point2i p1, int idx_el)
{
    setPoints(p0,p1);
    idx_edge_link=idx_el;
}
/*----------------------------------------------------------------------------------*/
void JLine::setPoints(cv::Point2i p0, cv::Point2i p1)
{
    point0=p0;
    point1=p1;
    angRad=atan2(point1.y-point0.y, point1.x-point0.x); //[-PI..PI]//
    angRad=Preprocessing::image2math(angRad,JANG_RADIANS); //[0..2PI]
    if(angRad>PI) angRad=angRad-CV_PI; //ahora [0..PI]
    ang=180.0*(angRad/CV_PI);
    length=sqrt((point1.y-point0.y)*(point1.y-point0.y)+
                (point1.x-point0.x)*(point1.x-point0.x));
    idx_edge_link=-1;
}
/*----------------------------------------------------------------------------------*/
void JLine::setEndPoint(cv::Point2i p)
{
    point1=p;
    setPoints(point0,point1);
}
/*----------------------------------------------------------------------------------*/
void JLine::setInitialPoint(cv::Point2i p)
{
    point0=p;
    setPoints(point0,point1);
}
/*----------------------------------------------------------------------------------*/
float JLine::getAngle(int tipo)
{
    if(tipo==JLINE_DEGREE) return ang;
    else if(tipo==JLINE_RADIANS) return angRad;
    else return -1;
}
/*----------------------------------------------------------------------------------*/
float JLine::getLength()
{
    return length;
}
/*----------------------------------------------------------------------------------*/
cv::Point2i JLine::getEndPoint()
{
    return point1;
}
/*----------------------------------------------------------------------------------*/
cv::Point2i JLine::getInitialPoint()
{
    return point0;
}
/*-------------------------------------------------------------------------------*/
void JLine::setEdgeLinkIdx(int idx)
{
    idx_edge_link=idx;
}
/*-------------------------------------------------------------------------------*/
int JLine::getEdgeLinkIdx()
{
    return idx_edge_link;
}

/*-------------------------------------------------------------------------------*/
JEllipse::JEllipse()
{
    ang_deg=0; //in image coordinates
    max_rad=0;
    min_rad=0;
    center=cv::Point2i(0,0);
}
/*-------------------------------------------------------------------------------*/
JEllipse::JEllipse(float _ang_deg, float _max_rad, float _min_rad, cv::Point2i _center)
{
    ang_deg=_ang_deg;//angle in degrees [0..180]
    max_rad=_max_rad;
    min_rad=_min_rad;
    center=_center;
}
/*-------------------------------------------------------------------------------*/
float JEllipse::getAngle(int tipo)
{
    if(tipo==JANG_DEGREE) return ang_deg;
    return ang_deg*(CV_PI/180.0);
}
/*-------------------------------------------------------------------------------*/
float JEllipse::getMaxRad(){return max_rad;}
/*-------------------------------------------------------------------------------*/
float JEllipse::getMinRad(){return min_rad;}
/*-------------------------------------------------------------------------------*/
cv::Point2i JEllipse::getCenter(){return center;}
/*-------------------------------------------------------------------------------*/
float JEllipse::getArea(){return CV_PI*max_rad*min_rad;}
/*-------------------------------------------------------------------------------*/
float JEllipse::getPerimeter()
{
    return CV_PI*(3*(max_rad+min_rad)-sqrt((3*max_rad+min_rad)*(max_rad+3*min_rad)));
}
/*-------------------------------------------------------------------------------*/
float JEllipse::getRadiiRatio(){return max_rad/min_rad;}
/*-------------------------------------------------------------------------------*/
Preprocessing::Preprocessing()
{
}
/*-------------------------------------------------------------------------------*/
cv::Mat Preprocessing::preprocessDigit(cv::Mat image)
{
    //image is a grayscale image
    CComponent cconn;
    double min_val=0, max_val=0;
    cv::Mat elem=(cv::Mat_<double>(3,3) << 1,1,1, 1,1,1, 1,1,1);
    /*---------------------------------------------------------------------------*/
    minMaxLoc(image,&min_val, &max_val);
    //adaptiveThreshold(image, image, 1, ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25,7);
    threshold(image,image, 0.8*max_val,1,cv::THRESH_BINARY);
    image=1-image;
    /*---------------------------------------------------------------------------*/
    dilate(image,image, elem);
    dilate(image,image, elem);
    /*---------------------------------------------------------------------------*/
    image=CComponent::fillHoles(image);
    /*---------------------------------------------------------------------------*/
    cconn.setImage(image);
    image=cconn.bwBigger();
    return image;
}
/*------------------------------------------------------------------------------*/
cv::Mat Preprocessing::preprocessDigit_1(cv::Mat image)
{
    //image is a grayscale image
    double min_val=0, max_val=0;
    cv::Mat elem=(cv::Mat_<double>(3,3) << 1,1,1, 1,1,1, 1,1,1);
    /*---------------------------------------------------------------------------*/
    minMaxLoc(image,&min_val, &max_val);
    //adaptiveThreshold(image, image, 1, ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25,7);
    //threshold(image,image, 0.8*max_val,1,cv::THRESH_BINARY); //estaba este
    int umbral=Preprocessing::umbralOtsu(image);
    threshold(image,image, umbral,1,cv::THRESH_BINARY);
    /*---------------------------------------------------------------------------*/
    //image=Morphological::thinning_Zhang_Sue(image);
    //cv::Mat stel=getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
    //morphologyEx(image, image, cv::MORPH_DILATE, stel,cv::Point(-1,-1),2);
    /*---------------------------------------------------------------------------*/
    image=1-image;
    /*---------------------------------------------------------------------------*/
    //dilate(image,image, elem);//hay que poner para el entrenamiento
    //image=CComponent::fillHoles(image);

    return image;
}
/*-------------------------------------------------------------------*/
cv::Mat Preprocessing::histograma(cv::Mat input)
{
    //asumimos en escala de grises
    cv::Mat hist=cv::Mat::zeros(256,1,CV_32S);
    int i=0, j=0;

    for(i=0;i<input.size().height;i++)
    {
        for(j=0;j<input.size().width;j++)
        {
            hist.at<int>(input.at<uchar>(i,j),0)++;
        }
    }
    return hist;
}
/*-------------------------------------------------------------------------------*/
int Preprocessing::umbralOtsu(cv::Mat imagen, float *var)//
{    
    cv::Mat hist; 
    hist=histograma(imagen);
    hist.convertTo(hist,CV_32F);
    hist=hist*(1.0/sum(hist)[0]);
    float media[256];
    float acum[256];
    float m2=0, m1=0;
    float P1=0, P2=0;

    media[0]=0;
    acum[0]=hist.at<float>(0,0);
    int i=0;
    for(i=1;i<256;i++)
    {
        media[i]=i*hist.at<float>(i,0)+media[i-1];
        acum[i]=hist.at<float>(i,0)+acum[i-1];
    }
    int t=0;
    float maxVal=0, val=0;
    float m=0;
    m=media[255];
    for(i=0;i<256;i++)
    {

        P1=acum[i];
        P2=1-P1;

        if(P1!=0 && P2!=0)
        {
            m1=media[i]/P1;
            m2=(m-media[i])/P2;
            val=P1*(m1-m)*(m1-m)+P2*(m2-m)*(m2-m);

            if(val>maxVal)
            {
                maxVal=val;
                t=i;
            }
        }
    }    
    if(var!=NULL) *var=maxVal;
    return t;
}
/*-------------------------------------------------------------------------------*/
int Preprocessing::umbralByPercentil(cv::Mat imagen, float percentil)//
{
    cv::Mat hist;
    hist=histograma(imagen);
    hist.convertTo(hist,CV_32F);
    hist=hist*(1.0/sum(hist)[0]); //normalized histogram
    float acum[256];
    acum[0]=hist.at<float>(0,0);
    int i=0;
    int t=0;
    for(i=1;i<256;i++)
    {
        acum[i]=hist.at<float>(i,0)+acum[i-1];
    }
    t=0;
    while(acum[t]<(percentil/100.0))t++;
    if(t!=0) t--;
    return t;
}
/*--------------------------------------------------------------------------------------------*/
void Preprocessing::smoothVector(float *vector, int n)
{
    if(n>=3)
    {
        float *new_vector=new float[n];
        int i=0;        
        for(i=1;i<n-1;i++)
        {
            new_vector[i]=vector[i-1]*0.2+vector[i]*0.6+vector[i+1]*0.2;
        }
        for(i=1;i<n-1;i++)
        {
            vector[i]=new_vector[i];
        }
        delete[] new_vector;
    }
}
/*--------------------------------------------------------------------------------------------*/
std::vector<cv::Point2i>  Preprocessing::getLinePoints_Bresenham(cv::Point2i p, cv::Point2i q)
{
    //using algorithm of Bresenham, see the algorithm in wiki,
    //returns the set of points between two given points
     int x0=0, x1=0;
     int y0=0, y1=0;
     int dx=0, dy=0;
     int inc_x=0, inc_y=0;
     int error=0, error2=0;
     std::vector<cv::Point2i> set_of_points;
     set_of_points.clear();

     x0=p.x;
     y0=p.y;
     x1=q.x;
     y1=q.y;
     dx=abs(x1-x0);
     dy=abs(y1-y0);
     error=dx-dy;
     if(x0<=x1) inc_x=1;
     else inc_x=-1;
     if(y0<=y1) inc_y=1;
     else inc_y=-1;

     while((x0!=x1) || (y0!=y1))
     {
         set_of_points.push_back(cv::Point2i(x0,y0));         
         error2=2*error;
         if(error2>-dy)
         {
             error=error-dy;
             x0=x0+inc_x;
         }
         if(error2<dx)
         {
             error=error+dx;
             y0=y0+inc_y;
         }

     }
    return set_of_points;
}
/*--------------------------------------------------------------------------------------------*/
void Preprocessing::setImageValue(cv::Mat &im, std::vector<cv::Point2i> sop, uchar value, int offset_x, int offset_y)//this does not work with multichanner, but it is easy to be adapted
{
    //im is CV_8UC1
    //value is uchar
    for(int i=0;i<(int)sop.size();i++)
    {
        if((sop[i].y+offset_y)>=0 && (sop[i].y+offset_y)<im.rows && (sop[i].x+offset_x)>=0 && (sop[i].x+offset_x)<im.cols)
        {
            im.at<uchar>(sop[i].y+offset_y, sop[i].x+offset_x)=value;
        }        
    }
}
/*--------------------------------------------------------------------------------------------*/
void Preprocessing::setImageValue(cv::Mat &im, std::vector<JLine> list_lines, uchar value, int offset_x, int offset_y, std::vector<int> mask)
{
    //im is CV_8UC1
    //value is uchar
    cv::Point p0;
    cv::Point p1;
    std::vector<cv::Point2i> list_points;
    if(mask.size()==0)
    {
        for(int i=0;i<(int)list_lines.size();i++)
        {

            p0=list_lines[i].getInitialPoint();
            p1=list_lines[i].getEndPoint();
            list_points=getLinePoints_Bresenham(p0,p1);

            setImageValue(im,list_points,value,offset_x,offset_y);
        }
    }
    else
    {
        for(int i=0;i<(int)mask.size();i++)
        {

            p0=list_lines[mask[i]].getInitialPoint();
            p1=list_lines[mask[i]].getEndPoint();
            list_points=getLinePoints_Bresenham(p0,p1);

            setImageValue(im,list_points,value,offset_x,offset_y);
        }
    }
}
/*--------------------------------------------------------------------------------------------*/
void Preprocessing::setImageColorValue(cv::Mat &im, std::vector<cv::Point2i> sop, cv::Scalar value, int offset_x, int offset_y)
{
    //im is CV_8UC1
    //value is uchar
    for(int i=0;i<(int)sop.size();i++)
    {
        if((sop[i].y+offset_y)>=0 && (sop[i].y+offset_y)<im.rows && (sop[i].x+offset_x)>=0 && (sop[i].x+offset_x)<im.cols)
        {
            im.at<cv::Vec3b>(sop[i].y+offset_y, sop[i].x+offset_x)[0]=value[0];
            im.at<cv::Vec3b>(sop[i].y+offset_y, sop[i].x+offset_x)[1]=value[1];
            im.at<cv::Vec3b>(sop[i].y+offset_y, sop[i].x+offset_x)[2]=value[2];
        }
    }
}
/*--------------------------------------------------------------------------------------------*/
void Preprocessing::drawContour(cv::Mat &im, std::vector<cv::Point2i> sop, uchar value)//draw a line between points
{
    //im is CV_8UC1
    //value is uchar
    int i=0;
    for(i=0;i<(int)sop.size()-1;i++)
    {
        //if(sop[i].y>=0 && sop[i].y<im.rows && sop[i].x>=0 && sop[i].x<im.cols)
        //{
        cv::line(im,sop[i], sop[i+1], value);
        //}
    }
    cv::line(im,sop[i], sop[0], value);
}
/*--------------------------------------------------------------------------------------------*/
std::vector<uchar> Preprocessing::getImageValue(cv::Mat im, std::vector<cv::Point2i> sop)
{   //im is a cV_8UC1
    //retrieve values from an one-channel uchar image
    std::vector<uchar> set_of_values;
    set_of_values.clear();

    for(int i=0;i<(int)sop.size();i++)
    {
        if(sop[i].y>=0 && sop[i].y<im.rows && sop[i].x>=0 && sop[i].x<im.cols)
        {
            set_of_values.push_back(im.at<uchar>(sop[i].y, sop[i].x));
        }
    }
    return set_of_values;
}
/*-------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------*/
/*anistropicDiffusion:
/ This implements the anisotropic diffusion  proposed by Perona & Malik 1988
 -------------------------------------------------------------------*/
//lambda:
//K:
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::computeC1(int x, double K)
{
    float v=0;
    v=exp(-(pow(x/K,2.0)));
    return v;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::computeC2(int x, double K)
{
    double v=0;
    v=1.0/(1+ pow(x/K,2.0));
    return v;
}
/*-------------------------------------------------------------------*/
cv::Mat Preprocessing::computeThermalDiffusion(cv::Mat input, double K)
{
    cv::Mat c=cv::Mat(input.size(), CV_32F);
    int i=0, j=0;

    for(i=0;i<input.rows;i++)
    {
        for(j=0; j<input.cols;j++)
        {
            c.at<float>(i,j)=computeC1(abs(input.at<float>(i,j)), K);
        }
    }

    return c;
}
/*-------------------------------------------------------------------*/
//lambda < 0.25, according to paper
//K: to be fixed manually, it regularizes the gradient value, 10, 20
//max_iter: n'umero de iteraciones ej. 20
/*-------------------------------------------------------------------*/
cv::Mat Preprocessing::anisotropicDiffusion(const cv::Mat& input, double lambda, double K,int max_it)
{

    cv::Mat ani;
    cv::Mat ani_N;
    cv::Mat ani_S;
    cv::Mat ani_E;
    cv::Mat ani_W;

    //Máscaras para obtener gradiente en una dirección*/
    cv::Mat fil_N=(cv::Mat_<float>(3,1)<<1,-1,0);
    cv::Mat fil_S=(cv::Mat_<float>(3,1)<<0,-1,1);
    cv::Mat fil_E=(cv::Mat_<float>(1,3)<<0,-1,1);
    cv::Mat fil_W=(cv::Mat_<float>(1,3)<<1,-1,0);

    cv::Mat I_N;
    cv::Mat I_S;
    cv::Mat I_E;
    cv::Mat I_W;

    cv::Mat c_N;
    cv::Mat c_S;
    cv::Mat c_E;
    cv::Mat c_W;

    input.convertTo(ani, CV_32F);//convertir a float y guardarlo en ani
    int it=0;
    for(it=0;it<max_it;it++)
    {

        /*-----Obtener valores por correlación--------------*/
        cv::filter2D(ani,I_N,CV_32F, fil_N);
        cv::filter2D(ani,I_S,CV_32F, fil_S);
        cv::filter2D(ani,I_E,CV_32F, fil_E);
        cv::filter2D(ani,I_W,CV_32F, fil_W);
        /*--------------------------------------------------*/
        //calculamos los coeficientes de difusión
        c_N=computeThermalDiffusion(I_N, K);
        c_S=computeThermalDiffusion(I_S, K);
        c_E=computeThermalDiffusion(I_E, K);
        c_W=computeThermalDiffusion(I_W, K);
        /*--------------------------------------------------*/
        cv::multiply(c_N,I_N, ani_N);
        cv::multiply(c_S,I_S, ani_S);
        cv::multiply(c_E,I_E, ani_E);
        cv::multiply(c_W,I_W, ani_W);

        ani=lambda*(ani_N+ani_S+ani_E+ani_W)+ani;

    }
    ani.convertTo(ani,CV_8UC1);//uint8
    return ani;
}
/*--------------------------------------------------------------*/
cv::Mat Preprocessing::edge_derivadas(cv::Mat input, int TIPO)
{
    cv::Mat mask_x;
    cv::Mat mask_y;
    cv::Point p;
    cv::Mat Gx;
    cv::Mat Gy;
    cv::Mat G;


    if(TIPO==EDGE_DERIVADA)
    {
        mask_x=(cv::Mat_<int>(1,2)<<-1,1);
        mask_y=(cv::Mat_<int>(2,1)<<-1,1);
        p=cv::Point(0,0);
    }
    else if(TIPO==EDGE_PREWITT)
    {
        mask_x=(cv::Mat_<int>(3,3)<<-1, 0, 1, -1, 0, 1, -1, 0, 1);
        mask_y=(cv::Mat_<int>(3,3)<<-1, -1, -1, 0, 0, 0, 1, 1, 1);
        p=cv::Point(-1,-1);
    }
    else if(TIPO==EDGE_SOBEL)
    {
        mask_x=(cv::Mat_<int>(3,3)<<-1, 0, 1, -2, 0, 2, -1, 0, 1);
        mask_y=(cv::Mat_<int>(3,3)<<-1, -2, -1, 0, 0, 0, 1, 2, 1);
        p=cv::Point(-1,-1);
    }

    filter2D(input,Gx,CV_32F,mask_x,p);
    filter2D(input,Gy,CV_32F,mask_y,p);
    convertScaleAbs(Gx,Gx);//valor absoluto y convierte a CV_8UC1
    convertScaleAbs(Gy,Gy);

    G=Gx+Gy;
    convertScaleAbs(G,G);
    return G;
}
/*-------------------------------------------------------------------------------------------*/
//detect breakpoints, tipo is set for terminations or bifurcation
cv::Mat Preprocessing::
detectBreakpoints(cv::Mat input, int tipo)
{
    //input is a binary image CV_8UC1
    cv::Mat borde=cv::Mat(1,9,CV_32F);
    cv::Mat mask=(cv::Mat_<float>(1,2)<<1,-1);
    cv::Mat breaks=cv::Mat::zeros(input.size(), input.type());
    cv::Mat result;
    int mark_termination=0;
    int mark_bifurcation=0;
    if((tipo==BKP_TERMINATION)||(tipo==BKP_ALL))  mark_termination=1;
    if((tipo==BKP_BIFURCATION)||(tipo==BKP_ALL)) mark_bifurcation=1;

    int n=0;
    for(int i=0;i<input.rows; i++)
    {
        for(int j=0; j<input.cols;j++)
        {
            if(input.at<uchar>(i,j)==1)
            {
                if(isValidPoint(i-1,j-1,input.size())) borde.at<float>(0,0)=input.at<uchar>(i-1,j-1);
                else borde.at<float>(0,0)=0;

                if(isValidPoint(i-1,j,input.size())) borde.at<float>(0,1)=input.at<uchar>(i-1,j);
                else borde.at<float>(0,1)=0;

                if(isValidPoint(i-1,j+1,input.size())) borde.at<float>(0,2)=input.at<uchar>(i-1,j+1);
                else borde.at<float>(0,2)=0;

                if(isValidPoint(i,j+1,input.size())) borde.at<float>(0,3)=input.at<uchar>(i,j+1);
                else borde.at<float>(0,3)=0;

                if(isValidPoint(i+1,j+1,input.size())) borde.at<float>(0,4)=input.at<uchar>(i+1,j+1);
                else borde.at<float>(0,4)=0;

                if(isValidPoint(i+1,j,input.size())) borde.at<float>(0,5)=input.at<uchar>(i+1,j);
                else borde.at<float>(0,5)=0;

                if(isValidPoint(i+1,j-1,input.size())) borde.at<float>(0,6)=input.at<uchar>(i+1,j-1);
                else borde.at<float>(0,6)=0;

                if(isValidPoint(i,j-1,input.size()))  borde.at<float>(0,7)=input.at<uchar>(i,j-1);
                else borde.at<float>(0,7)=0;

                if(isValidPoint(i-1,j-1,input.size())) borde.at<float>(0,8)=input.at<uchar>(i-1,j-1);
                else borde.at<float>(0,8)=0;

                borde=borde*2-1;// obtenemos 1->1,0->-1
                filter2D(borde,result,CV_32F,mask, cv::Point(0,0));
                n=0;
                for(int k=0; k<8;k++)
                {
                    if(result.at<float>(0,k)==2) n++;
                }
                //cout<<borde<<" "<<result<<" "<<n<<endl;
                if(n==1) breaks.at<uchar>(i,j)=mark_termination;
                if(n>=3) breaks.at<uchar>(i,j)=mark_bifurcation;
            }
        }
    }
    return breaks;
}
/*-------------------------------------------------------------------------------------------*/
//detect breakpoints, tipo is set for terminations or bifurcation, this returns a std::vector of cv::Point2i
std::vector<cv::Point2i> Preprocessing::detectBreakpoints2(cv::Mat input, int tipo)
{
    //input is a binary image CV_8UC1
    cv::Mat borde=cv::Mat(1,9,CV_32F);
    cv::Mat mask=(cv::Mat_<float>(1,2)<<1,-1);
    cv::Mat breaks=cv::Mat::zeros(input.size(), input.type());
	std::vector<cv::Point2i> break_points;
	break_points.clear();
    cv::Mat result;
    int mark_termination=0;
    int mark_bifurcation=0;
    if((tipo==BKP_TERMINATION)||(tipo==BKP_ALL))  mark_termination=1;
    if((tipo==BKP_BIFURCATION)||(tipo==BKP_ALL)) mark_bifurcation=1;

    int n=0;
    for(int i=0;i<input.rows; i++)
    {
        for(int j=0; j<input.cols;j++)
        {
            if(input.at<uchar>(i,j)==1)
            {
                if(isValidPoint(i-1,j-1,input.size())) borde.at<float>(0,0)=input.at<uchar>(i-1,j-1);
                else borde.at<float>(0,0)=0;

                if(isValidPoint(i-1,j,input.size())) borde.at<float>(0,1)=input.at<uchar>(i-1,j);
                else borde.at<float>(0,1)=0;

                if(isValidPoint(i-1,j+1,input.size())) borde.at<float>(0,2)=input.at<uchar>(i-1,j+1);
                else borde.at<float>(0,2)=0;

                if(isValidPoint(i,j+1,input.size())) borde.at<float>(0,3)=input.at<uchar>(i,j+1);
                else borde.at<float>(0,3)=0;

                if(isValidPoint(i+1,j+1,input.size())) borde.at<float>(0,4)=input.at<uchar>(i+1,j+1);
                else borde.at<float>(0,4)=0;

                if(isValidPoint(i+1,j,input.size())) borde.at<float>(0,5)=input.at<uchar>(i+1,j);
                else borde.at<float>(0,5)=0;

                if(isValidPoint(i+1,j-1,input.size())) borde.at<float>(0,6)=input.at<uchar>(i+1,j-1);
                else borde.at<float>(0,6)=0;

                if(isValidPoint(i,j-1,input.size()))  borde.at<float>(0,7)=input.at<uchar>(i,j-1);
                else borde.at<float>(0,7)=0;

                if(isValidPoint(i-1,j-1,input.size())) borde.at<float>(0,8)=input.at<uchar>(i-1,j-1);
                else borde.at<float>(0,8)=0;

                borde=borde*2-1;// obtenemos 1->1,0->-1
                filter2D(borde,result,CV_32F,mask, cv::Point(0,0));
                n=0;
                for(int k=0; k<8;k++)
                {
                    if(result.at<float>(0,k)==2) n++;
                }
                //cout<<borde<<" "<<result<<" "<<n<<endl;
                if(n==1) breaks.at<uchar>(i,j)=mark_termination;
                if(n>=3) breaks.at<uchar>(i,j)=mark_bifurcation;
				if(breaks.at<uchar>(i,j)==1) break_points.push_back(cv::Point2i(j,i));
            }
        }
    }
    return break_points;;
}
/*----------------------------------------------------------------------------------*/
std::vector<std::vector<cv::Point2i> > Preprocessing::edgeLink(cv::Mat input, int TH_MIN_SIZE)
{
    //input is a binary image
    //trace store all visited
    //TH_MIN_SIZE
    //delete 1s from borders---------------------*/    
    //This segment is not longer required, breakpoints are already computed on the border!!
    /*input.row(0)=cv::Scalar(0);
    input.col(0)=cv::Scalar(0);
    input.col(input.cols-1)=cv::Scalar(0);
    input.row(input.rows-1)=cv::Scalar(0);*/
    /*------------------------------------------*/
    cv::Mat bks;
    cv::Mat traceM=cv::Mat::zeros(input.size(), input.type());
    cv::Mat input_aux=input.clone();
    std::vector<std::vector<cv::Point2i> > edgelinks;
    std::vector<cv::Point2i> link;
    edgelinks.clear();
    bool fin=false;
    int it=0;
    //------------------- We use breakpoints as start and end points
    while((it<2)&&(!fin))//maximo dos iteraciones
    {
        bks=detectBreakpoints(input_aux, BKP_ALL);        
        //traceM=cv::Mat::zeros(input.size(), input.type());
        traceM=cv::Scalar(0);
       // cout<<traceM<<endl;
        for(int i=0;i<input.rows; i++)
        {
            for(int j=0;j<input.cols; j++)
            {
                if((bks.at<uchar>(i,j)==1) && (traceM.at<uchar>(i,j)==0))//no visited
                {
                    link=getEdgeLink(input_aux, traceM, bks, i, j);
                    if((int)link.size()>=TH_MIN_SIZE)
                    {
                        edgelinks.push_back(link);
                    }
                }
            }
        }
        traceM=1-traceM;
        cv::multiply(input_aux, traceM, input_aux);
        if(sum(input_aux)[0]==0) fin=true;
        it++;
    }    
    //------------------ In the case of having close curves, no breakpoint will be found    
    if (sum(input_aux)[0]>0)
    {
        bks=detectBreakpoints(input_aux, BKP_ALL);
        //traceM=cv::Mat::zeros(input.size(), input.type());
        traceM=cv::Scalar(0);
        CComponent ccomp;
        ccomp.setImage(input_aux);
        int nccomp=ccomp.getNumberOfComponents();
        std::vector<cv::Point2i> list_cc;
        for(int i=0;i<nccomp;i++)
        {
            list_cc=ccomp.getPoints(i);
            link=getEdgeLink(input_aux, traceM, bks, list_cc[0].y, list_cc[0].x);
            if((int)link.size()>=TH_MIN_SIZE)
            {
                edgelinks.push_back(link);
            }
        }
    }

    return edgelinks;
}
/*-------------------------------------------------------------------------------------------*/
bool Preprocessing::isIsolated(cv::Mat input, int i, int j)
{
    bool ans=false;
    int u=0, v=0, s=0;
    for(u=i-1;u<=i+1;u++)
    {
        for(v=j-1;v<=j+1;v++)
        {
            if((u!=i || v!=j)&& isValidPoint(u,v,input.size()))
            {
                s+=input.at<uchar>(u,v);
            }
        }
    }
    if(s==0) ans=true;
    return ans;
}
/*-------------------------------------------------------------------------------------------*/
bool Preprocessing::isValidPoint(int i, int j, cv::Size im_size)
{
    if((i>=0) && (j>=0) && (i<im_size.height) && (j<im_size.width))
        return true;
    else
        return false;
}
/*-------------------------------------------------------------------------------------------*/
bool Preprocessing::isNearBreakPoint(cv::Mat bkps, cv::Mat visited,  int *ii, int *jj)
{
    //It is true if the point i,j is a bkp or there is a breakpoint neighbor around it.
    //ii, jj store the found bkp point
    bool ans=false;
    int i=*ii;
    int j=*jj;
    //if(isValidPoint(i, j, bkps.size()) &&  bkps.at<uchar>(i,j)==1) {i=i; j=j; ans=true;}
    if(isValidPoint(i-1, j-1, bkps.size()) &&  visited.at<uchar>(i-1, j-1)==0 && bkps.at<uchar>(i-1,j-1)==1) {i=i-1; j=j-1; ans=true;}
    else if(isValidPoint(i-1, j, bkps.size()) &&  visited.at<uchar>(i-1, j)==0 && bkps.at<uchar>(i-1,j)==1) {i=i-1; j=j; ans=true;}
    else if(isValidPoint(i-1, j+1, bkps.size()) &&  visited.at<uchar>(i-1, j+1)==0 && bkps.at<uchar>(i-1,j+1)==1) {i=i-1; j=j+1; ans=true;}
    else if(isValidPoint(i, j+1, bkps.size())   &&  visited.at<uchar>(i, j+1)==0 && bkps.at<uchar>(i,j+1)==1) {i=i; j=j+1; ans=true;}
    else if(isValidPoint(i+1, j+1, bkps.size()) &&  visited.at<uchar>(i+1, j+1)==0 && bkps.at<uchar>(i+1,j+1)==1) {i=i+1; j=j+1; ans=true;}
    else if(isValidPoint(i+1, j, bkps.size())   &&  visited.at<uchar>(i+1, j)==0 && bkps.at<uchar>(i+1,j)==1) {i=i+1; j=j; ans=true;}
    else if(isValidPoint(i+1, j-1, bkps.size()) &&  visited.at<uchar>(i+1, j-1)==0 && bkps.at<uchar>(i+1,j-1)==1) {i=i+1; j=j-1; ans=true;}
    else if(isValidPoint(i, j-1, bkps.size())   &&  visited.at<uchar>(i, j-1)==0 && bkps.at<uchar>(i,j-1)==1) {i=i; j=j-1; ans=true;}
    *ii=i;
    *jj=j;
    return ans;
}

/*-------------------------------------------------------------------------------------------*/
//delete the edgelink from the input matrix
std::vector<cv::Point2i> Preprocessing::getEdgeLink(cv::Mat input, cv::Mat &traceM, cv::Mat breakpoints, int pos_i, int pos_j)
{
    //the initial point has to be a breakpoint
    //trace is marked each time the point is visited
    std::vector<cv::Point2i> link;
    cv::Mat visited;
    int i=pos_i;
    int j=pos_j;
    int ii=0;
    int jj=0;    
    bool changed=false;
    if ((traceM.at<uchar>(i,j)==0) && isIsolated(input, i, j) )
    {
        traceM.at<uchar>(i,j)=1;
        link.push_back(cv::Point2i(j,i));//x,y
        return link;
    }
    else
    {
        visited=cv::Mat::zeros(input.size(), input.type());        
        do
        {
            changed=false;
            link.push_back(cv::Point2i(j,i));
            traceM.at<uchar>(i,j)=1;
            visited.at<uchar>(i,j)=1;
            if(isValidPoint(i-1, j-1, input.size()) && (visited.at<uchar>(i-1,j-1)==0) && input.at<uchar>(i-1,j-1)==1) {i=i-1; j=j-1; changed=true;}
            else if(isValidPoint(i-1, j, input.size()) && (visited.at<uchar>(i-1,j)==0) && input.at<uchar>(i-1,j)==1) {i=i-1; j=j; changed=true;}
            else if(isValidPoint(i-1, j+1, input.size()) && (visited.at<uchar>(i-1,j+1)==0) && input.at<uchar>(i-1,j+1)==1) {i=i-1; j=j+1;changed=true;}
            else if(isValidPoint(i, j+1, input.size()) && (visited.at<uchar>(i,j+1)==0) && input.at<uchar>(i,j+1)==1) {i=i; j=j+1;changed=true;}
            else if(isValidPoint(i+1, j+1, input.size()) && (visited.at<uchar>(i+1,j+1)==0) && input.at<uchar>(i+1,j+1)==1) {i=i+1; j=j+1;changed=true;}
            else if(isValidPoint(i+1, j, input.size())  && (visited.at<uchar>(i+1,j)==0) && input.at<uchar>(i+1,j)==1) {i=i+1; j=j;changed=true;}
            else if(isValidPoint(i+1, j-1, input.size()) && (visited.at<uchar>(i+1,j-1)==0) &&input.at<uchar>(i+1,j-1)==1) {i=i+1; j=j-1;changed=true;}
            else if(isValidPoint(i, j-1, input.size()) && (visited.at<uchar>(i,j-1)==0) && input.at<uchar>(i,j-1)==1) {i=i; j=j-1;changed=true;}
            ii=i;
            jj=j;            

        }while(changed && !isNearBreakPoint(breakpoints, visited, &i, &j));//until a breakpoint is achieved
        if(ii!=i || jj!=j)
        {
            traceM.at<uchar>(ii,jj)=1;
            link.push_back(cv::Point2i(jj,ii));
        }
        traceM.at<uchar>(i,j)=1;
        link.push_back(cv::Point2i(j,i));
        return link;
    }

}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::drawEdgeLinks(cv::Mat &image, std::vector<std::vector<cv::Point2i> > edgeLinks, cv::Scalar color)
{
    //
    bool random_color=false;
    if(color[0]==-1 || color[1]==-1  || color[2]==-1  ) random_color=true;
    for(int k=0; k<(int)edgeLinks.size();k++)
    {
        if(random_color)
        {
            color[0]=(int)(rand() % 256);
            color[1]=(int)(rand() % 256);
            color[2]=(int)(rand() % 256);
        }
        Preprocessing::setImageColorValue(image,edgeLinks[k],color);
    }
}
/*-------------------------------------------------------------------------------------------*/
//Compute the perpendicular distance from a point to a line defined by point0 and point1
float Preprocessing::getPerpendicularDistance(cv::Point2f point, cv::Point2f point0, cv::Point2f point1)
{
    float d=0;
    if(abs(point0.x-point1.x)==0)//vertical line
    {
        d=abs(point.x-point0.x);
        return d;
    }
    float m=(point0.y-point1.y)/((point0.x-point1.x));
    float b=point0.y-m*point0.x;
    d=abs(m*point.x-point.y+b)/sqrt(m*m+1);
    return d;
}
/*-------------------------------------------------------------------------------------------*/
//Compute the maximun deviation in a set of points, with respect to a line defined by
//the terminal points, returns pos and dist
void Preprocessing::findMaxDeviation(std::vector<cv::Point2i> set_of_points, int *pos, float *dist)
{
    cv::Point2f point0;
    cv::Point2f point1;
    point0.x=set_of_points[0].x;
    point0.y=set_of_points[0].y;
    point1.x=set_of_points[set_of_points.size()-1].x;
    point1.y=set_of_points[set_of_points.size()-1].y;
    float max_d=-1, d=0;
    for(int i=0; i<(int)set_of_points.size();i++)
    {
        d=getPerpendicularDistance(set_of_points[i], point0, point1);
        if(d>max_d)
        {
            *dist=d;
            max_d=d;
            *pos=i;
        }
    }
}
/*-------------------------------------------------------------------------------------------*/
//Compute the maximun deviation in a set of points[inip, endp], with respect to a line defined by
//the terminal points, returns pos and dist
void Preprocessing::findMaxDeviation(std::vector<cv::Point2i> set_of_points, int *pos, float *dist, int inip, int endp)
{
    cv::Point2f point0;
    cv::Point2f point1;
    point0.x=set_of_points[inip].x;
    point0.y=set_of_points[inip].y;

    point1.x=set_of_points[endp].x;
    point1.y=set_of_points[endp].y;
    float max_d=-1, d=0;
    for(int i=inip; i<endp;i++)
    {
        d=getPerpendicularDistance(set_of_points[i], point0, point1);
        if(d>max_d)
        {
            *dist=d;
            max_d=d;
            *pos=i;
        }
    }
}
/*-------------------------------------------------------------------------------------------*/
//find inflection points with respect to a line approximation of the contour defined by
//set_of_points
std::vector<int> Preprocessing::findInflectionPoints(std::vector<cv::Point2i> set_of_points, float TH)
{
  std::vector<int> inflections;
  inflections=findInflectionPoints(set_of_points, TH, 0, set_of_points.size()-1);
  return inflections;
}
/*-------------------------------------------------------------------------------------------*/
//as findInplectionPoints but between a segment defined in [inip, endp]
//this is a recursive method
std::vector<int> Preprocessing::findInflectionPoints(std::vector<cv::Point2i> set_of_points, float TH, int inip, int endp)
{
    int pos=0;
    float dist=0;
    std::vector<int> inflections;
    inflections.clear();
    if(endp-inip+1<=2)
    {
        return inflections;
    }
    findMaxDeviation(set_of_points, &pos, &dist, inip, endp);    
    if (dist>TH)
    {
        std::vector<int> A=findInflectionPoints(set_of_points, TH, inip, pos-1);
        std::vector<int> B=findInflectionPoints(set_of_points, TH, pos+1, endp);
        //merging results
        for(int i=0;i<(int)A.size();i++)
        {
            inflections.push_back(A[i]);
        }
        inflections.push_back(pos);
        for(int i=0;i<(int)B.size();i++)
        {
            inflections.push_back(B[i]);
        }
    }
    return inflections;
}
/*-------------------------------------------------------------------------------------------*/
//Transform a set of edgelinks to a set of lines, using a TH for maximum deviation
std::vector<JLine> Preprocessing::edgeLinks2Lines(std::vector<std::vector<cv::Point2i> > edgelinks, float TH)
{
    std::vector<JLine> lines;
    lines.clear();
    std::vector<cv::Point2i> set_of_points;
    std::vector<int> inflections;
    int k=0, j=0;
    cv::Point2i p0;
    cv::Point2i p1;
    for(k=0;k<(int)edgelinks.size();k++)
    {
        set_of_points=edgelinks[k];
        inflections=findInflectionPoints(set_of_points, TH);
        p0=set_of_points[0];
        for(j=0;j<(int)inflections.size();j++)
        {
            p1=set_of_points[inflections[j]];
            lines.push_back(JLine(p0,p1,k));
            p0=p1;
        }
        p1=set_of_points[set_of_points.size()-1];
        lines.push_back(JLine(p0,p1,k));
    }
    return lines;
}
/*-------------------------------------------------------------------------------------------*/
//Transform a set of edgelinks to a set of lines, using a TH for maximum deviation
std::vector<JLine> Preprocessing::edgeLinks2Lines(std::vector<cv::Point2i> edgelink, float TH)
{
    std::vector<JLine> lines;
    lines.clear();
    std::vector<cv::Point2i> set_of_points;
    std::vector<int> inflections;
    int  j=0;
    cv::Point2i p0;
    cv::Point2i p1;
    set_of_points=edgelink;
    inflections=findInflectionPoints(set_of_points, TH);
    p0=set_of_points[0];
    for(j=0;j<(int)inflections.size();j++)
    {
        p1=set_of_points[inflections[j]];
        lines.push_back(JLine(p0,p1,0));
        p0=p1;
    }
    p1=set_of_points[set_of_points.size()-1];
    lines.push_back(JLine(p0,p1,0));
    return lines;
}
/*-------------------------------------------------------------------------------------------*/
//Given a set of edgelinks, this implementation gives the points of maximum deviation
std::vector<cv::Point2i> Preprocessing::getInflectionPoints(std::vector<std::vector<cv::Point2i> > edgelinks, float TH)
{
    std::vector<cv::Point2i> set_of_inflection_points;
    set_of_inflection_points.clear();
    std::vector<cv::Point2i> set_of_points;
    std::vector<int> inflections;
    int k=0, j=0;
    cv::Point2i p0;
    cv::Point2i p1;
    for(k=0;k<(int)edgelinks.size();k++)
    {
        set_of_points=edgelinks[k];
        inflections=findInflectionPoints(set_of_points, TH);
        p0=set_of_points[0];
        set_of_inflection_points.push_back(p0);
        for(j=0;j<(int)inflections.size();j++)
        {
            p1=set_of_points[inflections[j]];
            set_of_inflection_points.push_back(p1);
        }
        p1=set_of_points[set_of_points.size()-1];
        set_of_inflection_points.push_back(p1);
    }
    return set_of_inflection_points;
}
/*-------------------------------------------------------------------------------------------*/
//Given a set of edgelinks, this implementation gives the points of maximum deviation (A version with only one edgelink)
std::vector<cv::Point2i> Preprocessing::getInflectionPoints(std::vector<cv::Point2i> edgelink, float TH)
{
    std::vector<cv::Point2i> set_of_inflection_points;
    set_of_inflection_points.clear();
    std::vector<cv::Point2i> set_of_points;
    std::vector<int> inflections;
    cv::Point2i p0;
    cv::Point2i p1;
    int j=0;
    set_of_points=edgelink;
    inflections=findInflectionPoints(set_of_points, TH);
    p0=set_of_points[0];
    set_of_inflection_points.push_back(p0);
    for(j=0;j<(int)inflections.size();j++)
    {
        p1=set_of_points[inflections[j]];
        set_of_inflection_points.push_back(p1);
    }
    p1=set_of_points[set_of_points.size()-1];
    set_of_inflection_points.push_back(p1);
    return set_of_inflection_points;
}
/*-------------------------------------------------------------------------------------------*/
//Draw  a seg or straigh lines stores as JLine objects
//image is a color image
//mask contains indexes of the lines that will be drawn
void Preprocessing::drawJLines(cv::Mat &image, std::vector<JLine> lines, cv::Scalar color, std::vector<int> mask)
{
    //write color=[-1,-1,-1] for random o uno de ellos -1
    bool random_color=false;
    if(color[0]==-1  || color[1]==-1 || color[2]==-1) random_color=true;
    if(mask.size()>0)
    {
        for(int k=0;k<(int)mask.size();k++)
        {
            if(random_color)
            {
                color[0]=(int)(rand()%255);
                color[1]=(int)(rand()%255);
                color[2]=(int)(rand()%255);
            }
            line(image,lines[mask[k]].getInitialPoint(), lines[mask[k]].getEndPoint(), color);
        }
    }
    else
    {
        for(int k=0;k<(int)lines.size();k++)
        {
            if(random_color)
            {
                color[0]=(int)(rand()%255);
                color[1]=(int)(rand()%255);
                color[2]=(int)(rand()%255);
            }
            line(image,lines[k].getInitialPoint(), lines[k].getEndPoint(), color);
        }
    }
}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::drawHistogram(std::string name, cv::Mat _histogram)
{
    // Draw the histograms for B, G or R
    cv::Mat histogram;
    if (_histogram.type()!=CV_32F) _histogram.convertTo(histogram, CV_32F);
    else histogram=_histogram.clone();

    int im_w=600; int im_h=400;
    int hist_w = (int)(im_w*0.8);
    int hist_h = (int)(im_h*0.8);

    int histSize=std::max(histogram.cols, histogram.rows);
    //cout<<histSize<<endl;
    float bin_w=hist_w/(float)histSize;
    cv::Mat h;
    cv::Mat histImage(im_h, im_w, CV_8UC3, cv::Scalar( 0,0,0) );
    /// Normalize the result to [ 0, histImage.rows ]
    //normalize(histogram,h, 1,hist_h, NORM_L1, CV_32F, cv::Mat());
    double _max_val=0, _min_val=0;
    cv::minMaxIdx(histogram,&_min_val, &_max_val);
    h=(histogram/_max_val)*hist_h;        
    /// Draw for each channel
    int offset_x=(int)(im_w*0.1);
    int offset_y=(int)(im_h*0.1);
    putText(histImage, "Histogram", cv::Point(offset_x, offset_y-10),1,1,cv::Scalar(255,255,255));
    rectangle(histImage, cv::Rect(offset_x-2,offset_y-2,hist_w+4, hist_h+4), cv::Scalar(255,255,255));
    for( int i = 0; i < histSize; i++ )
    {
        //para dibujar solamente el contorno
//        line(histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(h.at<float>(i-1)) ) ,
//                           cv::Point( bin_w*(i), hist_h - cvRound(h.at<float>(i)) ),
//                           cv::Scalar( 255, 0, 0), 2, 8, 0  );
        putText(histImage,Preprocessing::intToString(i),
                cv::Point(i*bin_w+offset_x, offset_y+hist_h+4),1,0.5,1);        
        line(histImage,  cv::Point(i*bin_w+offset_x,hist_h-h.at<float>(i)+offset_y),
                         cv::Point(i*bin_w+offset_x,hist_h+offset_y),
                         cv::Scalar( 0, 0, 255), 1, 8, 0);

    }
    imshow(name, histImage);
}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::plot1D(std::string name, float *_values, int n)
{
    // plot values
    //copying values
    float *values=new float[n];
    for(int i=0; i<n;i++) values[i]=_values[i];

    int im_w=600; int im_h=0;
    float max_value=-1;
    for(int i=0;i<n;i++)
    {
        if(values[i]>max_value) max_value=values[i];
    }
    for(int i=0;i<n;i++)
    {
        values[i]=(im_w/4.0)*values[i]/max_value;
    }

    im_h=round(2*(im_w*0.5));
    int datos_w=n;



    int hist_w = (int)(im_w*0.8);
    int hist_h = (int)(im_h*0.8);

    //cout<<histSize<<endl;
    float bin_w=hist_w/(float)datos_w;

    cv::Mat histImage(im_h, im_w, CV_8UC3, cv::Scalar( 0,0,0) );
    //cout<<im_h<<" "<<im_w<<endl;
    //cout<<h<<endl;
    /// Draw for each channel
    int offset_x=(int)(im_w*0.1);
    int offset_y=(int)(im_h*0.1);
    putText(histImage, "Plot", cv::Point(offset_x, offset_y-10),1,1,cv::Scalar(255,255,255));
    rectangle(histImage, cv::Rect(offset_x-2,offset_y-2,hist_w+4, hist_h+4), cv::Scalar(255,255,255));
    for( int i = 0; i < datos_w; i++ )
    {
        //para dibujar solamente el contorno
//        line(histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(h.at<float>(i-1)) ) ,
//                           cv::Point( bin_w*(i), hist_h - cvRound(h.at<float>(i)) ),
//                           cv::Scalar( 255, 0, 0), 2, 8, 0  );
        putText(histImage,Preprocessing::intToString(i),
                cv::Point(i*bin_w+offset_x, offset_y+hist_h+10),1,0.5,cv::Scalar(255,255,255));
        line(histImage,  cv::Point(i*bin_w+offset_x,hist_h-0.8*values[i]+offset_y),
                         cv::Point(i*bin_w+offset_x,hist_h+offset_y),
                         cv::Scalar( 0, 0, 255), 1, 8, 0);
    }
    delete[] values;
    imshow(name, histImage);
}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::plot1D(std::string name, int *_values, int n)
{
    // plot values
    //copying values
    float  *values=new float[n];
    for(int i=0; i<n;i++) values[i]=_values[i];

    int im_w=600; int im_h=0;
    float max_value=-1;
    for(int i=0;i<n;i++)
    {
        if(values[i]>max_value) max_value=values[i];
    }
    for(int i=0;i<n;i++)
    {
        values[i]=(im_w/4.0)*values[i]/max_value;
    }

    im_h=round(2*(im_w*0.5));
    int datos_w=n;



    int hist_w = (int)(im_w*0.8);
    int hist_h = (int)(im_h*0.8);

    //cout<<histSize<<endl;
    float bin_w=hist_w/(float)datos_w;

    cv::Mat histImage(im_h, im_w, CV_8UC3, cv::Scalar( 0,0,0) );
    //cout<<im_h<<" "<<im_w<<endl;
    //cout<<h<<endl;
    /// Draw for each channel
    int offset_x=(int)(im_w*0.1);
    int offset_y=(int)(im_h*0.1);
    putText(histImage, "Plot", cv::Point(offset_x, offset_y-10),1,1,cv::Scalar(255,255,255));
    rectangle(histImage, cv::Rect(offset_x-2,offset_y-2,hist_w+4, hist_h+4), cv::Scalar(255,255,255));
    for( int i = 0; i < datos_w; i++ )
    {
        //para dibujar solamente el contorno
//        line(histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(h.at<float>(i-1)) ) ,
//                           cv::Point( bin_w*(i), hist_h - cvRound(h.at<float>(i)) ),
//                           cv::Scalar( 255, 0, 0), 2, 8, 0  );
        putText(histImage,Preprocessing::intToString(i),
                cv::Point(i*bin_w+offset_x, offset_y+hist_h+10),1,0.5,cv::Scalar(255,255,255));
        line(histImage,  cv::Point(i*bin_w+offset_x,hist_h-0.8*values[i]+offset_y),
                         cv::Point(i*bin_w+offset_x,hist_h+offset_y),
                         cv::Scalar( 0, 0, 255), 1, 8, 0);
    }
    delete[] values;
    imshow(name, histImage);
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::getLineOrientationHistogram(std::vector<JLine> lines, int NBINS)
{
    float max_angle=PI;
    float index=0;
    int *vhist=new int[NBINS];
    int izq=0, der=0;
    float w_izq=0, w_der=0;
    float ang=0;
    for(int k=0; k<NBINS; k++)  vhist[k]=0;
    for(int k=0; k<(int)lines.size();k++)
    {
        ang=lines[k].getAngle(JLINE_RADIANS);//ang varia entre 0 y 360
        if(ang>PI) ang=ang-CV_PI;
        index=((ang/max_angle)*NBINS);
        linearInterBIN(index,&izq, &der, &w_izq, &w_der);
        if(izq<0) izq=NBINS-1;
        if(der==NBINS) der=0;
        vhist[izq]+=lines[k].getLength()*w_izq;
        vhist[der]+=lines[k].getLength()*w_der;        
    }
    cv::Mat hist=cv::Mat::zeros(1,NBINS,CV_32S);
    for(int k=0; k<NBINS; k++)
    {
        hist.at<int>(k)=vhist[k];
    }
    return hist;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::DOC_skewCorrection(cv::Mat image, float scale)
{
    //image is an image
    cv::Mat img=image.clone();
    if(image.channels()==3)
        cvtColor(img, img, CV_BGR2GRAY);
    //--------smoothing------------------------------------
    medianBlur(img,img, 7);
    resize(img,img,cv::Size(),scale, scale);
    img=Preprocessing::anisotropicDiffusion(img,0.2,10,10);
    //-----------------------------------------------------
    //Detecting edge to represent text or box borders
    cv::Mat bin;
    Canny(img, bin, 50, 200, 3);
    cv::Mat stel=getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    morphologyEx(bin,bin,cv::MORPH_DILATE,stel,cv::Point(-1,-1),3);
    bin=CComponent::fillHoles(bin);
    //-----------------------------------------------------
    //Detectando solamente bordes horizontales
    cv::Mat sobely=(cv::Mat_<float>(3,3)<<1, 2,1, 0,0,0,-1, -2, -1);
    cv::Mat imgx;
    cv::filter2D(bin*255,imgx, CV_32F, sobely);
    convertScaleAbs(imgx,imgx);
    threshold(imgx,bin, 150,1,cv::THRESH_BINARY);//nos quedamos con los bordes mas fuertes
    CComponent cconn; //Eliminamos bordes irrelevantes
    cconn.setImage(bin);
    bin=cconn.bwAreaOpen(100);
    bin=Morphological::thinning_Zhang_Sue(bin); //thinning
    /*--------------------------------------------------------*/
    //Detecting edgelinks and convert them to lines
    std::vector<std::vector<cv::Point2i> > els=Preprocessing::edgeLink(bin,10);//MIN_LINK_SIZE=10
    //cv::Mat im_els=cv::Mat::zeros(bin.size(), CV_8UC3);
    //Preprocessing::drawEdgeLinks(im_els, els, cv::Scalar(-1,-1,-1));
    std::vector<JLine> lines;
    lines=Preprocessing::edgeLinks2Lines(els, 10); //TH_MAX_DEVIATION=10

  //  cv::Mat im_lines=cv::Mat::zeros(bin.size(), CV_8UC3);
 //   Preprocessing::drawJLines(im_lines, lines, cv::Scalar(-1,-1,-1));
//    imshow("lines", im_lines);
    /*--------------------------------------------------------*/
    cv::Mat hist;
    hist=Preprocessing::getLineOrientationHistogram(lines, 180);//180 is number of bins
    //cout<<hist<<endl;
    //Preprocessing::drawHistogram("hist", hist);
    //cout<<hist<<endl;
    /*--------------------------------------------------------*/
    //finding angle with maximal occurrence
    cv::Point minPoint;
    cv::Point maxPoint;
    double minH=0, maxH=0;
    float ang;
    minMaxLoc(hist,&minH, &maxH, &minPoint, &maxPoint);
    //cout<<"max value: "<<maxH<<" max point: "<<maxPoint.x<<" "<<maxPoint.y<<endl;
    ang=maxPoint.x;
    if(ang<90) ang=-ang;
    else ang=180-ang;
    cv::Point2f im_center;
    im_center.x=img.cols*0.5;
    im_center.y=img.rows*0.5;
    cv::Mat R=getRotationMatrix2D(im_center,ang,1);
    //imshow("img_antes", img);
    img.release();
    warpAffine(image,img, R, img.size());
    //imshow("img_alienada", img);
    //cout<<lines.size()<<endl;
    return img;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::CHECK_skewCorrection(cv::Mat image,  float scale, float *angle)
{
    //image is an image
    cv::Mat img=image.clone();
    if(image.channels()==3)
        cvtColor(img, img, CV_BGR2GRAY);
    //--------smoothing------------------------------------
    //medianBlur(img,img, 7);
    resize(img,img,cv::Size(),scale, scale);
    img=Preprocessing::anisotropicDiffusion(img,0.1,5,5);
    //-----------------------------------------------------
    //Detecting edge to represent text or box borders
    cv::Mat bin;
    bin=Preprocessing::canny(img,-1,-1,1);
    cv::Mat stel=getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    morphologyEx(bin,bin,cv::MORPH_DILATE,stel,cv::Point(-1,-1),2);
    //bin=CComponent::fillHoles(bin);
    //imshow("bin", bin*255);
    //waitKey();
    //-----------------------------------------------------
    //Detectando solamente bordes horizontales
    cv::Mat sobely=(cv::Mat_<float>(3,3)<<1, 2,1, 0,0,0,-1, -2, -1);
    cv::Mat imgx;
    cv::filter2D(bin*255,imgx, CV_32F, sobely);
    convertScaleAbs(imgx,imgx);
    threshold(imgx,bin, 150,1,cv::THRESH_BINARY);//nos quedamos con los bordes mas fuertes
    CComponent cconn; //Eliminamos bordes irrelevantes
    cconn.setImage(bin);
    bin=cconn.bwAreaOpen(100);

    bin=Morphological::thinning_Zhang_Sue(bin); //thinning
    /*--------------------------------------------------------*/
    //Detecting edgelinks and convert them to lines
    std::vector<std::vector<cv::Point2i> > els=Preprocessing::edgeLink(bin,10);//MIN_LINK_SIZE=10
    //cv::Mat im_els=cv::Mat::zeros(bin.size(), CV_8UC3);
    //Preprocessing::drawEdgeLinks(im_els, els, cv::Scalar(-1,-1,-1));
    //waitKey();
    std::vector<JLine> lines;
    lines=Preprocessing::edgeLinks2Lines(els, 1); //TH_MAX_DEVIATION=10
    //discarding_lines
    std::vector<JLine>::iterator it_lines;

    it_lines=lines.begin();
    while(it_lines!=lines.end())
    {
        if(it_lines->getLength()<50) it_lines=lines.erase(it_lines);
        else it_lines++;
    }

    //cv::Mat im_lines=cv::Mat::zeros(bin.size(), CV_8UC3);
    //Preprocessing::drawJLines(im_lines, lines, cv::Scalar(-1,-1,-1));
    //imshow("lines", im_lines);
    /*--------------------------------------------------------*/
    cv::Mat hist;
    int nbins=720;
    hist=Preprocessing::getLineOrientationHistogram(lines, nbins);
    //cout<<hist<<endl;
    //Preprocessing::drawHistogram("hist", hist);
    //cout<<hist<<endl;
    //cout<<hist<<endl;
    /*--------------------------------------------------------*/
    //finding angle with maximal occurrence
    cv::Point minPoint;
    cv::Point maxPoint;
    double minH=0, maxH=0;
    float ang;
    minMaxLoc(hist,&minH, &maxH, &minPoint, &maxPoint);
    //cout<<"max value: "<<maxH<<" max point: "<<maxPoint.x<<" "<<maxPoint.y<<endl;
    //drawHistogram("hist",hist);
    ang=maxPoint.x*180.0/nbins;//the max angle in  hist is PI
    //std::cout<<"angle: "<<ang<<endl;
    if(angle!=NULL) *angle=ang;//degree in image coord
    if(ang<90) ang=-ang;
    else ang=180-ang;
    cv::Point2f im_center;
    im_center.x=image.cols*0.5;
    im_center.y=image.rows*0.5;
    cv::Mat R=getRotationMatrix2D(im_center,ang,1);
    //imshow("img_antes", img);
    img.release();
    warpAffine(image,img, R, image.size());
    //imshow("img_alienada", img);
    //cout<<lines.size()<<endl;
    return img;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::getAngleBySkewCorrection(cv::Mat image,  float scale)
{
    //image is an image
    cv::Mat img=image.clone();
    if(image.channels()==3)
        cvtColor(img, img, CV_BGR2GRAY);
    //--------smoothing------------------------------------
    //medianBlur(img,img, 7);
    resize(img,img,cv::Size(),scale, scale);
    img=Preprocessing::anisotropicDiffusion(img,0.1,5,5);
    //-----------------------------------------------------
    //Detecting edge to represent text or box borders
    cv::Mat bin;
    bin=Preprocessing::canny(img,-1,-1,1);
    cv::Mat stel=getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    morphologyEx(bin,bin,cv::MORPH_DILATE,stel,cv::Point(-1,-1),2);
    //bin=CComponent::fillHoles(bin);
    //imshow("bin", bin*255);
    //waitKey();
    //-----------------------------------------------------
    //Detectando solamente bordes horizontales
    cv::Mat sobely=(cv::Mat_<float>(3,3)<<1, 2,1, 0,0,0,-1, -2, -1);
    cv::Mat imgx;
    cv::filter2D(bin*255,imgx, CV_32F, sobely);
    convertScaleAbs(imgx,imgx);
    threshold(imgx,bin, 150,1,cv::THRESH_BINARY);//nos quedamos con los bordes mas fuertes
    CComponent cconn; //Eliminamos bordes irrelevantes
    cconn.setImage(bin);
    bin=cconn.bwAreaOpen(100);

    bin=Morphological::thinning_Zhang_Sue(bin); //thinning
    /*--------------------------------------------------------*/
    //Detecting edgelinks and convert them to lines
    std::vector<std::vector<cv::Point2i> > els=Preprocessing::edgeLink(bin,10);//MIN_LINK_SIZE=10
    //cv::Mat im_els=cv::Mat::zeros(bin.size(), CV_8UC3);
    //Preprocessing::drawEdgeLinks(im_els, els, cv::Scalar(-1,-1,-1));
    //waitKey();
    std::vector<JLine> lines;
    lines=Preprocessing::edgeLinks2Lines(els, 1); //TH_MAX_DEVIATION=10
    //discarding_lines
    std::vector<JLine>::iterator it_lines;

    it_lines=lines.begin();
    while(it_lines!=lines.end())
    {
        if(it_lines->getLength()<50) it_lines=lines.erase(it_lines);
        else it_lines++;
    }

    //cv::Mat im_lines=cv::Mat::zeros(bin.size(), CV_8UC3);
    //Preprocessing::drawJLines(im_lines, lines, cv::Scalar(-1,-1,-1));
    //imshow("lines", im_lines);
    /*--------------------------------------------------------*/
    cv::Mat hist;
    int nbins=720;
    hist=Preprocessing::getLineOrientationHistogram(lines, nbins);
    //cout<<hist<<endl;
    //Preprocessing::drawHistogram("hist", hist);
    //cout<<hist<<endl;
    //cout<<hist<<endl;
    /*--------------------------------------------------------*/
    //finding angle with maximal occurrence
    cv::Point minPoint;
    cv::Point maxPoint;
    double minH=0, maxH=0;
    float ang;
    minMaxLoc(hist,&minH, &maxH, &minPoint, &maxPoint);
    //cout<<"max value: "<<maxH<<" max point: "<<maxPoint.x<<" "<<maxPoint.y<<endl;
    //drawHistogram("hist",hist);
    ang=maxPoint.x*180.0/nbins;//the max angle in  hist is PI
    //std::cout<<"angle: "<<ang<<endl;
    return ang; //[0..180]
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::getProb_MOG(float vector[], int DIM, float U[], float COV[], float W[], int N)
{
    int i=0, k=0;
    float a=0, x=0, det=0, p=0, aux=0;
    p=0;
    for(i=0; i<N; i++)
    {
        det=1;
        aux=0;
        for(k=0;k<DIM;k++)
        {
            a=vector[k]-U[i*DIM+k];
            x=1.0/COV[i*DIM+k];
            det*=COV[i*DIM+k];
            aux+=a*a*x;
        }
        aux=exp(-0.5*aux)/sqrt(det);
        p=p+W[i]*aux;
    }
    return p/pow(2*CV_PI,1.5);
}

/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::skinSegmentationMOG(cv::Mat input, double umbral)
{
    //input is an BGR image;
    /*---Weight for skin gaussian models-------*/
    float W_S[]= {0.0294, 0.0331, 0.0654, 0.0756,
                  0.0554, 0.0314, 0.0454, 0.0469,
                  0.0956, 0.0763, 0.1100, 0.0676,
                  0.0755, 0.0500, 0.0667, 0.0749};

    /*---Weight for non-skin gaussian models-------*/
    float W_NS[]={0.0637, 0.0516, 0.0864, 0.0636,
                  0.0747, 0.0365, 0.0349, 0.0649,
                  0.0656, 0.1189, 0.0362, 0.0849,
                  0.0368, 0.0389, 0.0943, 0.0477};

    /*---Means for skin gaussian models------------*/
    float U_S[]={73.53,	 29.94,  17.76,
                 249.71, 233.94, 217.49,
                 161.68, 116.25, 96.95,
                 186.07, 136.62, 114.40,
                 189.26, 98.37,	 51.18,
                 247.00, 152.20, 90.84,
                 150.10, 72.66,	 37.76,
                 206.85, 171.09, 156.34,
                 212.78, 152.82, 120.04,
                 234.87, 175.43, 138.94,
                 151.19, 97.74,	 74.59,
                 120.52, 77.55,	 59.82,
                 192.20, 119.62, 82.32,
                 214.29, 136.08, 87.24,
                 99.57,	 54.33,	 38.06,
                 238.88, 203.08, 176.91};

    float U_NS[]={254.37,	254.41,	253.82,
                9.39,	8.09,	8.52,
                96.57,	96.95,	91.53,
                160.44, 162.49, 159.06,
                74.98,	63.23,	46.33,
                121.83, 60.88,	18.31,
                202.18, 154.88, 91.04,
                193.06, 201.93, 206.55,
                51.88,	57.14,	61.55,
                30.88,	26.84,	25.32,
                44.97,	85.96,	131.95,
                236.02, 236.27, 230.70,
                207.86, 191.20, 164.12,
                99.83,	148.11, 188.17,
                135.06, 131.92, 123.10,
                135.96, 103.89, 66.88};

    float COV_S[]={ 765.40, 121.44, 112.80,
                    39.94,  154.44, 396.05,
                    291.03, 60.48,  162.85,
                    274.95, 64.60,  198.27,
                    633.18, 222.40, 250.69,
                    65.23,  691.53, 609.92,
                    408.63, 200.77, 257.57,
                    530.08, 155.08, 572.79,
                    160.57, 84.52,  243.90,
                    163.80, 121.57, 279.22,
                    425.40, 73.56,  175.11,
                    330.45, 70.34,  151.82,
                    152.76, 92.14,  259.15,
                    204.90, 140.17, 270.19,
                    448.13, 90.18,  151.29,
                    178.38, 156.27, 404.99};

    float COV_NS[]={2.77,   2.81,   5.46,
                    46.84,  33.59,  32.48,
                    280.69, 156.79, 436.58,
                    355.98, 115.89, 591.24,
                    414.84, 245.95, 361.27,
                    2502.24,1383.53,237.18,
                    957.42, 1766.94,1582.52,
                    562.88, 190.23, 447.28,
                    344.11, 191.77, 433.40,
                    222.07, 118.65, 182.41,
                    651.32, 840.52, 963.67,
                    225.03, 117.29, 331.95,
                    494.04, 237.69, 533.52,
                    955.88, 654.95, 916.70,
                    350.35, 130.30, 388.43,
                    806.44, 642.20, 350.36};

    cv::Mat result=cv::Mat(input.size(), CV_8UC1);
    float rgb[3];
    float p_s=0, p_ns=0;
    for(int i=0;i<result.rows;i++)
    {
        for(int j=0;j<result.cols;j++)
        {
            rgb[0]=input.at<cv::Vec3b>(i,j)[2];//Red
            rgb[1]=input.at<cv::Vec3b>(i,j)[1];//Green
            rgb[2]=input.at<cv::Vec3b>(i,j)[0];//Blue
            p_s=getProb_MOG(rgb, 3, U_S, COV_S, W_S, 16);
            p_ns=getProb_MOG(rgb, 3, U_NS, COV_NS, W_NS, 16);
            if((p_s/p_ns)>umbral)  result.at<uchar>(i,j)=1;
            else result.at<uchar>(i,j)=0;
        }
    }
    return result;
}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::drawGrid(cv::Mat &image, int stepx, int stepy, cv::Scalar color, bool num_line)
{
    //horizontal lines
    for(int i=0;i<image.rows;i=i+stepy)
    {
        line(image, cv::Point(0,i), cv::Point(image.cols,i), color);
        if(num_line) putText(image,intToString(i),cv::Point(stepy*0.5,i),1,0.6,color);
    }
    //vertical lines
    for(int j=0;j<image.cols;j=j+stepx)
    {
        line(image, cv::Point(j,0), cv::Point(j,image.rows), color);
        if(num_line) putText(image,intToString(j),cv::Point(j,stepx*0.5),1,0.6,color);
    }

}
/*-------------------------------------------------------------------------------------------*/
std::string Preprocessing::intToString(int number)
{
    std::stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::getGaussianFilter1D(float sigma)
{
    int radio=ceil(sigma*3);
    cv::Mat filtro=cv::Mat(1,radio*2+1,CV_32F);
    float val=0;
    for(int x=-radio;x<=radio;x++)
    {
        val=exp(-0.5*(x*x)/(sigma*sigma))*1.0/(sqrt(2*PI)*sigma);
        filtro.at<float>(0,x+radio)=val;
    }
    return filtro;
}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::smoothContour(std::vector<cv::Point2i> &contour, float sigma)
{
    int NSAMPLES=contour.size();
    cv::Mat filtro=Preprocessing::getGaussianFilter1D(sigma);
    cv::Mat matX=cv::Mat(1,NSAMPLES,CV_32F);
    cv::Mat matY=cv::Mat(1,NSAMPLES,CV_32F);
    cv::Mat matXf;
    cv::Mat matYf;
    for(int k=0; k<NSAMPLES;k++)
    {
        matX.at<float>(0,k)=contour[k].x;
        matY.at<float>(0,k)=contour[k].y;
    }
    //cout<<"filtrando"<<endl;
    filter2D(matX,matXf,CV_32F,filtro);
    filter2D(matY,matYf,CV_32F,filtro);
    //cout<<"filtrando"<<endl;
    for(int k=0; k<NSAMPLES;k++)
    {
        contour[k].x=matXf.at<float>(0,k);
        contour[k].y=matYf.at<float>(0,k);
    }
}
/*-------------------------------------------------------------------------------------------*/
//Calculate zeroCrossing points of the curvature according to Curvature Scale Space
std::vector<int> Preprocessing::getZeroCrossingCurvature(std::vector<cv::Point2i> contour, float sigma)
{
    int n=contour.size();
    cv::Mat filter=getGaussianFilter1D(sigma);
    cv::Mat der=(cv::Mat_<float>(1,3)<<-1,0,1);
    cv::Mat der2=(cv::Mat_<float>(1,3)<<-1,2,-1);
    cv::Mat valsx=cv::Mat(1,3*n,CV_32F);
    cv::Mat valsy=cv::Mat(1,3*n,CV_32F);
    cv::Mat dX=cv::Mat(1,3*n,CV_32F);
    cv::Mat dY=cv::Mat(1,3*n,CV_32F);
    cv::Mat ddX=cv::Mat(1,3*n,CV_32F);
    cv::Mat ddY=cv::Mat(1,3*n,CV_32F);
    float *cc=new float[n+1];
    for(int i=0; i<n;i++)
    {
        valsx.at<float>(0,i)=contour[i].x;
        valsy.at<float>(0,i)=contour[i].y;
        valsx.at<float>(0,i+n)=contour[i].x;
        valsy.at<float>(0,i+n)=contour[i].y;
        valsx.at<float>(0,i+2*n)=contour[i].x;
        valsy.at<float>(0,i+2*n)=contour[i].y;
    }
    cv::Mat G;
    cv::Mat dG, ddG;
    //transpose(getGaussianKernel(width, sigma, CV_32F), G);
    //Sobel(G, dG, G.depth(), 1, 0, 3);
    //Sobel(G, ddG, G.depth(), 2, 0, 3);
    filter2D(filter, dG, CV_32F, der);
    filter2D(filter, ddG, CV_32F, der2);
    //flip(dg, dg, 0);
    //flip(ddg, ddg, 0);
    //cv::Point anchor(dg.cols – fwhm -1, dg.rows – 0 – 1);
    filter2D(valsx, dX, CV_32F, dG);
    filter2D(valsy, dY, CV_32F, dG);
    filter2D(valsx, ddX, CV_32F, ddG);
    filter2D(valsy, ddY, CV_32F, ddG);
    float deno=0, nume=0;
    for(int i=n;i<2*n;i++)
    {
        nume=(dX.at<float>(0,i+1)*ddY.at<float>(0,i+1)-ddX.at<float>(0,i+1)*dY.at<float>(0,i+1));
        deno=dX.at<float>(0,i+1)*dX.at<float>(0,i+1)+dY.at<float>(0,i+1)*dY.at<float>(0,i+1);
        deno=pow(deno,2);
        cc[i-n]=nume/(deno+P_EPS);
    }
    cc[n]=cc[0];
    std::vector<int> cp;
    cp.clear();
    for(int i=0;i<n;i++)
    {
        if(cc[i]*cc[i+1]<0)
        {
            cp.push_back(i);
        }
    }
    delete[] cc;
    return cp;
}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::normalizeVector(float *vector, int n, int tipo)
{
    if(n>0 && tipo!=NO_NORMALIZATION)
    {
        int i=0;
        if (tipo==NORMALIZE_ROOT || tipo==NORMALIZE_ROOT_UNIT){
        	float v=1;
        	for(i=0; i<n;i++){
        		if(vector[i]<0) v=-1;
        		else v=1;
        		vector[i]=v*std::sqrt(std::abs(vector[i]));
        	}
        }
        if (tipo==NORMALIZE_MAX ||
        		tipo==NORMALIZE_SUM ||
				tipo==NORMALIZE_UNIT ||
				tipo==NORMALIZE_ROOT_UNIT){
        	float sum=0, norma=0, max=0;
        	max=vector[0];
			for(i=0; i<n;i++)
			{
				if(max<vector[i]) max=vector[i];
				sum+=vector[i];
				norma+=vector[i]*vector[i];//norma vectorial
			}
			if(tipo==NORMALIZE_MAX)
			{
			   norma=max;
			}
			else if(tipo==NORMALIZE_SUM)
			{
			   norma=sum;
			}
			else if(tipo==NORMALIZE_UNIT || tipo==NORMALIZE_ROOT_UNIT)
			{
				norma=sqrt(norma);
			}
			else
			{
				std::cerr<<"Error: Normalization type is incorrect!!"<<std::endl;
				exit(EXIT_FAILURE);
			}
			for(i=0; i<n;i++)
			{
				vector[i]=vector[i]/(norma+P_EPS);
			}
        }
    }
}
/*-------------------------------------------------------------------------------------------*/
void Preprocessing::normalizeVector(float *vector, int n, jmsr::NormMethod tipo){
	int old_tipo=NO_NORMALIZATION;
	if (tipo==jmsr::MAX) old_tipo=NORMALIZE_MAX;
	else if(tipo==jmsr::UNIT) old_tipo=NORMALIZE_UNIT;
	else if(tipo==jmsr::SUM) old_tipo=NORMALIZE_SUM;
	else if(tipo==jmsr::ROOT) old_tipo=NORMALIZE_ROOT;
	else if(tipo==jmsr::ROOT_UNIT) old_tipo=NORMALIZE_ROOT_UNIT;
	else if(tipo==jmsr::NONE) old_tipo=NO_NORMALIZATION;
	else {
		std::cerr<<"Error: Normalization type is incorrect!!"<<std::endl;
		exit(EXIT_FAILURE);
	}
	normalizeVector(vector, n, old_tipo);
}
/*-------------------------------------------------------------------------------------------*/
//Obtiene las posiciones vecinas que son afectadas por un binning --> evita aliasing
void Preprocessing::linearInterBIN(float pos, int *izq, int *der, float *w_izq, float *w_der)
{
    //Modificar de acuerdo  a la implementación en matlab
	/**izq=floor(pos);
	*der=ceil(pos);
	float dist_mid=pos-floor(pos);
	*w_izq=1-dist_mid;
	*w_der=1-*w_izq;
	*w_der=1-*w_izq;*/
	float dist_mid=0;
	*izq=static_cast<int>(floor(pos-0.5));
    *der=static_cast<int>(floor(pos+0.5));
    dist_mid=pos-floor(pos);
    if(dist_mid<0.5)
    {
        *w_izq=0.5-dist_mid;
        *w_der=1-*w_izq;
    }
    else
    {
        *w_der=dist_mid-0.5;
        *w_izq=1-*w_der;
    }
}
/*-------------------------------------------------------------------------------------------*/
//Compute automatically canny thresholds
void Preprocessing::cannyThreshold(const cv::Mat& im, float *l_th, float *h_th)
{
    //im is a gray scale images CV_8UC1
    cv::Mat Gx=(cv::Mat_<float>(3,3)<<-1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat Gy;
    transpose(Gx, Gy);
    cv::Mat im_f;
    cv::Mat imx(im.size(), CV_32F);
    cv::Mat imy(im.size(), CV_32F);
    cv::Mat grad(im.size(), CV_32F);

    im.convertTo(im_f, CV_32F);
    filter2D(im_f, imx, CV_32F, Gx);
    filter2D(im_f, imy, CV_32F, Gy);
    /*-----------------------------------------------------------------*/
    float x=0, y=0;
    float max_grad=0;
    int K=64;
    int i=0, j=0, indx=0;
    for(i=0; i<im.rows;i++)
    {
        for(j=0; j<im.cols; j++)
        {
            x=imx.at<float>(i,j);
            y=imy.at<float>(i,j);
            grad.at<float>(i,j)=sqrt(x*x+y*y);
            if(grad.at<float>(i,j)>max_grad) max_grad=grad.at<float>(i,j);
        }

    }
    /*-----------------------------------------------------------------*/
    int im_size=im.rows*im.cols;
    int *h_mag=new int[K];
    int acum=0;

    for(i=0; i<K;i++) h_mag[i]=0;

    for(i=0; i<im_size; i++)
    {
        indx=(grad.data[i]/max_grad)*K;
        if(indx<K) h_mag[indx]+=1;
    }

    /*-----------------------------------------------------------------*/
    i=K-1;
    acum=h_mag[i];
    float th=0.1*im_size;
    while(acum<th)
    {
        i--;
        acum=acum+h_mag[i];
    }
    *h_th=(i+1)*(max_grad/K);
    *l_th=0.4*(*h_th);
    //cout<<"iii: "<<i<<endl;
    delete[] h_mag;
}

/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::canny(const cv::Mat& im, float l_th, float h_th, float sigma)
{
    //im is a gray scale image
    //Canny return 0 y 255
    if(l_th==-1 || h_th==-1)
    {
        cannyThreshold(im,&l_th, &h_th);
    }
    cv::Mat img;
    cv::Mat ed;
    int size=0;
    size=2*floor(sigma*3)+1;
    cv::GaussianBlur(im, img, cv::Size(size, size), sigma);
    cv::Canny(img,ed, l_th, h_th);

    return ed;
}
/*-------------------------------------------------------------------------------------------*/
cv::Rect getBoundingBox_Binary(const cv::Mat& bin)
{
    //
    int i=0;
    int x_min=0, y_min=0;
    int x_max=0, y_max=0;
    int *h_sum=new int[bin.rows];
    int *v_sum=new int[bin.cols];

    for(i=0; i<bin.rows; i++)
    {
        h_sum[i]=sum(bin.row(i))[0];
        //cout<<h_sum[i]<<endl;
    }
    //cout<<endl;
    for(i=0; i<bin.cols;i++)
    {
        v_sum[i]=sum(bin.col(i))[0];
        //cout<<v_sum[i]<<endl;
    }
    //cout<<endl;
    /*------------------------------------------*/
    y_min=0;
    while(y_min<bin.rows && h_sum[y_min]==0) y_min++;

    y_max=bin.rows-1;
    while(y_max>=0 && h_sum[y_max]==0) y_max--;
    x_min=0;

    while(x_min<bin.cols && v_sum[x_min]==0) x_min++;
    x_max=bin.cols-1;
    while(x_max>=0 && v_sum[x_max]==0) x_max--;

    delete[] h_sum;
    delete[] v_sum;
    /*------------------------------------------*/
    cv::Rect rect(x_min, y_min, x_max-x_min+1, y_max-y_min+1);
    return rect;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::crop(const cv::Mat& im, int padding)
{   //crop an image padding with zeros according to the input variable "padding"
    //im is a binary image
	JUtil::jmsr_assert(im.type()==CV_8UC1," image format must be CV_8UC1");
    cv::Mat cropped;
    cv::Rect rect=getBoundingBox_Binary(im);
    int x_min=rect.x;
    int y_min=rect.y;
    if(padding>0)
    {
        int n_width=2*padding+rect.width;
        int n_height=2*padding+rect.height;
        cropped=cv::Mat::zeros(n_height, n_width, im.type());
        for(int i=0; i<rect.height; i++)
        {
            for(int j=0;j<rect.width;j++)
            {
                cropped.data[(i+padding)*n_width+(j+padding)]=im.data[(i+y_min)*im.cols+(j+x_min)];
            }
        }
    }
    else
    {
        cropped=im(rect).clone();
    }

    return cropped;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::crop(const cv::Mat& im, unsigned char bk_color, int padding)
{
	assert(im.type()==CV_8UC1);
	cv::Rect rect=get_cropping_rect(im, bk_color);
	cv::Mat cropped(2*padding+rect.height,2*padding+rect.width, im.type());
	cropped.setTo(bk_color);
	im(rect).copyTo(cropped(cv::Rect(padding, padding, rect.width, rect.height)));
	return cropped;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::crop_rgb(cv::Mat im, cv::Scalar color, int padding)
{
	assert(im.type()==CV_8UC3);
	cv::Rect rect=get_cropping_rect(im,color);
    cv::Mat cropped(2*padding+rect.height,2*padding+rect.width, im.type());
    cropped.setTo(cv::Scalar(255,255,255));
    im(rect).copyTo(cropped(cv::Rect(padding, padding, rect.width, rect.height)));
    return cropped;
}
/*-------------------------------------------------------------------------------------------*/
cv::Rect Preprocessing::get_cropping_rect(cv::Mat im, unsigned char bk_color)
{
    assert(im.type()==CV_8UC1);
    uchar val;
    cv::Mat mask=cv::Mat::ones(im.size(), CV_8UC1);
    for(int i=0; i<im.rows; i++)
    {
        for(int j=0; j<im.cols; j++)
        {
            val=im.at<uchar>(i,j);
            if(val==bk_color)
            {
                mask.at<uchar>(i,j)=0;
            }
        }
    }
    cv::Rect rect=getBoundingBox_Binary(mask);
    return rect;
}

/*-------------------------------------------------------------------------------------------*/
cv::Rect Preprocessing::get_cropping_rect(cv::Mat im, cv::Scalar color)
{
	assert(im.type()==CV_8UC3);

    cv::Mat mask=cv::Mat::ones(im.size(), CV_8UC1);
    cv::Vec3b val_rgb;
    for(int i=0; i<im.rows; i++)
    {
        for(int j=0; j<im.cols; j++)
        {
            val_rgb=im.at<cv::Vec3b>(i,j);
            if(val_rgb[0]==color[0] && val_rgb[1]==color[1] && val_rgb[2]==color[2])
            {
                mask.at<uchar>(i,j)=0;
            }
        }
    }
    cv::Rect rect=getBoundingBox_Binary(mask);
    return rect;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<cv::Point2d> Preprocessing::getPointsByValue(const cv::Mat& image, unsigned char gray_value){
	std::vector<cv::Point2d> points;
	assert(image.type()==CV_8UC1);
	for(int i=0; i<image.rows; i++){
		for(int j=0; j<image.cols;j++){
			if(image.at<unsigned char>(i,j)==gray_value){
				points.push_back(cv::Point2d(j,i));
			}
		}
	}
	return points;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::getPatch(const cv::Mat& image, int i, int j, cv::Size patch_size, int border_type){
	assert(image.type()==CV_8UC1);
	assert((patch_size.width % 2 ==1) && (patch_size.height % 2 ==1));

	int n_rows=image.rows;
	int n_cols=image.cols;
	int start_x=0, end_x=0, start_y=0, end_y=0;
	int rx=0, ry=0;
	rx=(patch_size.width-1)/2;
	ry=(patch_size.height-1)/2;
	start_x=j-rx;
	end_x=j+rx;
	start_y=i-ry;
	end_y=i+ry;
	cv::Mat patch;
	if(border_type==J_ROI_COMPLETE &&
			(start_x<0 || end_x>n_cols ||
					start_y<0 || end_y>n_rows)){
		return patch;
	}
	else{
		patch.create(patch_size.height, patch_size.width, CV_8UC1);
		patch.setTo(0);
		start_x=std::max(0, start_x);
		start_y=std::max(0, start_y);
		end_x=std::min(n_cols-1, end_x);
		end_y=std::min(n_rows-1, end_y);
		for (int xp=start_x; xp<=end_x; xp++){
			for (int yp=start_y; yp<=end_y; yp++){
				patch.at<unsigned char>(ry+(yp-i),rx+(xp-j))=image.at<unsigned char>(yp,xp);
			}
		}
	}
	return patch;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<std::vector<int> > Preprocessing::getSubsets(int N)
{
    //N<=6, this function support subsets for N<=6, in other cases we will increase the number of bits
    //This function is interesing since a subset is represented by the position of bits 1 in the binary
    //representation of each number from 1 to 2^N
    std::vector<std::vector<int> > subsets;
    int bit[]={1,2,4,8,16,32};
    int n_subsets=pow(2,N);
    for(int i=0;i<n_subsets;i++)
    {
        std::vector<int> subset;
        for(int bit_p=0;bit_p<N;bit_p++)
        {
            if(bit[bit_p] & i) subset.push_back(bit_p+1);
        }
        subsets.push_back(subset);
    }
    return subsets;
}
/*-------------------------------------------------------------------------------------------*/
//compute the center of mass of a binary image
cv::Point2i Preprocessing::getCenterOfMass(cv::Mat image)
{
    cv::Point2i com=cv::Point2i(0,0);
    int suma=sum(image)[0];
    for(int i=0; i<image.rows; i++)
    {
        for(int j=0; j<image.cols;j++)
        {
            com.x+=image.at<uchar>(i,j)*j;
            com.y+=image.at<uchar>(i,j)*i;
        }
    }
    com.x=com.x/suma;
    com.y=com.y/suma;
    return com;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::computeDistance(cv::Point p1, cv::Point p2, int type)
{
    float dx=p1.x-p2.x;
    float dy=p1.y-p2.y;
    float dist=0;
    if (type==JDIST_L1) dist=fabs(dx)+fabs(dy);
    else if (type==JDIST_L2) dist=sqrt(dx*dx+dy*dy);
    return dist;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::computeDistance(cv::Point2f p1, cv::Point2f p2, int type)
{
    float dx=p1.x-p2.x;
    float dy=p1.y-p2.y;
    float dist=0;
    if (type==JDIST_L1) dist=fabs(dx)+fabs(dy);
    else if (type==JDIST_L2) dist=sqrt(dx*dx+dy*dy);
    return dist;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::computeDistance(float *v1, float*v2, int size, int type)
{
    cv::Mat mv1=cv::Mat(1,size,CV_32F,v1);
    cv::Mat mv2=cv::Mat(1,size,CV_32F,v2);
    float dist=0;
    if(type==JDIST_L1) dist=cv::norm(mv1,mv2,cv::NORM_L1);
    else if(type==JDIST_L2) dist=cv::norm(mv1,mv2,cv::NORM_L2);
    return dist;
}
/*-------------------------------------------------------------------------------------------*/
//This function estimates the best bimodal separation using Otsu algorithm
//pos: the best separation point
//conf_sep: the confidence of separation
void Preprocessing::estimateBiModalSeparationOtsu(float *hist, int start, int end, int *pos, float *conf_sep)
{
    float sum=0;
    int hist_size=end-start+1;
    float *hist_prob=new float[hist_size];
    int i=0;
    for(i=start;i<=end; i++) sum+=hist[i];
    for(i=start;i<=end; i++) hist_prob[i-start]=hist[i]/sum;

    float *media=new float[hist_size];
    float *acum=new float[hist_size];
    float m2=0, m1=0;
    float P1=0, P2=0;

   //------------------- computing cumulative prob and media in the firsr group
    media[0]=0;
    acum[0]=hist_prob[0];
    for(i=1;i<hist_size;i++)
    {
        media[i]=i*hist_prob[i]+media[i-1];
        acum[i]=hist_prob[i]+acum[i-1];
    }
    //-------------------------------------------------------------------------
    int t=0;
    float maxVal=-1, val=0;
    float m=0;
    m=media[hist_size-1];
    for(i=0;i<hist_size;i++)
    {
        P1=acum[i];
        P2=1-P1;
        if(P1!=0 && P2!=0)
        {
            m1=media[i]/P1;
            m2=(m-media[i])/P2;
            val=P1*(m1-m)*(m1-m)+P2*(m2-m)*(m2-m);
            val=val/m;
            if(val>maxVal)
            {
                maxVal=val;
                t=i;
            }
        }
    }
    *pos=start+t;
    *conf_sep=maxVal;
    delete[] media;
    delete[] acum;
    delete[] hist_prob;
}

/*-------------------------------------------------------------------------------------------*/
//getHistogramModesOtsuR divide a multimodal histograms recursiverly
std::vector<int> Preprocessing::getHistogramModesOtsuR(float *hist, int start, int end)
{
    float TH_SEP=4;
    std::vector<int> C;
    C.clear();
    if(start<end)
    {
        std::vector<int> A;
        std::vector<int> B;
        int pos=0;
        float conf_sep=0;
        estimateBiModalSeparationOtsu(hist, start, end, &pos, &conf_sep);
        //cout<<"start "<<start<<" end "<<end<<" pos: "<<pos<<" conf_sep: "<<conf_sep<<endl;
        A.clear();
        B.clear();
        if(conf_sep>=TH_SEP)
        {
            if(start<pos) A=getHistogramModesOtsuR(hist, start, pos);
            if(pos+1<end) B=getHistogramModesOtsuR(hist, pos+1, end);
            C.reserve(A.size()+B.size()+1);
            C.insert(C.end(),A.begin(), A.end());
            C.push_back(pos);
            C.insert(C.end(), B.begin(), B.end());
        }


    }
    return C;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<int> Preprocessing::getHistogramModesOtsu(cv::Mat hist_im)
{
    std::vector<int> list_sep;
    int n=hist_im.cols;
    float *hist=new float[n];
    int i=0;
    for(i=0; i<n;i++) hist[i]=hist_im.at<float>(i);
    list_sep=getHistogramModesOtsuR(hist, 0,n-1);
    return list_sep;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<int> Preprocessing::getHistogramLocalMaximums(cv::Mat hist_h)
{
    std::vector<int> local_max;
    local_max.clear();
    cv::Mat fil=(cv::Mat_<float>(1,3)<<0, -1, 1);
    cv::Mat der;
    filter2D(hist_h, der, CV_32F, fil);
    for(int i=1;i<hist_h.cols-1;i++)
    {
        if (der.at<float>(i)>=0 && der.at<float>(i+1)<0)//change the condition for minimus
        {
            local_max.push_back(i);
        }
    }
    return local_max;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<int> Preprocessing::getHistogramLocalMinimums(cv::Mat hist_h)
{
    std::vector<int> local_max;
    local_max.clear();
    cv::Mat fil=(cv::Mat_<float>(1,3)<<0, -1, 1);
    cv::Mat der;
    filter2D(hist_h, der, CV_32F, fil);
    for(int i=1;i<hist_h.cols-1;i++)
    {
        if (der.at<float>(i)<=0 && der.at<float>(i+1)>0) //change the condition for maximums
        {
            local_max.push_back(i);
        }
    }
    return local_max;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<int> Preprocessing::getHistogramValleys(cv::Mat hist_h, int TH_MIN)
{
    std::vector<int> local_max;
    std::vector<int> local_min;
    local_max=getHistogramLocalMaximums(hist_h);
    //---------------------------
    cv::Mat fil=(cv::Mat_<float>(1,3)<<0, -1, 1);
    cv::Mat der;
    filter2D(hist_h, der, CV_32F, fil);
    //---------------------------
    int i_max=0;
    int curr_max_pos=0;
    int sig_max_pos=0;
    int pos_min=0;
    float best_min=INT_MAX;
    float max_1=0, max_2=0, min_1=0;
    float dist=0, avg_max=0;
    curr_max_pos=local_max[0];
    i_max=0;
    while(i_max<(int)local_max.size()-1)
    {
        sig_max_pos=local_max[i_max+1];
        //------------- seek local minimum between curr_max_pos and sig_max_pos
        best_min=INT_MAX;
        for(int i=curr_max_pos;i<=sig_max_pos;i++)
        {
            if (der.at<float>(i)<=0 && der.at<float>(i+1)>0)
            {
                if(hist_h.at<float>(i)<best_min)
                {
                    best_min=hist_h.at<float>(i);
                    pos_min=i;
                }
            }
        }
        //------------------------------------------------------------------------
        max_1=hist_h.at<float>(curr_max_pos);
        max_2=hist_h.at<float>(sig_max_pos);
        min_1=hist_h.at<float>(pos_min);
        avg_max=(max_1+max_2)*0.5;
        dist=avg_max-min_1;
        //------------------------------------------------------------------------
        if(dist>avg_max*0.5 && min_1<TH_MIN) //ok
        {
            local_min.push_back(pos_min);
            curr_max_pos=sig_max_pos;
        }
        else
        {
            if(max_2>max_1) curr_max_pos=sig_max_pos;
        }
        i_max++;
    }
    return local_min;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::getHorizontalProyection(cv::Mat image)
{
    cv::Mat h=cv::Mat::zeros(1,image.cols, CV_32F);
    for(int i=0; i<image.cols;i++)
    {
        h.at<float>(i)=sum(image.col(i))[0];
    }
    return h;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::getVerticalProyection(cv::Mat image)
{
    cv::Mat h=cv::Mat::zeros(1,image.rows, CV_32F);
    for(int i=0; i<image.rows;i++)
    {
        h.at<float>(i)=sum(image.row(i))[0];
    }
    return h;
}
/*-------------------------------------------------------------------------------------------*/
//Compute the ellipse represented by a rotated rect
JEllipse Preprocessing::rotatedRectToEllipse(cv::RotatedRect rect)
{
    cv::Point2f vertices[4];
    cv::Point2f p1, p2;
    float max_len=0, min_len=0, best_ang=0, len=0, ang=0;
    rect.points(vertices);
    max_len=-1;
    min_len=INT_MAX;
    for(int i_v=0; i_v<4;i_v++)
    {
        p1=vertices[i_v];
        p2=vertices[(i_v+1)%4];
        len=Preprocessing::computeDistance(p1, p2);
        ang=atan2(p1.y-p2.y, p1.x-p2.x);
        if(len>max_len)
        {
            best_ang=ang;
            max_len=len;
        }
        if(len<min_len)
        {
            min_len=len;
        }
    }
    best_ang=(best_ang<0)?best_ang+CV_PI:best_ang; //0..PI
    return JEllipse(best_ang*(180.0/CV_PI), max_len*0.5, min_len*0.5, rect.center);
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::getRMSContrast(cv::Mat image)
{
    if(image.channels()!=1) return -1;
    float sum=0;
    float mean=0;
    float val=0;
    int n=image.cols*image.rows;
    for(int i=0; i<n;i++)
    {
        val=(float)image.data[i];
        sum+=val;
    }
    mean=sum/n;
    sum=0;
    //----------------------------------- DSVR
    for(int i=0; i<n;i++)
    {
        val=(float)image.data[i];
        sum+=(val-mean)*(val-mean);
    }
    sum=sum/(image.rows*image.cols);
    return sqrt(sum);
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::math2image(float ang, int tipo)
{
    float ang_r=0;
    float max_ang=360;
    ang_r=ang*(-1);
    if (tipo==JANG_DEGREE) max_ang=360;
    if (tipo==JANG_RADIANS) max_ang=2*CV_PI;
    if(ang_r<0) ang_r=max_ang+ang_r;
    return ang_r;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::image2math(float ang, int tipo)
{
    float ang_r=0;
    float max_ang=360;
    ang_r=ang*(-1);
    if (tipo==JANG_DEGREE) max_ang=360;
    if (tipo==JANG_RADIANS) max_ang=2*CV_PI;
    if(ang_r<0) ang_r=max_ang+ang_r;
    return ang_r;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::getFocusValue(cv::Mat image, int N, int DPI)
{
    if(image.type()!=CV_8UC1) return -1; //not a grayscale image
    //------------------------------------------- Computing derivatives
    cv::Mat filtro_x=(cv::Mat_<float>(1,3)<<0,1,-1);
    cv::Mat filtro_y=(cv::Mat_<float>(3,1)<<0,1,-1);

    cv::Mat image_fx=cv::Mat::zeros(image.size(), CV_32F);;
    cv::Mat image_fy=cv::Mat::zeros(image.size(), CV_32F);;
    cv::Mat image_f;
    cv::Mat image_g=cv::Mat::zeros(image.size(), CV_32F);
    image.convertTo(image_f, CV_32F);

    filter2D(image_f, image_fx, CV_32F,filtro_x);
    filter2D(image_f, image_fy, CV_32F,filtro_y);

    for(int i=0; i<image.rows; i++)
    {
        for(int j=0; j<image.cols; j++)
        {
            image_g.at<float>(i,j)=std::max(abs(image_fx.at<float>(i,j)), abs(image_fy.at<float>(i,j)));
        }
    }
    //------------------------------------------- Computing derivatives
    int size=image.cols*image.rows;
    float *data_g=reinterpret_cast<float*>(image_g.data);
    std::vector<float> v_data_g(data_g, data_g+size);
    std::vector<uchar> v_data_im(image.data, image.data+size);
    //------------------------------------------- Computing dynamic range
    sort(v_data_g.begin(), v_data_g.end());//< a mayor
    sort(v_data_im.begin(), v_data_im.end());//< a mayor

    float dynamic_range=0;
    float avg_menor=0;
    float avg_mayor=0;

    for(int i=0; i<N; i++)
    {
       // cout<<data[i]<<endl;
        avg_menor+=v_data_im[i];
        avg_mayor+=v_data_im[(size-1)-i];
    }
    avg_menor=avg_menor/N;
    avg_mayor=avg_mayor/N;
    dynamic_range=avg_mayor-avg_menor;
    //-------------------------------------------
    //cout<<v_data_g[size-1]<<" "<<avg_menor<< " "<<avg_mayor<<endl;
    float focus=(DPI*v_data_g[size-1]/(dynamic_range));
    return focus;
}
/*-------------------------------------------------------------------------------------------*/
int Preprocessing::detectHorizontalBlackLines(cv::Mat bin_image, std::vector<int> &pos_lines, std::vector<int> &height_lines, float per_black_pixels)
{
    //input is a binary image 1:foreground 0:background
    //return -1 is an error occurs
    if(bin_image.type()!=CV_8UC1) return -1;
    double max_value=0, min_value=0;
    minMaxLoc(bin_image, &min_value, &max_value);
    if(min_value!=0 || max_value!=1) return -1;
    pos_lines.clear();
    height_lines.clear();
    int n_blacks=0;
    float p_blacks=0;
    int *idx_lines=new int[bin_image.rows];
    //--------------------------------------- Detection horizontal lines
    for(int i_row=0; i_row<(int)bin_image.rows; i_row++)
    {
        n_blacks=bin_image.rows-(sum(bin_image.row(i_row))[0]);
        p_blacks=n_blacks/(float)bin_image.rows;
        if(p_blacks>=per_black_pixels) idx_lines[i_row]=1;
        else idx_lines[i_row]=0;
    }

    //--------------------------------------- Grouping lines
    int suma=0;
    int h_line=0;
    for(int i_row=0; i_row<(int)bin_image.rows; i_row++)
    {

        if(idx_lines[i_row]==1)
        {
            suma+=i_row;
            h_line++;
        }
        else
        {
            if(h_line>0)
            {
                pos_lines.push_back(suma/h_line);
                height_lines.push_back(h_line);
                suma=0;
                h_line=0;
            }
        }
    }
    if(h_line>0)
    {
        pos_lines.push_back(suma/h_line);
        height_lines.push_back(h_line);
    }
    //-------------------------------------------------------
    delete[] idx_lines;
    return 0;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::getPercentageAvgBrightness(cv::Mat image, int input_type, float input)
{
    if (image.type()!=CV_8UC1) return -1;
    int N=0;
    int size=image.rows*image.cols;
    if(input_type==ABSOLUTE_VAL) N=input;
    else if(input_type==RELATIVE_VAL) N=std::max(0,std::min(size,(int)(size*input)));
    std::vector<uchar> v_data(image.data, image.data+size);
    float suma=0;
    sort(v_data.begin(), v_data.end());//<
    for(int i=0; i<N; i++) //It takes the N lightest pixels
    {
        suma+=v_data[(size-1)-i];
    }
    suma=suma/(float)N;
    return suma/255.0;
}
/*-------------------------------------------------------------------------------------------*/
float Preprocessing::getPercentageAvgContrast(cv::Mat image, int input_type, float input1, float input2)
{
    if (image.type()!=CV_8UC1) return -1;
    int N1=0, N2=0;
    int size=image.rows*image.cols;
    if(input_type==ABSOLUTE_VAL)
    {
        N1=input1; N2=input2;
    }
    else if(input_type==RELATIVE_VAL)
    {
        N1=std::max(0,std::min(size,(int)(size*input1)));
        N2=std::max(0,std::min(size,(int)(size*input2)));

    }
    std::vector<uchar> v_data(image.data, image.data+size);
    float suma_whitest=0;
    float suma_blackest=0;
    sort(v_data.begin(), v_data.end());//<
    for(int i=0; i<N1; i++) //It takes the N lightest pixels
    {
        suma_whitest+=v_data[(size-1)-i];
    }
    suma_whitest=suma_whitest/N1;

    for(int i=0; i<N2; i++) //It takes the N blackest pixels
    {
        suma_blackest+=v_data[i];
    }
    suma_blackest=suma_blackest/N2;
    return (suma_whitest-suma_blackest)/255.0;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::alignCheckImage(cv::Mat image)
{
    if(image.type()!=CV_8UC1) return cv::Mat();
    cv::Mat im_aligned;
    cv::Mat bin_check;

    threshold(image, bin_check, Preprocessing::umbralOtsu(image),1,cv::THRESH_BINARY);
    cv::Mat stel=getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    morphologyEx(bin_check,bin_check,cv::MORPH_CLOSE,stel,cv::Point(-1,-1),3);
    morphologyEx(bin_check,bin_check,cv::MORPH_DILATE,stel,cv::Point(-1,-1),1);
    bin_check=CComponent::fillHoles(bin_check);
    //---------------------------------------------------------------------------------
    CComponent ccomp;
    ccomp.setImage(bin_check);
    //---------------------------------------------------------------------------------
    cv::Rect rect_check=ccomp.getMinMaxAllBoundingBoxes();
    //---------------------------------------------------------------------------------
    bin_check=ccomp.bwBigger();//The bigger is used to estimate the skew orientation
    ccomp.setImage(bin_check);
    //-----------------------------Verifying the mask around
    cv::RotatedRect rRect= minAreaRect(ccomp.getPoints(0));
    JEllipse jEllipse=Preprocessing::rotatedRectToEllipse(rRect);
    float the_angle=jEllipse.getAngle(); //image coord
    if(the_angle>0) the_angle=180.0-the_angle;
    else if(the_angle<0) the_angle=-the_angle; //
    //--------------------------------
    cv::Mat the_check;
    image(rect_check).copyTo(the_check);
    //--------------------------------- Final rotation
    //--------------------------------- Correct skew according to the ellipse
    cv::Point2f im_center;
    if (the_angle<90) the_angle=-the_angle;
    else the_angle=180-the_angle;
    im_center.x=the_check.cols*0.5;
    im_center.y=the_check.rows*0.5;
    cv::Mat R=getRotationMatrix2D(im_center,the_angle,1);
    warpAffine(the_check,im_aligned, R, the_check.size());
    //-------------------------------- final cropping
    threshold(im_aligned, bin_check, Preprocessing::umbralOtsu(im_aligned),1,cv::THRESH_BINARY);
    morphologyEx(bin_check,bin_check,cv::MORPH_OPEN,stel,cv::Point(-1,-1),3);
    morphologyEx(bin_check,bin_check,cv::MORPH_DILATE,stel,cv::Point(-1,-1),1);
    bin_check=CComponent::fillHoles(bin_check);
    ccomp.setImage(bin_check);
    rect_check=ccomp.getMinMaxAllBoundingBoxes(); //similar as above
    //rect_check=ccomp.getBoundingBox(0);
    im_aligned(rect_check).copyTo(the_check);
    return the_check;
}
/*-------------------------------------------------------------------------------------------*/
cv::Mat Preprocessing::alignCheckImage(cv::Mat image, float the_angle)
{
    if(image.type()!=CV_8UC1) return cv::Mat();
    cv::Mat im_aligned;
    cv::Mat bin_check;

    threshold(image, bin_check, Preprocessing::umbralOtsu(image),1,cv::THRESH_BINARY);
    cv::Mat stel=getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    morphologyEx(bin_check,bin_check,cv::MORPH_CLOSE,stel,cv::Point(-1,-1),3);
    morphologyEx(bin_check,bin_check,cv::MORPH_DILATE,stel,cv::Point(-1,-1),1);
    bin_check=CComponent::fillHoles(bin_check);
    //---------------------------------------------------------------------------------
    CComponent ccomp;
    ccomp.setImage(bin_check);
    //---------------------------------------------------------------------------------
    cv::Rect rect_check=ccomp.getMinMaxAllBoundingBoxes();
    cv::Mat the_check;
    image(rect_check).copyTo(the_check);
    //--------------------------------- Final rotation
    //--------------------------------- Correct skew according to the ellipse
    cv::Point2f im_center;
    if (the_angle<90) the_angle=-the_angle;
    else the_angle=180-the_angle;
    im_center.x=the_check.cols*0.5;
    im_center.y=the_check.rows*0.5;
    cv::Mat R=getRotationMatrix2D(im_center,the_angle,1); //coord math
    warpAffine(the_check,im_aligned, R, the_check.size());
    //-------------------------------- final cropping
    threshold(im_aligned, bin_check, Preprocessing::umbralOtsu(im_aligned),1,cv::THRESH_BINARY);
    morphologyEx(bin_check,bin_check,cv::MORPH_OPEN,stel,cv::Point(-1,-1),3);
    morphologyEx(bin_check,bin_check,cv::MORPH_DILATE,stel,cv::Point(-1,-1),1);
    bin_check=CComponent::fillHoles(bin_check);
    ccomp.setImage(bin_check);
    rect_check=ccomp.getMinMaxAllBoundingBoxes(); //similar as above
    //rect_check=ccomp.getBoundingBox(0);
    im_aligned(rect_check).copyTo(the_check);
    return the_check;
}
/*-------------------------------------------------------------------------------------------*/
//textSmoothing
//This follows the implementation of
// N.W. Strathy "A method for segmentation of touching handwritten numerals"
cv::Mat Preprocessing::textSmoothing(cv::Mat image)
{
    //validando data type
    if(image.type()!=CV_8UC1) return cv::Mat();
    //image must come in binary format 1,0
    //adding extra border
    cv::Mat image_tmp=cv::Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
    cv::Mat image_rsp=cv::Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
    image.copyTo(image_tmp(cv::Rect(1,1,image.cols, image.rows)));

    int is[][5]={{ 0, 0, +1, +1, +1},
                 {-1, +1, -1, 0, +1},
                 { 0, 0, -1, -1, -1},
                 {+1, -1, +1, 0, -1}};


    int js[][5]={{-1, +1, -1, 0, +1},
                 { 0, 0, -1, -1, -1},
                 { +1, -1, +1, 0, -1},
                 { 0, 0, +1, +1, +1}};

    int suma=0;
    int n_masks=4;
    bool igual=false;
    for(int i=image.rows; i>=1; i--)
    {
        for(int j=image.cols; j>=1; j--)
        {

            igual=false;
            for(int idx_mask=0; (idx_mask<n_masks && !igual); idx_mask++)
            {
                suma=0;
                for(int k=0; k<5;k++) suma+=image_tmp.at<uchar>(i+is[idx_mask][k], j+js[idx_mask][k]);
                if (suma==5)
                {
                    image_rsp.at<uchar>(i,j)=1;
                    igual=true;
                }
                else if(suma==0)
                {
                    image_rsp.at<uchar>(i,j)=0;
                    igual=true;
                }
                else
                {
                    image_rsp.at<uchar>(i,j)=image_tmp.at<uchar>(i,j);
                }
            }
        }
    }
    return image_rsp;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<std::string> Preprocessing::splitString(std::string input,char delimiter)
{
    std::vector<std::string> result;
    if(!input.empty()){
		int pos1=-1;
		int pos2=0;
		pos2=input.find_first_of(delimiter, pos1+1);
		while(pos2!=(int)std::string::npos)		{
			result.push_back(input.substr(pos1+1,pos2-pos1-1));
			pos1=pos2;
			pos2=input.find_first_of(delimiter, pos1+1);
		}
		result.push_back(input.substr(pos1+1));
    }
    return result;
}

/*----------------------------------------------------------------------------*/
cv::Mat Preprocessing::toUint8(cv::Mat input)
{
    //input is only one channel
    double minv, maxv;
    cv::minMaxIdx(input, &minv, &maxv);
    cv::Mat adjImage;
    cv::convertScaleAbs(input-minv, adjImage, 255.0 / (maxv-minv));
    return adjImage;

}
/*----------------------------------------------------------------------------*/
cv::Mat Preprocessing::toUint8_3(cv::Mat input)
{
    //input is only 3-channels
    cv::Mat output;
    std::vector<cv::Mat> channels;
    cv::split(input, channels);    
    channels[0]=toUint8(channels[0]);
    channels[1]=toUint8(channels[1]);
    channels[2]=toUint8(channels[2]);
    cv::merge(channels, output);    
    return output;
}
/*----------------------------------------------------------------------------*/
//Gamma corrections, is the flag is false, it returns a float matrix varying between 0..1
cv::Mat Preprocessing::gammaCorrection(cv::Mat image, float gamma, bool FLAG_UCHAR)
{
    cv::Mat  gimage;
    image.convertTo(gimage, CV_32F);
    cv::pow(gimage/255.0, gamma, gimage);
    if(FLAG_UCHAR)
        return Preprocessing::toUint8(gimage);
    else
        return gimage;
}
/*----------------------------------------------------------------------------*/
cv::Mat Preprocessing::illuminationNormalization_Tan_Triggs(cv::Mat input, float gamma, float alpha, float tau)
{

    cv::Mat gray, gimage, gaussian1, gaussian2, dog, aux, output;
    cv::Mat e, einv;
    float mean=0, total=0;
    //--------------------------input must be in a grayscale format
    if(input.channels()==3) cv::cvtColor(input, gray, CV_BGR2GRAY);
    else  input.copyTo(gray);
    //-------------------------------------------------------------
    //1. Gamma correction, compress bright regions, expand dark ones
    gimage=gammaCorrection(gray,gamma,false);
    //2. DoG
    cv::GaussianBlur(gimage,gaussian1,cv::Size(5,5),1.0,1.0,cv::BORDER_REFLECT);
    cv::GaussianBlur(gimage,gaussian2,cv::Size(11,11),2.0,2.0,cv::BORDER_REFLECT);
    dog=gaussian1-gaussian2;
    //3. Contrast Equalization
    //3.a
    cv::pow(abs(dog),alpha,aux);
    total=gray.rows*gray.cols;
    mean=cv::sum(aux)[0];
    mean=std::pow(mean/total,1.0/alpha);
    dog=dog/mean;
    //3.b
    aux=cv::min(abs(dog),tau);
    pow(aux,alpha,aux);
    mean=cv::sum(aux)[0];
    mean=cv::pow(mean/total,1.0/alpha);
    dog=dog/mean;
    //4. Tangent hiperbolic
    dog=dog/tau;
    cv::exp(dog,e);
    einv=1.0/e;
    output=(e-einv)/(e+einv);
    return Preprocessing::toUint8(tau*output);

}
/*----------------------------------------------------------------------------*/
void Preprocessing::grabCutSegmentation(cv::Mat &input, cv::Mat &outmask, cv::Rect rect, int n_iter)
{
    cv::Mat bgdModel;
    cv::Mat fgdModel;
    cv::Mat mask;
    cv::Mat bin_mask;            
    cv::grabCut(input, mask, rect, bgdModel, fgdModel, n_iter, cv::GC_INIT_WITH_RECT);    
    if(mask.empty()|| mask.type()!=CV_8UC1)
    {
         CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    }
    //capturing a binary mask wiht pixeles representing foreground (FGD) and probable foreground (PRG_FGD)
    bin_mask = mask & 1;//Pues solamente interesa el FGD=1 y PRG_FGD=3 (impares)
    bin_mask.copyTo(outmask);
    //input.copyTo(output, bin_mask);
}
/*----------------------------------------------------------------------------*/
void Preprocessing::BGR2oRGB(cv::Mat &input, cv::Mat &output)
{
    std::vector<cv::Mat> out_channels(3);
    for(int i=0; i<3;i++) out_channels.push_back(cv::Mat(input.size(), CV_32FC1));
    std::vector<cv::Mat> in_channels;
    cv::split(input, in_channels);
    cv::Mat red, blue, green;
    in_channels[2].convertTo(red,CV_32FC1);
    in_channels[1].convertTo(green,CV_32FC1);
    in_channels[0].convertTo(blue,CV_32FC1);

    out_channels[0]=0.299*red+0.587*green+0.114*blue;
    out_channels[1]=0.5*red+0.5*green-1.0*blue;
    out_channels[2]=0.866*red-0.866*green+0*blue;
    cv::merge(out_channels,output); //output is a CV_32FC3 image
}

/*----------------------------------------------------------------------------*/

void Preprocessing::equalizeHistogram_color(cv::Mat &in_image, cv::Mat &out_image){
	std::vector<cv::Mat> vec_in_imgs;
	std::vector<cv::Mat> vec_out_imgs(3);
	cv::split(in_image, vec_in_imgs);
	for(int i=0; i<3; i++){
		cv::equalizeHist(vec_in_imgs[i], vec_out_imgs[i]);
	}
	cv::merge(vec_out_imgs, out_image);
}
/**
 * This function obtains the external border of a binary image
 * It is not so effective since, concave contours aren't extracted
 * @param mat_bin
 * @param mat_border
 */
void Preprocessing::bwGetExternalBorder(cv::Mat& mat_bin, cv::Mat& mat_border){
	//mat-bin is a binary image
	double min_val=0, max_val=0;
	cv::minMaxLoc(mat_bin,&min_val, &max_val);
	JUtil::jmsr_assert((min_val==0 && max_val==1 && mat_bin.type()==CV_8UC1)," Incorrect type of the input image (it must be binary)");
	mat_border.create(mat_bin.size(), CV_8UC1);
	mat_border.setTo(0);
	int height=mat_bin.rows;
	int width=mat_bin.cols;
	int i=0, j=0;
	const char border_value=1;
	//horizontal scanning
	for (i=0; i<height; i++){
		j=0;
		while(j<width && mat_bin.at<uchar>(i,j)==0) j++;
		if(j<width && j>=0) mat_border.at<uchar>(i,j)=border_value;
		j=width-1;
		while(j>=0 && mat_bin.at<uchar>(i,j)==0) j--;
		if(j<width && j>=0) mat_border.at<uchar>(i,j)=border_value;
	}
	//vertical scanning
	for (j=0; j<width; j++){
		i=0;
		while(i<height && mat_bin.at<uchar>(i,j)==0) i++;
		if(i<height && i>=0) mat_border.at<uchar>(i,j)=border_value;
		i=height-1;
		while(i>=0 && mat_bin.at<uchar>(i,j)==0) i--;
		if(i<height && i>=0) mat_border.at<uchar>(i,j)=border_value;
	}
}
/**
 * This function obtains the external border of a binary image
 * It is not so effective since, concave contours aren't extracted
 * @param mat_bin
 * @param mat_border
 */
std::vector<cv::Point> Preprocessing::bwGetExternalBorder(cv::Mat& mat_bin){
	//mat-bin is a binary image
	double min_val=0, max_val=0;
	cv::minMaxLoc(mat_bin,&min_val, &max_val);
	JUtil::jmsr_assert((min_val==0 && max_val==1 && mat_bin.type()==CV_8UC1)," Incorrect type of the input image (it must be binary)");
	std::vector<cv::Point> vec_points;
	int height=mat_bin.rows;
	int width=mat_bin.cols;
	int i=0, j=0;
	//horizontal scanning
	for (i=0; i<height; i++){
		j=0;
		while(j<width && mat_bin.at<uchar>(i,j)==0) j++;
		if(j<width && j>=0) vec_points.push_back(cv::Point(j,i));
		j=width-1;
		while(j>=0 && mat_bin.at<uchar>(i,j)==0) j--;
		if(j<width && j>=0) vec_points.push_back(cv::Point(j,i));
	}
	//vertical scanning
	for (j=0; j<width; j++){
		i=0;
		while(i<height && mat_bin.at<uchar>(i,j)==0) i++;
		if(i<height && i>=0) vec_points.push_back(cv::Point(j,i));
		i=height-1;
		while(i>=0 && mat_bin.at<uchar>(i,j)==0) i--;
		if(i<height && i>=0) vec_points.push_back(cv::Point(j,i));
	}
	return vec_points;
}
/**
 *
 * @param image_in
 * @param image_out
 * @param size_out
 * @param size_in  (the image add a white padd)
 *  output is a 8UC1
 */
void Preprocessing::padImage_8UC1(const cv::Mat& image_in, cv::Mat& image_out, int size_out, int size_in, uchar border_value){
	JUtil::jmsr_assert(image_in.type()==CV_8UC1," input image must to be 8UC1");
	cv::Mat image_gray;
	image_in.copyTo(image_gray);

	int target_size=size_in;
	int final_size=size_out;
	float scale_factor=(static_cast<float>(target_size))/std::max(image_gray.rows, image_gray.cols);
	int new_w=image_gray.cols*scale_factor;
	int new_h=image_gray.rows*scale_factor;
	int offset_x=(final_size-new_w)/2;
	int offset_y=(final_size-new_h)/2;

	cv::Mat resized_im;
	cv::resize(image_gray, resized_im, cv::Size(new_w, new_h));
	image_out.create(cv::Size(final_size,final_size),CV_8UC1);
	image_out.setTo(border_value);
	resized_im.copyTo(image_out(cv::Rect(offset_x,offset_y,new_w,new_h)));
}
/**
 *
 * @param image_in: an sketch image (fg=black, bg=white)
 * @param image_out
 * @param size_out
 * @param size_in
 *  output is a 8UC1
 */
void Preprocessing::preprocess_sketch(const cv::Mat& image_in, cv::Mat& image_out, int size_out, int size_in){
	JUtil::jmsr_assert(image_in.type()==CV_8UC1 || image_in.type()==CV_8UC3, " input image must to be 8UC1 or 8UC3");
	cv::Mat image_gray;
	if(image_in.channels()==3){
		cv::cvtColor(image_in, image_gray, CV_BGR2GRAY);
	}
	else{
		image_in.copyTo(image_gray);
	}
	cv::Mat mat_bin;
	float umbral=Preprocessing::umbralOtsu(image_gray);
	cv::threshold(image_gray, mat_bin, umbral,1, cv::THRESH_BINARY_INV);
	if(cv::sum(mat_bin)[0]>0){
		cv::Mat im_crop=Preprocessing::crop(mat_bin, 0,0);
		im_crop=255*(1-im_crop);
		padImage_8UC1(im_crop, image_out, size_out, size_in, 255);
	}
	else{
		cv::resize(mat_bin, image_out, cv::Size(size_out, size_out));
	}
}
/**
 *
 * @param sk_image_in: input sketch
 * @param sk_image_out: output sketch
 * @param width: the  input sketch will be resized to be widthxheight
 * @param height: the  input sketch will be resized to be widthxheight
 * @param resolution: target resolution
 */
void Preprocessing::getSketchWithLowResolution(cv::Mat& sk_image_in, cv::Mat& sk_image_out,
    		int width, int height, int resolution){
	//------------------------------------ image must be a CV_8U1
	cv::Mat im_gray;
	cv::Mat im_bin;
	if(sk_image_in.channels()==3){
		cv::cvtColor(sk_image_in, im_gray, CV_BGR2GRAY);
	}
	else if(sk_image_in.channels()==1){
		sk_image_in.copyTo(im_gray);
	}
	else
	{
		std::cerr<<"The format of the input image is not supported!!"<<std::endl;
		exit(EXIT_FAILURE);
	}
	cv::resize(im_gray, im_gray, cv::Size(width, height));
	std::cout<<"afte resizing"<<std::endl;
	//------------------------------------ image must be binary, we'll use a threshold operations
	cv::threshold(im_gray, im_bin, 128, 1, cv::THRESH_BINARY_INV);
	std::cout<<"threshold ok"<<std::endl;
	//------------------------------------
	int total_cells=resolution*resolution;
	int* n_in_cells=new int[total_cells];
	int* x_in_cells=new int[total_cells];
	int* y_in_cells=new int[total_cells];
	JUtil::setArray<int>(n_in_cells, total_cells, 0);
	JUtil::setArray<int>(x_in_cells, total_cells, 0);
	JUtil::setArray<int>(y_in_cells, total_cells, 0);
	unsigned char val=0;
	int pos_x=0, pos_y=0, pos=0;
	int offset[8][2]={	{-1,-1},
							{-1,0},
							{-1,+1},
							{0,-1},
							{0,+1},
							{+1,-1},
							{+1,0},
							{+1,+1}};
	int n_i=0, n_j=0;
	int neighbor=0;
	int n_pos_x=0;
	int n_pos_y=0;
	std::vector<std::pair<int, int>> vec_edges;
	for (int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			val=im_bin.at<uchar>(i,j);
			if(val==1){
				pos_x=std::round((j/static_cast<float>(width))*resolution);
				pos_y=std::round((i/static_cast<float>(height))*resolution);
				if(pos_x>=0 && pos_x<resolution && pos_y>=0 && pos_y<resolution){
					std::cout<<pos<<std::endl;
					pos=pos_y*resolution+pos_x;
					n_in_cells[pos]++;
					x_in_cells[pos]+=j;
					y_in_cells[pos]+=i;

					for(int i_n=0; i_n<8; i_n++){
						n_i=i+offset[i_n][0];
						n_j=j+offset[i_n][1];
						if(n_i>=0 && n_i<height && n_j>=0 && n_j<width){
							n_pos_x=std::round((n_j/static_cast<float>(width))*resolution);
							n_pos_y=std::round((n_i/static_cast<float>(height))*resolution);
							neighbor=n_pos_y*resolution+n_pos_x;
							val=im_bin.at<uchar>(n_i,n_j);
							if(val==1){
								if(n_pos_x!=pos_x || n_pos_y!=pos_y){
									vec_edges.push_back(std::pair<int, int>(pos, neighbor));
									std::cout<<"Added:"<<pos<<" "<<neighbor<<std::endl;
								}
							}
						}
					}
				}

			}
		}
	}
	std::cout<<"acum ok"<<std::endl;
	//------------------------------------
	for(int i=0; i<total_cells; i++){
		if(n_in_cells[i]>0){
			x_in_cells[i]=x_in_cells[i]/n_in_cells[i];
			y_in_cells[i]=y_in_cells[i]/n_in_cells[i];
		}
	}
	std::cout<<"position ok"<<std::endl;
	//------------------------------------
	std::cout<<"neighs ok: "<<vec_edges.size()<<std::endl;
	//------------------------------------
	int x_1=0, y_1=0;
	int x_2=0, y_2=0;
	std::vector<cv::Point2i> vec_points;
	sk_image_out.create(height, width, CV_8UC1);
	sk_image_out.setTo(0);
	std::cout<<"bres"<<std::endl;
	for(int i=0; i<(int)vec_edges.size(); i++){
		pos=vec_edges[i].first;
		neighbor=vec_edges[i].second;
		x_1=x_in_cells[pos];
		y_1=y_in_cells[pos];

		x_2=x_in_cells[neighbor];
		y_2=y_in_cells[neighbor];
		vec_points=getLinePoints_Bresenham(cv::Point2i(x_1, y_1), cv::Point2i(x_2, y_2));
		setImageValue(sk_image_out, vec_points,1);
	}
	std::cout<<"bres ok"<<std::endl;
}

//--------------------------------------------------------------------------------//
void Preprocessing::imresize(const cv::Mat& im_in, cv::Mat& mat_out, int target_size){
	int higher_dim=std::max(im_in.cols, im_in.rows);
	float factor=(static_cast<float>(target_size))/higher_dim;
	cv::resize(im_in, mat_out, cv::Size(), factor, factor);
}

//--------------------------------------------------------------------------------//
void Preprocessing::normalizeSketch(const cv::Mat& _mat_sketch_in, cv::Mat& mat_sketch_out, int width){
	cv::Mat mat_sketch_in;

	cv::Mat strel=cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
	if(_mat_sketch_in.channels()==3){
		cv::cvtColor(_mat_sketch_in, mat_sketch_in, CV_BGR2GRAY);
	}
	else{
		_mat_sketch_in.copyTo(mat_sketch_in);
	}
	cv::threshold(mat_sketch_in, mat_sketch_in, 128, 1, cv::THRESH_BINARY_INV);
	int n_it=0;
	cv::Mat thin_im=Morphological::thinning_Zhang_Sue(mat_sketch_in, &n_it);
	//std::cout<<"it: "<<n_it<<std::endl;
	if(n_it<width){
		cv::morphologyEx(mat_sketch_in, mat_sketch_in, cv::MORPH_DILATE, strel, cv::Point(-1,-1), width-n_it);
	}
	//mat_sketch_in=Preprocessing::textSmoothing(mat_sketch_in);
	mat_sketch_in.copyTo(mat_sketch_out);
	mat_sketch_out=255-mat_sketch_out*255;
}

