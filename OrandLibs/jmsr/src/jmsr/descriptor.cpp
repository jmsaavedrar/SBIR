/*
This file contains implementations of a variety of descriptors
focusing on orientation features like HELO, SHELO, HOG

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


#include <iostream>
#include <math.h>
#include "descriptor.h"
#include "morphological.h"
#include "ccomponent.h"
#include <cassert>
#ifndef  NO_VLFEAT
	extern "C"
	{
		#include <vl/hog.h>
		#include <vl/lbp.h>
	}
#endif

#ifndef PI
    #define PI 3.14159
#endif
#ifndef EPS
    #define EPS 10e-6
#endif
/*----------------------------------------------------------------------------------------*/
int Params::getDescriptorSize(){return -1;}
int Params::getDescriptorSize(cv::Size s)
{   s.width=s.width;
    return -1;
}
std::string Params::toString(){return "";}

Params::~Params(){}

/*----------------------------------------------------------------------------------------*/
HOGParams::HOGParams()
{
    cell_size=6;
    block_size=3;
    n_channels=9;
    th_intersection=0.5; 
    size=-1;

}
/*----------------------------------------------------------------------------------------*/
HOGParams::HOGParams(int _cell_size, int _block_size, int _n_channels, float _th_intersection)
{
    cell_size=_cell_size;
    block_size=_block_size;
    n_channels=_n_channels;
    th_intersection=_th_intersection;
    size=-1;
}
HOGParams::HOGParams(int _cell_size, int _block_size, int _n_channels, float _th_intersection, cv::Size _image_size)
{
    cell_size=_cell_size;
    block_size=_block_size;
    n_channels=_n_channels;
    th_intersection=_th_intersection;
    image_size=_image_size;
    size=-1;
}
/*----------------------------------------------------------------------------------------*/
int HOGParams::getDescriptorSize(cv::Size s)
{

    int n_cells_row=static_cast<int>(round(s.height/static_cast<float>(cell_size)));
    int n_cells_col=static_cast<int>(round(s.width/static_cast<float>(cell_size)));
    int step=block_size-ceil(block_size*th_intersection);
    int number_of_blocks_row=ceil((1+n_cells_row-block_size)/static_cast<float>(step));
    int number_of_blocks_col=ceil((1+n_cells_col-block_size)/static_cast<float>(step));
    int number_of_blocks=number_of_blocks_row*number_of_blocks_col;
    int block_hog_size=n_channels*(block_size*block_size);
    return number_of_blocks*block_hog_size;
}
/*----------------------------------------------------------------------------------------*/
int HOGParams::getDescriptorSize()//to be used only in the implemenation of  VLFeat
{

    return getDescriptorSize(image_size);
}
/*----------------------------------------------------------------------------------------*/
cv::Size HOGParams::getSizeParam()
{
    return image_size;
}

/*----------------------------------------------------------------------------------------*/
std::string HOGParams::toString()
{
    std::string str="";
    str="HOG : cell_size="+Preprocessing::intToString(cell_size);
    str=str+" block_size="+Preprocessing::intToString(block_size);
    str=str+" n_channels="+Preprocessing::intToString(n_channels);
    str=str+" th_intersection="+Preprocessing::intToString(th_intersection);
    return str;
}
/*----------------------------------------------------------------------------------------*/
HOG_SPParams::HOG_SPParams()
{
    cell_size=6;
    block_size=3;
    n_channels=9;
    th_intersection=0.5;
    size=-1;
}

/*----------------------------------------------------------------------------------------*/
HOG_SPParams::HOG_SPParams(int _cell_size, int _block_size, int _n_channels, float _th_intersection, cv::Size _image_size)
{
    cell_size=_cell_size;
    block_size=_block_size;
    n_channels=_n_channels;
    th_intersection=_th_intersection;
    image_size=_image_size;
    size=-1;
}

/*----------------------------------------------------------------------------------------*/
int HOG_SPParams::getDescriptorSize()
{
    cv::Size sp_size; //this supose a 2x2 division
    sp_size.width=static_cast<int>(ceil(image_size.width*0.5));
    sp_size.height=static_cast<int>(ceil(image_size.height*0.5));

    return 4*getDescriptorSize(sp_size);
}
/*----------------------------------------------------------------------------------------*/
std::string HOG_SPParams::toString()
{
    std::string str="";
    str="HOG_SP : cell_size="+Preprocessing::intToString(cell_size);
    str=str+" block_size="+Preprocessing::intToString(block_size);
    str=str+" n_channels="+Preprocessing::intToString(n_channels);
    str=str+" th_intersection="+Preprocessing::intToString(th_intersection);
    return str;
}
/*----------------------------------------------------------------------------------------*/
HELOParams::HELOParams()
{
    n_cells=25;
    n_bins=72;
    n_blocks=1;
    normalization=NORMALIZE_UNIT;
    squared_root=false;
}
/*----------------------------------------------------------------------------------------*/
HELOParams::HELOParams(int _n_cells, int _n_bins, bool sr)
{
    n_cells=_n_cells;
    n_bins=_n_bins;
    n_blocks=1;
    normalization=NORMALIZE_UNIT;
    squared_root=sr;
}
/*----------------------------------------------------------------------------------------*/
HELOParams::HELOParams(int _n_cells, int _n_bins, int _n_blocks, bool sr)
{
    n_cells=_n_cells;
    n_bins=_n_bins;
    n_blocks=_n_blocks;
    normalization=NORMALIZE_UNIT;
    squared_root=sr;
}
/*----------------------------------------------------------------------------------------*/
HELOParams::HELOParams(int _n_cells, int _n_bins, int _n_blocks, int _normalization, bool sr)
{
    n_cells=_n_cells;
    n_bins=_n_bins;
    n_blocks=_n_blocks;
    normalization=_normalization;
    squared_root=sr;
}
/*----------------------------------------------------------------------------------------*/
int HELOParams::getDescriptorSize()
{
    return n_bins*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
int HELOParams::getDescriptorSize(cv::Size s)
{   s.width=s.width;//just to do something
    return n_bins*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
std::string HELOParams::toString()
{
    std::string str="";
    str="HELO : n_cell="+Preprocessing::intToString(n_cells);
    str=str+" n_bins="+Preprocessing::intToString(n_bins);
    str=str+" n_blocks="+Preprocessing::intToString(n_blocks);
    return str;
}
/*----------------------------------------------------------------------------------------*/
SHELO_MSParams::SHELO_MSParams()
{
    n_cells=25;
    n_bins=36;
    n_blocks=6;
    normalization=NORMALIZE_UNIT;
    n_levels=3;
    n_cells_by_level.clear();
    n_cells_by_level.push_back(25);
    n_cells_by_level.push_back(13);
    n_cells_by_level.push_back(7);
}
/*----------------------------------------------------------------------------------------*/
SHELO_MSParams::SHELO_MSParams(int _n_cells, int _n_bins, int _n_blocks, int _normalization, int _n_levels)
{
    n_cells=_n_cells;
    n_bins=_n_bins;
    n_blocks=_n_blocks;
    normalization=_normalization;
    n_levels=_n_levels;
    n_cells_by_level.clear();
    for(int l=0; l<n_levels; l++)
    {
        n_cells_by_level.push_back((int)std::ceil(n_cells/std::pow(2.0,static_cast<double>(l))));
        //std::cout<<"lelel "<<l<<": "<<n_cells_by_level[l]<<std::endl;
    }
}
/*----------------------------------------------------------------------------------------*/
int SHELO_MSParams::getDescriptorSize()
{
    int size=n_levels*n_bins*n_blocks*n_blocks;
    return size;
}
/*----------------------------------------------------------------------------------------*/
int SHELO_MSParams::getDescriptorSize(cv::Size s)
{
    s.width=s.width;
    int size=n_levels*n_bins*n_blocks*n_blocks;
    return size;
}
/*----------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------*/
int SHELO_MSParams::getDescriptorSizeByLevel()
{
    int size=n_bins*n_blocks*n_blocks;
    return size;
}
/*----------------------------------------------------------------------------------------*/
int SHELO_MSParams::geNCells(int level)
{
    return n_cells_by_level[level];
}
/*----------------------------------------------------------------------------------------*/
std::string SHELO_MSParams::toString()
{
    std::string str="";
    str="SHELO : n_cell="+Preprocessing::intToString(n_cells);
    str=str+" n_bins="+Preprocessing::intToString(n_bins);
    str=str+" n_blocks="+Preprocessing::intToString(n_blocks);
    str=str+" n_levels="+Preprocessing::intToString(n_levels);
    return str;
}
/*----------------------------------------------------------------------------------------*/
SHELO_SPParams::SHELO_SPParams()
{
    int n_cells=25;
    int n_bins=36;
    int normalization=NORMALIZE_UNIT;
    SHELO_SPParams(n_cells, n_bins, normalization, 1);

}
/*----------------------------------------------------------------------------------------*/
SHELO_SPParams::SHELO_SPParams(int _n_cells, int _n_bins,
		int _normalization, int _n_levels, int _d_factor)
{
	int n_blocks=1;
	n_levels=_n_levels;
    for(int l=0; l<n_levels; l++)
    {
    	sp_params.push_back(HELOParams(_n_cells, _n_bins,
    			n_blocks, _normalization,true));
    	n_blocks=n_blocks*_d_factor;
    }
}
/*----------------------------------------------------------------------------------------*/
int SHELO_SPParams::getDescriptorSize()
{
    int size=0;
    for(int i=0; i<n_levels;i++)
    {
    	size+=sp_params[i].getDescriptorSize();
    }
    return size;
}
/*----------------------------------------------------------------------------------------*/
int SHELO_SPParams::getDescriptorSize(cv::Size s)
{
    s.width=s.width;
    int size=0;
    for(int i=0; i<n_levels;i++)
    {
    	size+=sp_params[i].getDescriptorSize();
    }
    return size;
}
/*----------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------*/
int SHELO_SPParams::getDescriptorSizeByLevel(int i)
{
     return sp_params[i].getDescriptorSize();
}

/*----------------------------------------------------------------------------------------*/
std::string SHELO_SPParams::toString()
{
    std::string str="";
    str="SHELO : n_levels="+Preprocessing::intToString(n_levels);
    return str;
}

/*----------------------------------------------------------------------------------------*/
JHOGParams::JHOGParams()
{
    n_blocks=4;
    n_bins=36;
}
/*----------------------------------------------------------------------------------------*/
JHOGParams::JHOGParams(int _n_blocks, int _n_bins)
{
    n_blocks=_n_blocks;
    n_bins=_n_bins;
}
/*----------------------------------------------------------------------------------------*/
int JHOGParams::getDescriptorSize()
{
    return n_bins*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
int JHOGParams::getDescriptorSize(cv::Size s)
{   s.width=s.width;//just to do something
    return n_bins*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
std::string JHOGParams::toString()
{
    std::string str="";
    str="JHOG : n_blocks="+Preprocessing::intToString(n_blocks);
    str=str+" n_bins="+Preprocessing::intToString(n_bins);
    return str;
}
/*----------------------------------------------------------------------------------------*/
JHOGParams2::JHOGParams2()
{
    n_blocks_v=2;
    n_blocks_h=2;
    n_bins=36;
}
/*----------------------------------------------------------------------------------------*/
JHOGParams2::JHOGParams2(int _n_blocks_v, int _n_blocks_h, int _n_bins)
{
    n_blocks_v=_n_blocks_v;
    n_blocks_h=_n_blocks_h;
    n_bins=_n_bins;
}
/*----------------------------------------------------------------------------------------*/
int JHOGParams2::getDescriptorSize()
{
    return n_bins*n_blocks_v*n_blocks_h;
}
/*----------------------------------------------------------------------------------------*/
int JHOGParams2::getDescriptorSize(cv::Size s)
{   s.width=s.width;//just to do something
    return n_bins*n_blocks_v*n_blocks_h;
}
/*----------------------------------------------------------------------------------------*/
std::string JHOGParams2::toString()
{
    std::string str="";
    str="JHOG : n_blocks_h="+Preprocessing::intToString(n_blocks_h);
    str=str+" n_blocks_v="+Preprocessing::intToString(n_blocks_v);
    str=str+" n_bins="+Preprocessing::intToString(n_bins);
    return str;
}
/*----------------------------------------------------------------------------------------*/
ConcavityParams::ConcavityParams()
{
    n_blocks=2;
}
/*----------------------------------------------------------------------------------------*/
ConcavityParams::ConcavityParams(int _n_blocks)
{
    n_blocks=_n_blocks;
}
/*----------------------------------------------------------------------------------------*/
int ConcavityParams::getDescriptorSize()
{
    return 16*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
int ConcavityParams::getDescriptorSize(cv::Size s)
{   s.width=s.width;//just to do something
    return 16*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
std::string ConcavityParams::toString()
{
    std::string str="";
    str="Concavity : n_blocks="+Preprocessing::intToString(n_blocks);
    return str;
}
/*----------------------------------------------------------------------------------------*/
DTParams::DTParams()
{
    n_blocks=2;
    im_size=cv::Size(30,30);
}
/*----------------------------------------------------------------------------------------*/
DTParams::DTParams(int _n_blocks, cv::Size _im_size)
{
    n_blocks=_n_blocks;
    im_size=_im_size;
}

/*----------------------------------------------------------------------------------------*/
DTParams::DTParams(int _n_blocks, int width, int height)
{
    n_blocks=_n_blocks;
    im_size=cv::Size(width, height);
}
/*----------------------------------------------------------------------------------------*/
int DTParams::getDescriptorSize()
{
    return im_size.width*im_size.height*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
int DTParams::getDescriptorSize(cv::Size s)
{   s.width=s.width;//just to do something
    return im_size.width*im_size.height*n_blocks*n_blocks;
}
/*----------------------------------------------------------------------------------------*/
std::string DTParams::toString()
{
    std::string str="";
    str="Concavity : n_blocks="+Preprocessing::intToString(n_blocks);
    str="            width="+Preprocessing::intToString(im_size.width);
    str="            height="+Preprocessing::intToString(im_size.height);
    return str;
}
/*----------------------------------------------------------------------------------------*/
LBPParams::LBPParams()
{
    num_cells=10;
    quantize_value=58;
    radius=1;
    n_neighbors=8;
}
/*----------------------------------------------------------------------------------------*/
LBPParams::LBPParams(int _num_cells)
{
    num_cells=_num_cells;
    quantize_value=59;
    radius=1;
    n_neighbors=8;
}
/*----------------------------------------------------------------------------------------*/
LBPParams::LBPParams(int _num_cells, int _quantize_value, int _radius, int _n_neighbors)
{
    num_cells=_num_cells;
    quantize_value=_quantize_value;
    radius=_radius;
    n_neighbors=_n_neighbors;
}
/*----------------------------------------------------------------------------------------*/
int LBPParams::getDescriptorSize()
{
    return num_cells*num_cells*quantize_value;
}
/*----------------------------------------------------------------------------------------*/
int LBPParams::getDescriptorSize(cv::Size s)
{
    s.width=s.width;
    return num_cells*num_cells*quantize_value;
}
/*----------------------------------------------------------------------------------------*/
std::string LBPParams::toString()
{
    std::string str="";
    str="LBP: cell_size = "+Preprocessing::intToString(num_cells);
    str+="    quantize_value = "+Preprocessing::intToString(quantize_value);
    return str;
}
/*----------------------------------------------------------------------------------------*/
CLDParams::CLDParams()
{
    num_cells_x=8;
    num_cells_y=8;
}

/*----------------------------------------------------------------------------------------*/
CLDParams::CLDParams(int n_cells_y, int n_cells_x)
{
    num_cells_y=n_cells_y;
    num_cells_x=n_cells_x;
}
/*----------------------------------------------------------------------------------------*/
int CLDParams::getDescriptorSize()
{
    return num_cells_x*num_cells_y;
}
/*----------------------------------------------------------------------------------------*/
int CLDParams::getDescriptorSize(cv::Size s)
{
    s.width=s.width;
    return num_cells_x*num_cells_y;
}
/*----------------------------------------------------------------------------------------*/
std::string CLDParams::toString()
{
    std::string str="";
    str="Grid: n_cells_x="+Preprocessing::intToString(num_cells_x);
    str+="     n_cells_y="+Preprocessing::intToString(num_cells_y);
    return str;
}

/*----------------------------------------------------------------------------------------*/
Descriptor::Descriptor()
{
}
/*---Compute de centroid distance descriptor---------------------------*/
/*-Using sample points of a specific shape contour---------------------*/
float *Descriptor::getCentroidDistance(int *vector_x, int *vector_y, int vector_size)
{
    int i=0;
    int x_c=0, y_c=0;
    float *output=new float[vector_size];
    /*-------------Computing centroids------------------------------------------*/
    for (i=0;i<vector_size;i++)
    {
        x_c=x_c+vector_x[i];
        y_c=y_c+vector_y[i];
    }
    x_c=x_c/vector_size;
    y_c=y_c/vector_size;
    /*--------------------------------------------------------------------------*/
    float d=0;
    for (i=0;i<vector_size;i++)
    {
        d=pow(float(vector_x[i]-x_c),2)+pow(float(vector_y[i]-y_c),2);
        d=sqrt(d);
        output[i]=d;
    }
    return output;

}
/*---------------------------------------------------------------------------*/
/*-input two set of points, for x-coordinates and for y-coordinates----------*/
void Descriptor::sampligByArcLenght(int *&vector_x, int *&vector_y, int vector_size, int number_of_samples)
{
    /*-----------------------------------------------------------------------*/
    int *s_vector_x=new int[number_of_samples];
    int *s_vector_y=new int[number_of_samples];
    /*-----------------------------------------------------------------------*/
    float *perim; //it's the perimeter
    float total_perim=0;
    perim=computerPerim2(vector_x, vector_y, vector_size, total_perim);
    //std::cout<<"Perimetro calculado: "<<total_perim<<std::endl;
    float perim_step=total_perim/(float)number_of_samples;
    float cont_perim=0;
    /*-----------------------------------------------------------------------*/
    s_vector_x[0]=vector_x[0];
    s_vector_y[0]=vector_y[0];
    int k=1;
    int i=1;
    cont_perim=perim_step;
    while(k<number_of_samples) //asumimos que i termina antes de terminar todo
    {
        while(perim[i]<cont_perim) i++;
        s_vector_x[k]=vector_x[i];
        s_vector_y[k]=vector_y[i];
        k++;

        //std::cout<<total_perim<<" "<<cont_perim<<" "<<vector_size<<" "<<i<<" "<<k<<std::endl;
        cont_perim+=perim_step;
    }
    delete[] vector_x;
    delete[] vector_y;
    delete[] perim;

    vector_x=s_vector_x;
    vector_y=s_vector_y;

}
/*---------------------------------------------------------------------------*/
/*----Compute the double perimeter of a curve--------------------------------*/
/*----Returns a vector containing the perimeter for each point---------------*/
float *Descriptor::computerPerim2(int *vector_x, int *vector_y, int vector_size, float &total_perim)
{
    //This function uses an approximating, summing 2 to the side pixel and
    //summing 3 to the diagonal pixel
    float *perim=new float[vector_size];
    perim[0]=0;
    int dx=0, dy=0, i=0, inc=0;
    total_perim=0;
    for(i=1;i<vector_size;i++)
    {
        dx=vector_x[i]-vector_x[i-1];
        dy=vector_y[i]-vector_y[i-1];
        if(dx==0 || dy==0) inc=2;
        else inc=3;
        perim[i]=perim[i-1]+inc;
        total_perim=total_perim+inc;
    }
    return perim;
}

/*--------------------------------------------------------------------------------------*/
float *Descriptor::fourierDescriptor(int *vector_x, int *vector_y, int vector_size, int number_of_samples)
{
    int descriptor_size=int(number_of_samples/2.0);
    float  *centroid_distances;
    cv::Mat m_vector;
    cv::Mat partes[2]; //aqu'i se recupera la parte real e imaginaria de los valores de fourier

    //number_of_samples is the number of point taken from the contour
    //The resulting vector size is a half of the number_of_samples
    //1. realizar el sampling usando the arclength approach
    sampligByArcLenght(vector_x, vector_y, vector_size, number_of_samples);
    centroid_distances=getCentroidDistance(vector_x,vector_y,number_of_samples);
    //3. calcular la dft sobre el muestreo
    //Utilizamos la clase Mat de opencv
    m_vector=cv::Mat(1,number_of_samples,CV_32F, centroid_distances);
    //4. mantener solamente la mitad del descriptor  considerando que es real
    cv::dft(m_vector, m_vector,cv::DFT_COMPLEX_OUTPUT); //applying fourier transform
    cv::split(m_vector, partes); //dividing m_vector into real an imaginary components
    cv::magnitude(partes[0], partes[1],m_vector); //computing the magnitude of the complex components
    //At this point m_vector represents the magnitude fo the FT descriptor
    //We gonna kept the half of the whole vector and then we normalize the vector using the first component
    m_vector=m_vector(cv::Range::all(),cv::Range(0, descriptor_size+1));
    m_vector=m_vector/m_vector.at<float>(0,0);
    m_vector=m_vector(cv::Range::all(),cv::Range(1, descriptor_size+1));

    int i=0;
    float *descriptor=new float[descriptor_size];
    for(i=0;i<descriptor_size;i++)
    {
        descriptor[i]=m_vector.at<float>(0,i);
    }

    delete[] centroid_distances;

    return descriptor;
}
/*--------------------------------------------------------------------------------------*/
float *Descriptor::fourierDescriptor(std::vector<cv::Point2i> contour, int number_of_samples)
{
    //requiere el contorno del objeto, ver la clase CComponent para tal fin
    int vector_size=contour.size();
    int *vector_x = new int[vector_size];
    int *vector_y = new int[vector_size];
    float  *centroid_distances;
    cv::Mat m_vector;
    cv::Mat partes[2]; //aqu'i se recupera la parte real e imaginaria de los valores de fourier

    int i=0;
    /*---------------------------------------------------------------------------------*/
    for (i=0;i<vector_size;i++)
    {
        vector_x[i]=contour[i].x;
        vector_y[i]=contour[i].y;
    }
    /*---------------------------------------------------------------------------------*/
    int descriptor_size=int(number_of_samples/2.0);
    //number_of_samples is the number of point taken from the contour
    //The resulting vector size is a half of the number_of_samples
    //1. realizar el sampling usando the arclength approach
    sampligByArcLenght(vector_x, vector_y, vector_size, number_of_samples);
    centroid_distances=getCentroidDistance(vector_x,vector_y,number_of_samples);
    //3. calcular la dft sobre el muestreo
    //Utilizamos la clase Mat de opencv
    m_vector=cv::Mat(1,number_of_samples,CV_32F, centroid_distances);
    //4. mantener solamente la mitad del descriptor  considerando que es real
    cv::dft(m_vector, m_vector,cv::DFT_COMPLEX_OUTPUT); //applying fourier transform
    cv::split(m_vector, partes); //dividing m_vector into real an imaginary components
    cv::magnitude(partes[0], partes[1],m_vector); //computing the magnitude of the complex components
    //At this point m_vector represents the magnitude fo the FT descriptor
    //We gonna kept the half of the whole vector and then we normalize the vector using the first component
    m_vector=m_vector(cv::Range::all(),cv::Range(0, descriptor_size+1));
    m_vector=m_vector/m_vector.at<float>(0,0);
    m_vector=m_vector(cv::Range::all(),cv::Range(1, descriptor_size+1));

    /*---------------------------------------------------------------------------------*/
    float *descriptor=new float[descriptor_size];
    for(i=0;i<descriptor_size;i++)
    {
        descriptor[i]=m_vector.at<float>(0,i);
    }
    /*---------------------------------------------------------------------------------*/

    delete[] vector_x;
    delete[] vector_y;
    delete[] centroid_distances;
    return descriptor;
}
/*-------------------------------------------------------------------------------------*/
float *Descriptor::getHOGDescriptor(cv::Mat image1, int number_of_bins)
{
    /*the input must be a  grayscale image
      */
    cv::Mat image=image1.clone();
    cv::Mat sobel_x;//x gradient
    cv::Mat sobel_y;//y gradient
    cv::Mat orienta;//orientation

    cv::equalizeHist(image, image);
    cv::GaussianBlur(image, image, cv::Size(3,3), 0.5,0.5);


    cv::Sobel(image, sobel_x,CV_32F, 1,0,3,1,0,cv::BORDER_REFLECT101);
    cv::Sobel(image, sobel_y,CV_32F, 0,1,3,1,0,cv::BORDER_REFLECT101);


    orienta=cv::Mat(sobel_x.size(), CV_32F);

    int i=0,j=0;
    float val=0;
    /*------------------------------------------------------------*/
    //orienta goes from 0 to 2*pi
    for(i=0;i<orienta.size().height;i++)
    {
        for(j=0;j<orienta.size().width;j++)
        {
            val=atan2(sobel_y.at<float>(i,j), sobel_x.at<float>(i,j));
            val=((val<0)?val+2*PI:val); // angulo de orientacion entre 0 y 360
            val=val-PI/2;//obtener la orientacion real
            val=(val<0)?val+2*PI:val;
            orienta.at<float>(i,j)=val;
        }
    }
    //std::cout<<std::endl;
    /*------------------------------------------------------------*/
    int channels[]= {0};
    int hsize[1];
    hsize[0]=number_of_bins;
    float hranges[2];
    hranges[0]=0.0;
    hranges[1]=2*PI;
    const float *ranges[1];
    ranges[0]=hranges;

    cv::Mat hist;
    cv::calcHist(&orienta, 1, channels, cv::Mat(), hist,1, hsize, ranges);
    /*------------------------------------------------------------*/
    hist.convertTo(hist,CV_32F);
    //std::cout<<hist<<std::endl;
    float suma=cv::sum(hist)[0];
    //float norma=cv::norm(hist, cv::NORM_L2);
    float *des=new float[number_of_bins];
    for(i=0;i<number_of_bins;i++)
    {
        des[i]=hist.at<float>(i,0);
        des[i]=des[i]/(suma+EPS);
    }
    return des;
}
/*-------------------------------------------------------------------------------------*/
//This version is used when the descriptor array is already allocated
void Descriptor::getHOGDescriptor(cv::Mat image1, int number_of_bins, float *des)
{
    /*the input must be a  grayscale image
      */
    cv::Mat image;
    image1.copyTo(image);
    cv::Mat sobel_x;//x gradient
    cv::Mat sobel_y;//y gradient
    cv::Mat orienta;//orientation

    cv::equalizeHist(image, image);
    cv::GaussianBlur(image, image, cv::Size(3,3), 0.5,0.5);


    cv::Sobel(image, sobel_x,CV_32F, 1,0,3,1,0,cv::BORDER_REFLECT101);
    cv::Sobel(image, sobel_y,CV_32F, 0,1,3,1,0,cv::BORDER_REFLECT101);


    orienta=cv::Mat(sobel_x.size(), CV_32F);

    int i=0,j=0;
    float val=0;
    /*------------------------------------------------------------*/
    //orienta goes from 0 to 2*pi
    for(i=0;i<orienta.size().height;i++)
    {
        for(j=0;j<orienta.size().width;j++)
        {
            val=atan2(sobel_y.at<float>(i,j), sobel_x.at<float>(i,j));
            val=((val<0)?val+2*PI:val); // angulo de orientacion entre 0 y 360
            val=val-PI/2;//obtener la orientacion real
            val=(val<0)?val+2*PI:val;
            orienta.at<float>(i,j)=val;
            if(std::isnan(val))
            {
                std::cout<<"NAN number"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

        }
    }
    //std::cout<<std::endl;
    /*------------------------------------------------------------*/
    int channels[]= {0};
    int hsize[1];
    hsize[0]=number_of_bins;
    float hranges[2];
    hranges[0]=0.0;
    hranges[1]=2*PI;
    const float *ranges[1];
    ranges[0]=hranges;

    cv::Mat hist;
    cv::calcHist(&orienta, 1, channels, cv::Mat(), hist,1, hsize, ranges);

    /*------------------------------------------------------------*/
    hist.convertTo(hist,CV_32F);
    //std::cout<<hist<<std::endl;
    float suma=cv::sum(hist)[0];
    //float norma=cv::norm(hist, cv::NORM_L2);
    for(i=0;i<number_of_bins;i++)
    {
        des[i]=hist.at<float>(i,0);
        des[i]=des[i]/(suma+EPS);
    }
}

/*-------------------------------------------------------------------------------------*/
float *Descriptor::getHOGDescriptorLocal(cv::Mat image1, int number_of_bins)
{
      /*the input must be a  grayscale image
      */
    cv::Mat image=image1.clone();
    cv::Mat sobel_x;
    cv::Mat sobel_y;
    cv::Mat orienta;

    //std::cout<<image<<std::endl;

    cv::equalizeHist(image, image);
    cv::GaussianBlur(image, image, cv::Size(3,3), 0.5,0.5);


    cv::Sobel(image, sobel_x,CV_16S, 1,0,3,1,0,cv::BORDER_REFLECT101);
    cv::Sobel(image, sobel_y,CV_16S, 0,1,3,1,0,cv::BORDER_REFLECT101);

    sobel_x.convertTo(sobel_x,CV_32F);//tener cuidado en los tipos de datos
    sobel_y.convertTo(sobel_y,CV_32F);//tener cuidado en los tipos de datos


    orienta=cv::Mat(sobel_x.size(), CV_32F);
    int i=0,j=0;
    float val=0;
    /*------------------------------------------------------------*/

    for(i=0;i<orienta.size().height;i++)
    {
        for(j=0;j<orienta.size().width;j++)
        {
            val=atan2(sobel_y.at<float>(i,j), sobel_x.at<float>(i,j));
            val=((val<0)?val+2*PI:val);
            val=val-PI/2            ;
            val=(val<0)?val+2*PI:val;
            orienta.at<float>(i,j)=val;
        }
    }
    //std::cout<<std::endl;
    /*------------------------------------------------------------*/
    int channels[]= {0};
    int hsize[1];
    hsize[0]=number_of_bins;
    float hranges[2];
    hranges[0]=0.0;
    hranges[1]=2*PI;
    const float *ranges[1];
    ranges[0]=hranges;

    int width2=int(orienta.size().width*0.5);
    int height2=int(orienta.size().height*0.5);
    int width=int(orienta.size().width);
    int height=int(orienta.size().height);

    cv::Mat orientaL1=orienta(cv::Range(0,height2), cv::Range(0,width2));
    cv::Mat orientaL2=orienta(cv::Range(0,height2), cv::Range(width2, width));
    cv::Mat orientaL3=orienta(cv::Range(height2, height), cv::Range(0,width2));
    cv::Mat orientaL4=orienta(cv::Range(height2, height), cv::Range(width2,width));
    cv::Mat hist1;
    cv::Mat hist2;
    cv::Mat hist3;
    cv::Mat hist4;
    cv::calcHist(&orientaL1, 1, channels, cv::Mat(), hist1,1, hsize, ranges);
    cv::calcHist(&orientaL2, 1, channels, cv::Mat(), hist2,1, hsize, ranges);
    cv::calcHist(&orientaL3, 1, channels, cv::Mat(), hist3,1, hsize, ranges);
    cv::calcHist(&orientaL4, 1, channels, cv::Mat(), hist4,1, hsize, ranges);

    /*------------------------------------------------------------*/
    hist1.convertTo(hist1,CV_16S);
    hist2.convertTo(hist2,CV_16S);
    hist3.convertTo(hist3,CV_16S);
    hist4.convertTo(hist4,CV_16S);


    //std::cout<<hist<<std::endl;
    float total1=cv::sum(hist1)[0];
    float total2=cv::sum(hist2)[0];
    float total3=cv::sum(hist3)[0];
    float total4=cv::sum(hist4)[0];
    int k=0;
    float *des=new float[number_of_bins*4];
    for(i=0;i<number_of_bins;i++)
    {
        des[i]=hist1.at<short int>(i,0);
        des[i]=des[i]/(total1+EPS);

        k=number_of_bins+i;
        des[k]=hist2.at<short int>(i,0);
        des[k]=des[k]/(total2+EPS);

        k=2*number_of_bins+i;
        des[k]=hist3.at<short int>(i,0);
        des[k]=des[k]/(total3+EPS);

        k=3*number_of_bins+i;
        des[k]=hist4.at<short int>(i,0);
        des[k]=des[k]/(total4+EPS);
    }

     return des;
}
/*---------------------------------------------------------------------------*/
float *Descriptor::getHOGLocal_2x1(cv::Mat image1, int number_of_bins)
{
      /*the input must be a  grayscale image
      */
    cv::Mat image;
    cv::resize(image1, image, cv::Size(60,60));
    //image1.copyTo(image);
    cv::Mat sobel_x;
    cv::Mat sobel_y;
    cv::Mat orienta;

    //std::cout<<image<<std::endl;

    cv::equalizeHist(image, image);
    cv::GaussianBlur(image, image, cv::Size(3,3), 0.5,0.5);


    cv::Sobel(image, sobel_x,CV_32F, 1,0,3,1,0,cv::BORDER_REFLECT101);
    cv::Sobel(image, sobel_y,CV_32F, 0,1,3,1,0,cv::BORDER_REFLECT101);

    orienta=cv::Mat(sobel_x.size(), CV_32F);
    int i=0,j=0;
    float val=0;
    /*------------------------------------------------------------*/

    for(i=0;i<orienta.size().height;i++)
    {
        for(j=0;j<orienta.size().width;j++)
        {
            val=atan2(sobel_y.at<float>(i,j), sobel_x.at<float>(i,j));
            val=((val<0)?val+2*PI:val);
            val=val-PI/2;
            val=(val<0)?val+2*PI:val;
            orienta.at<float>(i,j)=val;
        }
    }
    //std::cout<<std::endl;
    /*------------------------------------------------------------*/
    int channels[]= {0};
    int hsize[1];
    hsize[0]=number_of_bins;
    float hranges[2];
    hranges[0]=0.0;
    hranges[1]=2*PI;
    const float *ranges[1];
    ranges[0]=hranges;


    int height2=int(orienta.size().height*0.5);
    int width=int(orienta.size().width);
    int height=int(orienta.size().height);

    cv::Mat orientaL1=orienta(cv::Range(0,height2), cv::Range(0,width));
    cv::Mat orientaL2=orienta(cv::Range(height2, height), cv::Range(0,width));
    cv::Mat hist1;
    cv::Mat hist2;

    cv::calcHist(&orientaL1, 1, channels, cv::Mat(), hist1,1, hsize, ranges);
    cv::calcHist(&orientaL2, 1, channels, cv::Mat(), hist2,1, hsize, ranges);

    /*------------------------------------------------------------*/
    hist1.convertTo(hist1,CV_16S);
    hist2.convertTo(hist2,CV_16S);


    //std::cout<<hist<<std::endl;
    float total1=cv::sum(hist1)[0];
    float total2=cv::sum(hist2)[0];
    int k=0;
    float *des=new float[number_of_bins*2];
    for(i=0;i<number_of_bins;i++)
    {
        des[i]=hist1.at<short int>(i,0);
        des[i]=des[i]/(total1+EPS);

        k=number_of_bins+i;
        des[k]=hist2.at<short int>(i,0);
        des[k]=des[k]/(total2+EPS);
    }

     return des;
}
/*-------------------------------------------------------------------------*/
//The simplest Histogram of orientation using 2*PI values
// with spatial division
float *Descriptor::getJHOGLocalDescriptor(cv::Mat image1, JHOGParams params)
{

    //In this point is not necessary to clone the image, since image is cloned in getHOGDescriptor
    cv::Mat image;
    cv::resize(image1,image, cv::Size(48,64));//redimensionamos

    int block_size_row=floor(image.rows/static_cast<float>(params.n_blocks));
    int block_size_col=floor(image.cols/static_cast<float>(params.n_blocks));
    int i_start=0, i_end=0;
    int j_start=0, j_end=0;
    //allocate memory for  the descriptor
    std::vector<float*> deses(params.n_blocks*params.n_blocks);
    float *all_des=new float[params.getDescriptorSize()];
    //
    int k=0;
    for(int i_block=0; i_block<params.n_blocks; i_block++)
    {
        i_start=i_block*block_size_row;
        i_end=(i_block+1)*block_size_row;
        if(i_block==params.n_blocks-1) i_end=image.rows;

        for(int j_block=0; j_block<params.n_blocks; j_block++)
       {
            j_start=j_block*block_size_col;
            j_end=(j_block+1)*block_size_col;
            if(j_block==params.n_blocks-1) j_end=image.cols;

            deses[k++]=getHOGDescriptor(image(cv::Range(i_start, i_end),
                                            cv::Range(j_start, j_end)), params.n_bins);

        }
    }
    k=0;
    for(int i=0; i<params.n_blocks*params.n_blocks; i++)
    {
        for(int j=0; j<params.n_bins; j++)
        {
            all_des[k++]=deses[i][j];
        }
        delete[] deses[i];
    }

    return all_des;
}
/*-------------------------------------------------------------------------*/
//Distance Descriptor
float *Descriptor::getDistanceTransformDescriptor(cv::Mat _image, cv::Size im_size)
{
    cv::Mat image;
    cv::Mat bin;
    cv::Mat dt;
    int size_des=im_size.width*im_size.height;
    float *des=new float[size_des];
    cv::resize(_image, image, im_size);
    float umbral=0;
    umbral=Preprocessing::umbralOtsu(image);
    cv::threshold(image, bin, umbral, 1, cv::THRESH_BINARY);
    cv::distanceTransform(bin,dt , CV_DIST_L1, 3);
    cv::convertScaleAbs(dt,dt);   
    for(int j=0; j<size_des; j++)
    {
        des[j]=(float)dt.data[j];
    }
    Preprocessing::normalizeVector(des, size_des,NORMALIZE_UNIT);
    return des;
}
/*-------------------------------------------------------------------------*/
//Distance Local Descriptor
float *Descriptor::getLocalDistanceTransformDescriptor(cv::Mat _image, DTParams params)
{
        //In this point is not necessary to clone the image, since image is cloned in getHOGDescriptor
        cv::Mat image;
        _image.copyTo(image);
        cv::Size im_size=params.im_size;
        //cv::resize(image1,image, cv::Size(60,60));//redimensionamos
        int block_size_row=floor(image.rows/static_cast<float>(params.n_blocks));
        int block_size_col=floor(image.cols/static_cast<float>(params.n_blocks));
        int i_start=0, i_end=0;
        int j_start=0, j_end=0;
        //allocate memory for  the descriptor
        std::vector<float*> deses(params.n_blocks*params.n_blocks);
        float *all_des=new float[params.n_blocks*params.n_blocks*im_size.width*im_size.height];
        //
        int k=0;
        for(int i_block=0; i_block<params.n_blocks; i_block++)
        {
            i_start=i_block*block_size_row;
            i_end=(i_block+1)*block_size_row;
            if(i_block==params.n_blocks-1) i_end=image.rows;

            for(int j_block=0; j_block<params.n_blocks; j_block++)
           {
                j_start=j_block*block_size_col;
                j_end=(j_block+1)*block_size_col;
                if(j_block==params.n_blocks-1) j_end=image.cols;

                deses[k++]=getDistanceTransformDescriptor(image(cv::Range(i_start, i_end),
                                                cv::Range(j_start, j_end)), im_size);

            }
        }
        k=0;
        for(int i=0; i<params.n_blocks*params.n_blocks; i++)
        {
            for(int j=0; j<im_size.width*im_size.height; j++)
            {
                all_des[k++]=deses[i][j];
            }
            delete[] deses[i];
        }

        return all_des;
}


/*-------------------------------------------------------------------------*/
//The simplest Histogram of orientation using 2*PI values and Soft assignment
// with spatial division
float *Descriptor::getSoftJHOGLocalDescriptor(cv::Mat image1, JHOGParams params)
{
    //In this point is not necessary to clone the image, since image is cloned in getHOGDescriptor
    cv::Mat image;
    //image1.copyTo(image);
    cv::resize(image1,image, cv::Size(48,64));//redimensionamos

    //-----------------------------------------------------------------------
    int N_BLOCKS=params.n_blocks;
    float TH_MAG=0;
    std::vector<float*> histograms(N_BLOCKS*N_BLOCKS);

    for(int i=0; i<N_BLOCKS*N_BLOCKS; i++)
    {
        histograms[i]=new float[params.n_bins];
        for(int j=0;j<params.n_bins;j++)
        {
            histograms[i][j]=0;
        }
    }
    //-----------------------------------------------------------------------
    //Usamos sobel para obtener las orientaciones
    cv::Mat Gx=(cv::Mat_<float>(3,3)<<-1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gy;
    transpose(Gx,Gy);

    cv::Mat im_x=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat im_y=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat image_f;
    image.convertTo(image_f, CV_32F);
    filter2D(image_f, im_x, CV_32F, Gx);
    filter2D(image_f, im_y, CV_32F, Gy);
    cv::Mat angles=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat magnitudes=cv::Mat::zeros(image.size(), CV_32F);
    float val=0;
    float max_mag=-1;
    //--------------------------------------------------
    //Calculamos àngulos
    for(int i=0; i<image.rows; i++)
    {
        for(int j=0; j<image.cols; j++)
        {
            val=atan2(im_y.at<float>(i,j), im_x.at<float>(i,j));
            if(val<0) val+=2*PI;//varies between 0 and 2*PI
            val=val-PI/2;//obtener la orientacion real (restamos pi/2)
            val=(val<0)?val+2*PI:val;
            angles.at<float>(i,j)=val;
            magnitudes.at<float>(i,j)=sqrt(im_x.at<float>(i,j)*im_x.at<float>(i,j)+im_y.at<float>(i,j)*im_y.at<float>(i,j));
            if(max_mag<magnitudes.at<float>(i,j)) max_mag=magnitudes.at<float>(i,j);
        }
    }
    //-----------------------------------------------------------------------
    int x_izq=0, x_der=0, y_der=0, y_izq=0;
    float bin=0, x_block=0, y_block=0, certeza=0;
    int bin_izq=0, bin_der=0;
    float w_x_izq=0, w_x_der=0, w_y_der=0, w_y_izq=0;
    float w_bin_izq=0, w_bin_der=0;
    float angle=0, mag=0;
    int idx=0;
    float weight=0;
    float *descriptor_aux;
    //-------For each pixel (i_cell, j_cell)
    for(int i_cell=0; i_cell<image.rows; i_cell++)
    {
        for(int j_cell=0; j_cell<image.cols; j_cell++)
        {
            angle=angles.at<float>(i_cell,j_cell);
            mag=magnitudes.at<float>(i_cell,j_cell);
            /*-----------------------------------------------*/
            bin=(angle/(2.0*PI))*params.n_bins;
            Preprocessing::linearInterBIN(bin, &bin_izq, &bin_der, &w_bin_izq, &w_bin_der);
            if(bin_izq==params.n_bins) bin_izq=0;
            if(bin_izq<0) bin_izq=params.n_bins-1;
            if(bin_der==params.n_bins) bin_der=0;
            if(bin_der<0) bin_der=params.n_bins-1;

            certeza=mag/max_mag;//se debe buscar magnitudes locales
            if(certeza>TH_MAG)
            {
                x_block=round((j_cell/static_cast<float>(image.cols))*N_BLOCKS);
                y_block=round((i_cell/static_cast<float>(image.rows))*N_BLOCKS);
                Preprocessing::linearInterBIN(x_block, &x_izq, &x_der, &w_x_izq, &w_x_der);
                Preprocessing::linearInterBIN(y_block, &y_izq, &y_der, &w_y_izq, &w_y_der);

                if(isValid(x_izq, y_izq, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
                {
                    idx=x_izq+y_izq*N_BLOCKS;
                    weight=w_x_izq*w_y_izq;

                    descriptor_aux=histograms[idx];
                    descriptor_aux[bin_izq]+=mag*weight*w_bin_izq;
                    descriptor_aux[bin_der]+=mag*weight*w_bin_der;
                }

                if(isValid(x_izq, y_der, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
                {
                    idx=x_izq+y_der*N_BLOCKS;
                    weight=w_x_izq*w_y_der;

                    descriptor_aux=histograms[idx];
                    descriptor_aux[bin_izq]+=mag*weight*w_bin_izq;
                    descriptor_aux[bin_der]+=mag*weight*w_bin_der;
                }

                if(isValid(x_der, y_izq, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
                {
                    idx=x_der+y_izq*N_BLOCKS;
                    weight=w_x_der*w_y_izq;

                    descriptor_aux=histograms[idx];
                    descriptor_aux[bin_izq]+=mag*weight*w_bin_izq;
                    descriptor_aux[bin_der]+=mag*weight*w_bin_der;
                }

                if(isValid(x_der, y_der, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
                {
                    idx=x_der+y_der*N_BLOCKS;
                    weight=w_x_der*w_y_der;

                    descriptor_aux=histograms[idx];
                    descriptor_aux[bin_izq]+=mag*weight*w_bin_izq;
                    descriptor_aux[bin_der]+=mag*weight*w_bin_der;
                }
                //--------------------------------------------------------------------------
            }
        }
    }//end for
    //-Normalizing
    std::vector<float> sumas(N_BLOCKS*N_BLOCKS);
    for(int j=0; j<N_BLOCKS*N_BLOCKS;j++)
    {
        sumas[j]=0;
        for(int i=0; i<params.n_bins; i++) sumas[j]+=histograms[j][i];
    }

    float* descriptor=new float[params.n_bins*N_BLOCKS*N_BLOCKS];
    for(int i=0;i<params.n_bins;i++)
    {
        for(int j=0; j<N_BLOCKS*N_BLOCKS;j++)
        {
            descriptor[i+j*params.n_bins]=(histograms[j])[i]/(sumas[j]+EPS);
        }
    }
    //------------------------------------------------------------------------------
    //liberarmos memoria
    for(int j=0; j<N_BLOCKS*N_BLOCKS;j++)
    {
        delete[] histograms[j];
    }
    //------------------------------------------------------------------------------
    return descriptor;
}
/*-------------------------------------------------------------------------*/
float *Descriptor::getConcavidad(cv::Mat image)
{
    //image es binaria

    int width=image.size().width;
    int height=image.size().height;

    int i=0, j=0, ii=0, jj=0;
    int size_des=16;
    float *des=new float[size_des];
    int val=0;
    int arriba=0;
    int abajo=0;
    int izq=0;
    int der=0;

    int cod=0;
    int total_puntos=0;
    for(i=0;i<size_des;i++) des[i]=0;

    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            val=image.at<uchar>(i,j);
            arriba=0;abajo=0;izq=0;der=0;
            if(val==0)
            {
                total_puntos++;
                //arriba
                ii=i;
                while(ii>=0 && arriba==0)
                {
                    arriba=image.at<uchar>(ii,j);
                    ii--;
                }
                //abajo
                ii=i;
                while(ii<height && abajo==0)
                {
                    abajo=image.at<uchar>(ii,j);
                    ii++;
                }
                //izq
                jj=j;
                while(jj>=0 && izq==0)
                {
                    izq=image.at<uchar>(i,jj);
                    jj--;
                }
                //der
                jj=j;
                while(jj<width && der==0)
                {
                    der=image.at<uchar>(i,jj);
                    jj++;
                }

                cod=arriba*8+abajo*4+ izq*2+der*1;
                //std::cout<<arriba<<" "<<abajo<<" "<<izq<<" "<<der<<std::endl;
                des[cod]=des[cod]+1;
            }
        }
    }
    //normalizamos descriptor para hacerlo invariante a escalas    
    float norma=0;
    float suma=0;
    for(i=0;i<size_des;i++)
    {
        suma=suma+des[i];
        norma=norma+des[i]*des[i];
    }
    for(i=0;i<size_des;i++) des[i]=des[i]/(suma+EPS);


    return des;
}
/*-------------------------------------------------------------------------*/
float *Descriptor::getConcavidad8(cv::Mat image)
{
    //image es binaria

    int width=image.size().width;
    int height=image.size().height;

    int i=0, j=0, ii=0, jj=0;
    int size_des=16;
    float *des=new float[size_des];
    int val=0;

    int diag1=0, diag2=0, diag3=0, diag4=0;
    int cod=0;
    int total_puntos=0;
    for(i=0;i<size_des;i++) des[i]=0;

    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            val=image.at<uchar>(i,j);

            diag1=0; diag2=0; diag3=0; diag4=0;
            if(val==0)
            {
                total_puntos++;

                ii=i;
                jj=j;
                while(jj>=0 && ii>=0 && diag1==0)
                {
                    diag1=image.at<uchar>(ii,jj);
                    jj--;
                    ii--;
                }

                //diag2
                ii=i;
                jj=j;
                while(jj<width && ii>=0 && diag2==0)
                {
                    diag2=image.at<uchar>(ii,jj);
                    jj++;
                    ii--;
                }

                //diag3
                ii=i;
                jj=j;
                while(jj>=0 && ii<height && diag3==0)
                {
                    diag3=image.at<uchar>(ii,jj);
                    jj--;
                    ii++;
                }

                //diag4
                ii=i;
                jj=j;
                while(jj<width && ii<height && diag4==0)
                {
                    diag4=image.at<uchar>(ii,jj);
                    jj++;
                    ii++;
                }

                //cod=diag1*128+diag2*64+diag3*32+diag4*16+arriba*8+abajo*4+ izq*2+der*1;
                cod=diag1*8+diag2*4+diag3*2+diag4*1;


                //std::cout<<arriba<<" "<<abajo<<" "<<izq<<" "<<der<<std::endl;
                des[cod]=des[cod]+1;
            }
        }
    }
    //normalizamos descriptor para hacerlo invariante a escalas
    //for(i=0;i<size_des;i++) des[i]=des[i]/total_puntos;

    float norma=0;
    float suma=0;
    for(i=0;i<size_des;i++)
    {
        suma=suma+des[i];
        norma=norma+des[i]*des[i];
    }
    for(i=0;i<size_des;i++) des[i]=des[i]/(suma+EPS);

    return des;
}
/*-------------------------------------------------------------------------*/
//JConcavidad using simple spatial division
float *Descriptor::getJLocalConcavity4(cv::Mat image1, ConcavityParams params)
{
    //In this point is not necessary to clone the image, since image is cloned in getHOGDescriptor
    int n_bins=16;
    cv::Mat image;
    image1.copyTo(image);
    //cv::resize(image1,image, cv::Size(60,60));//redimensionamos
    int block_size_row=floor(image.rows/static_cast<float>(params.n_blocks));
    int block_size_col=floor(image.cols/static_cast<float>(params.n_blocks));
    int i_start=0, i_end=0;
    int j_start=0, j_end=0;
    //allocate memory for  the descriptor
    std::vector<float*> deses(params.n_blocks*params.n_blocks);
    float *all_des=new float[params.getDescriptorSize()];
    //
    int k=0;
    for(int i_block=0; i_block<params.n_blocks; i_block++)
    {
        i_start=i_block*block_size_row;
        i_end=(i_block+1)*block_size_row;
        if(i_block==params.n_blocks-1) i_end=image.rows;

        for(int j_block=0; j_block<params.n_blocks; j_block++)
       {
            j_start=j_block*block_size_col;
            j_end=(j_block+1)*block_size_col;
            if(j_block==params.n_blocks-1) j_end=image.cols;

            deses[k++]=getConcavidad(image(cv::Range(i_start, i_end),
                                            cv::Range(j_start, j_end)));

        }
    }
    k=0;
    for(int i=0; i<params.n_blocks*params.n_blocks; i++)
    {
        for(int j=0; j<n_bins; j++)
        {
            all_des[k++]=deses[i][j];
        }
        delete[] deses[i];
    }

    return all_des;
}
/*-------------------------------------------------------------------------*/
//JConcavidad using simple spatial division with 8-connected
float *Descriptor::getJLocalConcavity8(cv::Mat image1, ConcavityParams params)
{
    //In this point is not necessary to clone the image, since image is cloned in getHOGDescriptor
    int n_bins=16;
    cv::Mat image;
    image1.copyTo(image);
    //cv::resize(image1,image, cv::Size(60,60));//redimensionamos
    int block_size_row=floor(image.rows/static_cast<float>(params.n_blocks));
    int block_size_col=floor(image.cols/static_cast<float>(params.n_blocks));
    int i_start=0, i_end=0;
    int j_start=0, j_end=0;
    //allocate memory for  the descriptor
    std::vector<float*> deses(params.n_blocks*params.n_blocks);
    float *all_des=new float[params.getDescriptorSize()];
    //
    int k=0;
    for(int i_block=0; i_block<params.n_blocks; i_block++)
    {
        i_start=i_block*block_size_row;
        i_end=(i_block+1)*block_size_row;
        if(i_block==params.n_blocks-1) i_end=image.rows;

        for(int j_block=0; j_block<params.n_blocks; j_block++)
       {
            j_start=j_block*block_size_col;
            j_end=(j_block+1)*block_size_col;
            if(j_block==params.n_blocks-1) j_end=image.cols;

            deses[k++]=getConcavidad8(image(cv::Range(i_start, i_end),
                                            cv::Range(j_start, j_end)));

        }
    }
    k=0;
    for(int i=0; i<params.n_blocks*params.n_blocks; i++)
    {
        for(int j=0; j<n_bins; j++)
        {
            all_des[k++]=deses[i][j];
        }
        delete[] deses[i];
    }

    return all_des;
}
/*------------------------------------------------------------------------------*/
float *Descriptor::getBuenDescriptor(cv::Mat _image, int *size_des)
{ //Background=255, foreground=0
    float *final_descriptor;    
    float *descriptorConcaG;
    float *descriptorConca8G;
    float *descriptorOrienta;
    float *descriptorHProfile;    

    //JHOGParams params(3,32);   
    JHOGParams jhog_params(4,16);
    ConcavityParams conca_params(2);

    int size_orienta=jhog_params.getDescriptorSize();//
    int size_concaG=conca_params.getDescriptorSize();
    int size_conca8G=conca_params.getDescriptorSize();//
   int size_hprofile=80;//


    cv::Mat image;
    _image.copyTo(image);

    *size_des=size_concaG+size_conca8G+size_orienta+size_hprofile; //144
    int i=0;
    final_descriptor=new float[*size_des];

    if(image.channels()==3)
    {
        cvtColor(image,image,CV_BGR2GRAY);
    }
    image.convertTo(image, CV_8UC1);

    /*---------------------------------------------------------------------------*/    
    //cv::imshow("antes de", image*255);
    //cv::waitKey(0);
    /*---------------------------------------------------------------------------*/    
    /*---------------------------------------------------------------------------*/
    //std::cout<<"procesando"<<std::endl;
    //std::cout<<"HOGL"<<std::endl;    
    descriptorOrienta=Descriptor::getSoftJHOGLocalDescriptor(image,jhog_params);//number of bins for each region
    //std::cout<<"Concavidad"<<std::endl;    

    image=Preprocessing::preprocessDigit_1(image);// in binary format
    descriptorConcaG=Descriptor::getJLocalConcavity4(image, conca_params);
    //std::cout<<"Concavidad 8"<<std::endl;
    descriptorConca8G=Descriptor::getJLocalConcavity8(image, conca_params);
    //std::cout<<"Profile"<<std::endl;
    descriptorHProfile=Descriptor::getHorizontalProfile(image);//POCO IMPACTO
    //size_conca=16;
    //size_orienta=32*4;

    //HOGDalal

    //image=Preprocessing::preprocessDigit_1(image);
    //imshow("image", image);


    i=0;
    for(int k=0; k<size_orienta;k++)
    {
        final_descriptor[i++]=descriptorOrienta[k];
    }
    for(int k=0; k<size_conca8G;k++)
    {
        final_descriptor[i++]=descriptorConca8G[k];
    }
    for(int k=0; k<size_concaG;k++)
    {
        final_descriptor[i++]=descriptorConcaG[k];
    }
    for(int k=0; k<size_hprofile;k++)
    {
        final_descriptor[i++]=descriptorHProfile[k];
    }


    delete[] descriptorConca8G;
    delete[] descriptorConcaG;
    delete[] descriptorOrienta;
    delete[] descriptorHProfile;


    return final_descriptor;
}

/*------------------------------------------------------------------------------*/
float *Descriptor::getBuenDescriptor8(cv::Mat image, int *size_des)
{
    float *final_descriptor;
    float *descriptorConca;
    float *descriptorOrienta;

    int N_BINS_HOG=32;
    int size_conca=256;
    int size_orienta=N_BINS_HOG*4;//126
    *size_des=size_conca+size_orienta; //144
    int i=0;
    final_descriptor=new float[size_conca*size_orienta];

    if(image.channels()==3)
    {
        cvtColor(image,image,CV_BGR2GRAY);
    }
    image.convertTo(image, CV_8UC1);
    /*---------------------------------------------------------------------------*/
    image=Preprocessing::preprocessDigit_1(image);
    //cv::imshow("antes de", image*255);
    //cv::waitKey(0);
    /*---------------------------------------------------------------------------*/
    descriptorOrienta=Descriptor::getHOGDescriptorLocal(image,N_BINS_HOG);
    /*---------------------------------------------------------------------------*/
    //std::cout<<"procesando"<<std::endl;
    descriptorConca=Descriptor::getConcavidad8(image);
    //size_conca=16;

    //size_orienta=32*4;
    i=0;
        for(int k=0; k<size_orienta;k++)
    {
        final_descriptor[i++]=descriptorOrienta[k];
    }

    for(int k=0; k<size_conca;k++)
    {
        final_descriptor[i++]=descriptorConca[k];
    }


    delete[] descriptorConca;
    delete[] descriptorOrienta;
    return final_descriptor;
}
/*--------------------------------------------------------------------------------*/
float* Descriptor::getConcavidadLocal(cv::Mat image)
{
    //size = 16*4
    int width=image.size().width;
    int height=image.size().height;

    int width2=width/2;
    int height2=height/2;

    cv::Mat image1=image(cv::Range(0,height2), cv::Range(0,width2));
    cv::Mat image2=image(cv::Range(0,height2), cv::Range(width2,width));
    cv::Mat image3=image(cv::Range(height2, height), cv::Range(0,width2));
    cv::Mat image4=image(cv::Range(height2, height), cv::Range(width2,width));

    float *descriptor1=getConcavidad(image1);
    float *descriptor2=getConcavidad(image2);
    float *descriptor3=getConcavidad(image3);
    float *descriptor4=getConcavidad(image4);

    int size_des=16;
    int num_images=4;

    float *descriptor=new float[num_images*size_des];

    for(int i=0;i<size_des;i++)
    {
        descriptor[i]=descriptor1[i];
        descriptor[size_des+i]=descriptor2[i];
        descriptor[size_des*2+i]=descriptor3[i];
        descriptor[size_des*3+i]=descriptor4[i];
    }

    delete[] descriptor1;
    delete[] descriptor2;
    delete[] descriptor3;
    delete[] descriptor4;
    return descriptor;
}
/*-------------------------------------------------------------------------*/
float* Descriptor::getConcavidad8Local(cv::Mat image)
{
    //size = 16*4
    int width=image.size().width;
    int height=image.size().height;

    int width2=width/2;
    int height2=height/2;

    cv::Mat image1=image(cv::Range(0,height2), cv::Range(0,width2));
    cv::Mat image2=image(cv::Range(0,height2), cv::Range(width2,width));
    cv::Mat image3=image(cv::Range(height2, height), cv::Range(0,width2));
    cv::Mat image4=image(cv::Range(height2, height), cv::Range(width2,width));

    float *descriptor1=getConcavidad8(image1);
    float *descriptor2=getConcavidad8(image2);
    float *descriptor3=getConcavidad8(image3);
    float *descriptor4=getConcavidad8(image4);

    int size_des=16;
    int num_images=4;

    float *descriptor=new float[num_images*size_des];

    for(int i=0;i<size_des;i++)
    {
        descriptor[i]=descriptor1[i];
        descriptor[size_des+i]=descriptor2[i];
        descriptor[size_des*2+i]=descriptor3[i];
        descriptor[size_des*3+i]=descriptor4[i];
    }

    delete[] descriptor1;
    delete[] descriptor2;
    delete[] descriptor3;
    delete[] descriptor4;
    return descriptor;
}
/*-------------------------------------------------------------------------------------------------*/
float* Descriptor::getHorizontalProfile(cv::Mat image)
{
    //asume una buen pre-procesado
    cv::Mat bin;
    CComponent ccomp;
    int DIM=40;
    bin=image.clone();
    float *descriptor=new float[2*DIM];
    /*-----------------------------------------------------------*/
    //Pre-processing
    //int umbral=Preprocessing::umbralOtsu(image);
    //threshold(image,bin, umbral,1,CV_THRESH_BINARY);
    //bin=1-bin;
    /*-----------------------------------------------------------*/
    cv::resize(bin, bin, cv::Size(DIM,DIM),0,0,CV_INTER_CUBIC);
    threshold(bin,bin, 0.5,1,CV_THRESH_BINARY);    
    /*-----------------------------------------------------------*/
    //bin=CComponent::fillHoles(bin);
    /*-----------------------------------------------------------*/

    cv::Mat stel=getStructuringElement(cv::MORPH_CROSS,cv::Size(3,3));
    cv::morphologyEx(bin,bin,cv::MORPH_CLOSE,stel);

    /*-----------------------------------------------------------*/
    bin=Morphological::thinning_Zhang_Sue(bin);
    ccomp.setImage(bin);    
    /*-----------------------------------------------------------*/    
    cv::Rect rect=ccomp.getMinMaxAllBoundingBoxes();
    bin=bin(rect);
    /*-----------------------------------------------------------*/
    cv::resize(bin, bin, cv::Size(DIM,DIM),0,0,CV_INTER_CUBIC);
    threshold(bin,bin, 0.5,1,CV_THRESH_BINARY);
    /*-----------------------------------------------------------*/
    int k=0, i=0;
    float sum1=0;
    for(i=0;i<DIM;i++)
    {
        k=0;
        while(k<DIM && bin.at<uchar>(i,k)==0) k++;
        descriptor[i]=k+1;
        sum1=sum1+descriptor[i];
    }
    //cv::imshow("nofliped", bin*255);
    cv::flip(bin,bin,1);
    //cv::imshow("fliped", bin*255);
    //waitKey();
    float sum2=0;
    for(i=0;i<DIM;i++)
    {
        k=0;
        while(k<DIM && bin.at<uchar>(i,k)==0) k++;

        descriptor[DIM+i]=k+1;
        sum2=sum2+descriptor[DIM+i];
    }
    /*-----------------------------------------------------------*/

    //cv::imshow("xxx", bin*255);

    for(int k=0;k<DIM;k++) descriptor[k]=descriptor[k]/(sum1+EPS);//por ambos lados
    for(int k=0;k<DIM;k++) descriptor[DIM+k]=descriptor[DIM+k]/(sum2+EPS);

    return descriptor;
}
/*-------------------------------------------------------------------------------------------------*/
float* Descriptor::getVerticalProfile(cv::Mat image)
{
    //asume una buen pre-procesado, usar Preprocessing::preprocessDigit_1() previamente
    cv::Mat bin;
    CComponent ccomp;
    int DIM=40;
    bin=image.clone();
    float *descriptor=new float[2*DIM];
    /*-----------------------------------------------------------*/
    //Pre-processing
    //int umbral=Preprocessing::umbralOtsu(image);
    //threshold(image,bin, umbral,1,CV_THRESH_BINARY);
    //bin=1-bin;
    /*-----------------------------------------------------------*/
    cv::resize(bin, bin, cv::Size(DIM,DIM),0,0,CV_INTER_CUBIC);
    threshold(bin,bin, 0.5,1,CV_THRESH_BINARY);
    /*-----------------------------------------------------------*/
    //bin=CComponent::fillHoles(bin);
    /*-----------------------------------------------------------*/
    cv::Mat stel=getStructuringElement(cv::MORPH_CROSS,cv::Size(3,3));
    cv::morphologyEx(bin,bin,cv::MORPH_CLOSE,stel);
    /*-----------------------------------------------------------*/
    bin=Morphological::thinning_Zhang_Sue(bin);
    ccomp.setImage(bin);
    /*-----------------------------------------------------------*/
    cv::Rect rect=ccomp.getMinMaxAllBoundingBoxes();
    bin=bin(rect);
    /*-----------------------------------------------------------*/
    cv::resize(bin, bin, cv::Size(DIM,DIM),0,0,CV_INTER_CUBIC);
    threshold(bin,bin, 0.5,1,CV_THRESH_BINARY);
    /*-----------------------------------------------------------*/
    int k=0, j=0;
    float sum1=0;//Proceso
    for(j=0;j<DIM;j++)//por cada columna
    {
        k=0;
        while(k<DIM && bin.at<uchar>(k,j)==0) k++;
        descriptor[j]=k+1;
        sum1=sum1+descriptor[j];
    }
    //cv::imshow("nofliped", bin*255);
    cv::flip(bin,bin,0);//invertimos la images alrededor del eje x
    //cv::imshow("fliped", bin*255);
    //waitKey();
    float sum2=0;
    for(j=0;j<DIM;j++)
    {
        k=0;
        while(k<DIM && bin.at<uchar>(k,j)==0) k++;

        descriptor[DIM+j]=k+1;
        sum2=sum2+descriptor[DIM+j];
    }
    /*-----------------------------------------------------------*/

    //cv::imshow("xxx", bin*255);

    for(int k=0;k<DIM;k++) descriptor[k]=descriptor[k]/(sum1+sum2);//por ambos lados
    for(int k=0;k<DIM;k++) descriptor[DIM+k]=descriptor[DIM+k]/(sum1+sum2);

    return descriptor;
}
/*------------------------------------------------------------------------*/
float *Descriptor::getConcavityDescriptor(cv::Mat image, int *size_des)
{
    //asumimos que la imagen es CV_U8C1
    *size_des=13;
    int n=*size_des;
    int i=0, j=0;
    int dir_0;
    int dir_1;
    int dir_2;
    int dir_3;
    int dir_s1;
    int dir_s2;
    int dir_s3;
    int dir_s4;
    int n_hits=0;
    int n_hits_s=0;
    int aux_i=0;
    int aux_j=0;
    float *des=new float[n];
    /*--------------------------------------------*/
    for(int i=0; i<n; i++) des[i]=0;
    /*--------------------------------------------*/
    for(i=0; i<image.rows; i++)
    {
        for(j=0; j<image.cols; j++)
        {
            if(image.at<uchar>(i,j)==0)
            {
                //loking for 1s in direction 0
                aux_i=i;
                while(aux_i>=0 && image.at<uchar>(aux_i,j)==0) aux_i--;
                if(aux_i<0) dir_0=0;
                else dir_0=1;

                //loking for 1s in direction 1
                aux_j=j;
                while(aux_j<image.cols && image.at<uchar>(i,aux_j)==0) aux_j++;
                if(aux_j>=image.cols) dir_1=0;
                else dir_1=1;
                //loking for 1s in direction 2
                aux_i=i;
                while(aux_i<image.rows && image.at<uchar>(aux_i,j)==0) aux_i++;
                if(aux_i>=image.rows) dir_2=0;
                else dir_2=1;
                //loking for 1s in direction 3
                aux_j=j;
                while(aux_j>=0 && image.at<uchar>(i,aux_j)==0) aux_j--;
                if(aux_j<0) dir_3=0;
                else dir_3=1;
                /*-----------------------------------------------------*/
                n_hits=dir_0+dir_1+dir_2+dir_3;
                if(n_hits>1)//no-hit or one-hit points are discarded
                {
                    if(n_hits==3)//contabiliza por donde no ha hit
                    {
                        if(dir_0==0) des[4]=des[4]+1;
                        if(dir_1==0) des[5]=des[5]+1;
                        if(dir_2==0) des[6]=des[6]+1;
                        if(dir_3==0) des[7]=des[7]+1;
                    }
                    if(n_hits==2)
                    {
                        if(dir_0==0 && dir_1==0) des[0]=des[0]+1;
                        if(dir_1==0 && dir_2==0) des[1]=des[1]+1;
                        if(dir_2==0 && dir_3==0) des[2]=des[2]+1;
                        if(dir_3==0 && dir_0==0) des[3]=des[3]+1;
                    }
                    if(n_hits==4)
                    {
                        //looking  in direction s1 some escape
                        dir_s1=0;
                        aux_i=i-1;
                        while(aux_i>=0 && dir_s1==0)
                        {
                            aux_j=j;
                            while(aux_j>=0 && image.at<uchar>(aux_i, aux_j)==0) aux_j--;
                            if(aux_j<0) dir_s1=1;
                            aux_i--;
                        }

                        //looking  in direction s2 some escape
                        dir_s2=0;
                        aux_i=i-1;
                        while(aux_i>=0 && dir_s2==0)
                        {
                            aux_j=j;
                            while(aux_j<image.cols && image.at<uchar>(aux_i, aux_j)==0) aux_j++;
                            if(aux_j>=image.cols) dir_s2=1;
                            aux_i--;
                        }
                        //looking  in direction s3 some escape
                        dir_s3=0;
                        aux_i=i+1;
                        while(aux_i<image.rows && dir_s3==0)
                        {
                            aux_j=j;
                            while(aux_j>=0 && image.at<uchar>(aux_i, aux_j)==0) aux_j--;
                            if(aux_j<0) dir_s3=1;
                            aux_i++;
                        }

                        //looking  in direction s2 some escape
                        dir_s4=0;
                        aux_i=i+1;
                        while(aux_i<image.rows && dir_s4==0)
                        {
                            aux_j=j;
                            while(aux_j<image.cols && image.at<uchar>(aux_i, aux_j)==0) aux_j++;
                            if(aux_j>=image.cols) dir_s4=1;
                            aux_i++;
                        }
                        n_hits_s=dir_s1+dir_s2+dir_s3+dir_s4;
                        if(n_hits_s==0) des[8]=des[8]+1;//no escape
                        else
                        {
                            if(dir_s1==1) des[9]=des[9]+1;
                            if(dir_s2==1) des[10]=des[10]+1;
                            if(dir_s3==1) des[11]=des[11]+1;
                            if(dir_s4==1) des[12]=des[12]+1;
                        }

                    }
                }
            }
        }
    }
    //normalizar vector
//    float suma=0;
//    float norma=0;
//    for(int i=0; i<n;i++)
//    {
//        suma=suma+des[i];
//        norma=norma+des[i]*des[i];
//    }
//    norma=sqrt(norma);
//    for(int i=0;i<n;i++) //se puede normalizar por la suma o por la norma
//    {
//        des[i]=des[i]/(norma+EPS);
//    }
    return des;
}
/*----------------------------------------------------------------------------------*/
float *Descriptor::getLocalConcavityDescriptor(cv::Mat image, int *size_des)
{
    *size_des=6*13;
    int n=*size_des;
    float *des=new float[n];
    int h=image.rows;
    int w=image.cols;

    int h_local=h/3;
    int w_local=w/2;
    cv::Mat image_00;
    cv::Mat image_01;
    cv::Mat image_10;
    cv::Mat image_11;
    cv::Mat image_20;
    cv::Mat image_21;

    image_00=image(cv::Range(0, h_local), cv::Range(0, w_local));
    image_01=image(cv::Range(0, h_local), cv::Range(w_local, std::max(2*w_local, image.cols)));

    image_10=image(cv::Range(h_local, 2*h_local), cv::Range(0, w_local));
    image_11=image(cv::Range(h_local, 2*h_local), cv::Range(w_local, std::max(2*w_local, image.cols)));

    image_20=image(cv::Range(2*h_local, std::max(3*h_local, image.rows)), cv::Range(0, w_local));
    image_21=image(cv::Range(2*h_local, std::max(3*h_local, image.rows)), cv::Range(w_local, std::max(2*w_local, image.cols)));

    int size=13;
    float *des_00=getConcavityDescriptor(image_00, &size);
    float *des_01=getConcavityDescriptor(image_01, &size);
    float *des_10=getConcavityDescriptor(image_10, &size);
    float *des_11=getConcavityDescriptor(image_11, &size);
    float *des_20=getConcavityDescriptor(image_20, &size);
    float *des_21=getConcavityDescriptor(image_21, &size);

    for(int i=0; i<size;i++)
    {
        des[i+0]=des_00[i];
        des[i+size]=des_01[i];
        des[i+2*size]=des_10[i];
        des[i+3*size]=des_11[i];
        des[i+4*size]=des_20[i];
        des[i+5*size]=des_21[i];
    }

    delete[] des_00;
    delete[] des_01;
    delete[] des_10;
    delete[] des_11;
    delete[] des_20;
    delete[] des_21;

    float suma=0;
    float norma=0;
    for(int i=0; i<n;i++)
    {
        suma=suma+des[i];
        norma=norma+des[i]*des[i];
    }
    norma=sqrt(norma);
    for(int i=0;i<n;i++) //se puede normalizar por la suma o por la norma
    {
        des[i]=des[i]/(norma+EPS);
    }
    return des;
}
/*----------------------------------------------------------------------------------*/
float *Descriptor::getBuenDescriptor3(cv::Mat _image, int *size_des)
{
    float *final_descriptor;

    float *descriptorC;
    float *descriptorOrienta;

    int size_conca=13*6;
    int size_orienta=32*4;
    int i=0;
    cv::Mat image;
    _image.copyTo(image);
    *size_des=size_conca+size_orienta;

    final_descriptor=new float[*size_des];
    if(image.channels()==3)
    {
        cvtColor(image,image,CV_BGR2GRAY);
    }
    image.convertTo(image, CV_8UC1);
    /*---------------------------------------------------------------------------*/
    image=Preprocessing::preprocessDigit_1(image);

    descriptorC=Descriptor::getLocalConcavityDescriptor(image, &size_conca);

    descriptorOrienta=Descriptor::getHOGDescriptorLocal(image,32);//number of bins for each regio

    i=0;
    for(int k=0; k<size_conca;k++)
    {
        final_descriptor[i++]=descriptorC[k];
    }
    for(int k=0; k<size_orienta;k++)
    {
        final_descriptor[i++]=descriptorOrienta[k];
    }

    delete[] descriptorC;
    delete[] descriptorOrienta;

    return final_descriptor;
}

/*---------------------------------------------------------------------------------*/
bool Descriptor::isValid(int x, int y, int min_x,int max_x, int min_y, int max_y)
{
    bool r=false;
    if(x>=min_x && x<=max_x && y>=min_y && y<=max_y) r=true;
    else r=false;
    return r;
}
/*---------------------------------------------------------------------------------*/
//This implementation use a simple spatial division of the
//This only divide the image into 2x2 regions, then compute a descriptor for each region
// and concatenate each local region into a large final descriptor
float *Descriptor::getHOGDalalTriggs_SpatialPyramid(cv::Mat image, HOG_SPParams params)
{
	int n_rows2=static_cast<int>(floor(image.rows*0.5));
    int n_cols2=static_cast<int>(floor(image.cols*0.5));

    cv::Mat image00=image(cv::Range(0,n_rows2), cv::Range(0,n_cols2));
    cv::Mat image01=image(cv::Range(0,n_rows2), cv::Range(n_cols2, 2*n_cols2));
    cv::Mat image10=image(cv::Range(n_rows2, 2*n_rows2), cv::Range(0,n_cols2));
    cv::Mat image11=image(cv::Range(n_rows2, 2*n_rows2), cv::Range(n_cols2, 2*n_cols2));

    HOGParams params00(params);
    params00.image_size=image00.size();

    HOGParams params01(params);
    params01.image_size=image01.size();

    HOGParams params10(params);
    params10.image_size=image10.size();

    HOGParams params11(params);
    params11.image_size=image11.size();

    int des_size=params00.getDescriptorSize()+params01.getDescriptorSize();
    des_size+=params10.getDescriptorSize()+params11.getDescriptorSize();

    if(des_size!=4*params00.getDescriptorSize())
    {
        std::cout<<"ERROR: HOGDalalTriggs_SP incompatible region sizes"<<std::endl;
        std::cout<<"des_size= "<<des_size<<" region_size="<<params00.getDescriptorSize()<<std::endl;
        exit(EXIT_FAILURE);
    }

    float *descriptor=new float[des_size];
    for(int i=0; i<des_size; i++) descriptor[i]=0;

    float *descriptor1=getHOGDalalTriggs(image00, params00);
    float *descriptor2=getHOGDalalTriggs(image01, params01);
    float *descriptor3=getHOGDalalTriggs(image10, params01);
    float *descriptor4=getHOGDalalTriggs(image11, params01);

    int k=0;
    for(int i=0; i<params00.getDescriptorSize();i++)    descriptor[k++]=descriptor1[i];
    for(int i=0; i<params01.getDescriptorSize();i++)    descriptor[k++]=descriptor2[i];
    for(int i=0; i<params10.getDescriptorSize();i++)    descriptor[k++]=descriptor2[i];
    for(int i=0; i<params11.getDescriptorSize();i++)    descriptor[k++]=descriptor3[i];

    delete[] descriptor1;
    delete[] descriptor2;
    delete[] descriptor3;
    delete[] descriptor4;

    return descriptor;
}
/*---------------------------------------------------------------------------------*/
//This implementation is according to the Dalal-Triggs method
float *Descriptor::getHOGDalalTriggs(cv::Mat image, HOGParams params)
{
    /*-----------------------------------------------------------------------------*/
    //std::cout<<"--->"<<params.th_intersection<<std::endl;
    //Compute gradients for each cell
    cv::Mat Gx=(cv::Mat_<float>(1,3)<<-1,0,1);
    cv::Mat Gy=(cv::Mat_<float>(3,1)<<-1,0,1);
    cv::Mat im_x=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat im_y=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat image_f;
    cv::Mat angles=cv::Mat(image.size(), CV_32F);
    cv::Mat magnitudes=cv::Mat(image.size(), CV_32F);
    image.convertTo(image_f, CV_32F);
    filter2D(image_f, im_x, CV_32F, Gx);
    filter2D(image_f, im_y, CV_32F, Gy);
    int i=0, j=0;
    float mag_val=0;
    //std::cout<<"1. Calculating gradients."<<std::endl;
    for(i=0; i<image_f.rows; i++)
    {
        for(j=0; j<image_f.cols; j++)
        {   //angles varies between 0->pi
            angles.at<float>(i,j)=atan2(im_y.at<float>(i,j), im_x.at<float>(i,j));            
            if(angles.at<float>(i,j)<0) angles.at<float>(i,j)=angles.at<float>(i,j)+PI;
            mag_val=sqrt(im_x.at<float>(i,j)*im_x.at<float>(i,j)+im_y.at<float>(i,j)*im_y.at<float>(i,j));
            magnitudes.at<float>(i,j)=mag_val;            
        }
    }
    /*-----------------------------------------------------------------------------*/    
    //Buillding histogram of orientation for each cell
    //std::cout<<"2. Building histograms of orientation in cells."<<std::endl;
    int n_cells_row=static_cast<int>(round(image.rows/static_cast<float>(params.cell_size)));
    int n_cells_col=static_cast<int>(round(image.cols/static_cast<float>(params.cell_size)));

    float *hog_cell=NULL;
    float val_idx=0, val_ang=0;
    int k=0;
    std::vector<float*> list_of_hogs(n_cells_row*n_cells_col);
    for(k=0; k<n_cells_row*n_cells_col; k++)
    {
        hog_cell=new float[params.n_channels];
        for(int i=0; i<params.n_channels;i++) hog_cell[i]=0;
        list_of_hogs[k]=hog_cell;
    }
    //-----------------------------------------------------------------------------------------
    //Trilinear interpolation using position and orientation-----------------------------------
    int x_izq=0, x_der=0, y_izq=0, y_der=0, bin_izq=0, bin_der=0;
    float w_x_izq=0, w_x_der=0, w_y_izq=0, w_y_der=0, w_bin_izq=0, w_bin_der=0;
    float pos_x=0, pos_y=0;
    //float dist_midx=0, dist_midy=0, dist_mid_bin=0;
    float *hog;//this will point a certain hog of the list of hogs
    for(i=0; i<image.rows; i++)
    {
        for(j=0;j<image.cols;j++)
        {
            //------interpolating x position-------------------
            pos_x=(j/static_cast<float>(image.cols))*n_cells_col;
            pos_y=(i/static_cast<float>(image.rows))*n_cells_row;
            Preprocessing::linearInterBIN(pos_x, &x_izq, &x_der, &w_x_izq, &w_x_der);
            Preprocessing::linearInterBIN(pos_y, &y_izq, &y_der, &w_y_izq, &w_y_der);
            //------interpolating angles------------------------
            val_ang=angles.at<float>(i,j);            
            val_idx=(val_ang/PI)*params.n_channels;
            Preprocessing::linearInterBIN(val_idx, &bin_izq, &bin_der, &w_bin_izq, &w_bin_der);
            //ángulos son circulares----------------------------
            if(bin_izq<0) bin_izq=params.n_channels-1;
            if(bin_der>=params.n_channels) bin_der=0;
            //--------------------------------------------------
            mag_val=magnitudes.at<float>(i,j);
            if(isValid(x_izq, y_izq, 0, n_cells_col-1, 0, n_cells_row-1))
            {
                hog=list_of_hogs[y_izq*n_cells_col+x_izq];
                hog[bin_izq]=hog[bin_izq]+mag_val*w_x_izq*w_y_izq*w_bin_izq;
                hog[bin_der]=hog[bin_der]+mag_val*w_x_izq*w_y_izq*w_bin_der;                
            }

            if(isValid(x_izq, y_der, 0, n_cells_col-1, 0, n_cells_row-1))
            {
                hog=list_of_hogs[y_der*n_cells_col+x_izq];
                hog[bin_izq]=hog[bin_izq]+mag_val*w_x_izq*w_y_der*w_bin_izq;
                hog[bin_der]=hog[bin_der]+mag_val*w_x_izq*w_y_der*w_bin_der;
            }

            if(isValid(x_der, y_izq, 0, n_cells_col-1, 0, n_cells_row-1))
            {
                hog=list_of_hogs[y_izq*n_cells_col+x_der];
                hog[bin_izq]=hog[bin_izq]+mag_val*w_x_der*w_y_izq*w_bin_izq;
                hog[bin_der]=hog[bin_der]+mag_val*w_x_der*w_y_izq*w_bin_der;
            }

            if(isValid(x_der, y_der, 0, n_cells_col-1, 0, n_cells_row-1))
            {
                hog=list_of_hogs[y_der*n_cells_col+x_der];
                hog[bin_izq]=hog[bin_izq]+mag_val*w_x_der*w_y_der*w_bin_izq;
                hog[bin_der]=hog[bin_der]+mag_val*w_x_der*w_y_der*w_bin_der;
            }
        }
    }
    //-----------------------------------------------------------------------------------------
    //Normalizing by blocks
    //std::cout<<"3. Normalizing by blocks."<<std::endl;
    int step=params.block_size-ceil(params.block_size*params.th_intersection);
    int i_cell=0, j_cell=0;
    int curr_cell_i, curr_cell_j;    
    int i_block=0;
    int n_blocks=0;
    float norm_value=0;
    float norm_e=0.0001;
    //we process only blocks filled with cells
    int number_of_blocks_row=ceil((1+n_cells_row-params.block_size)/static_cast<float>(step));
    int number_of_blocks_col=ceil((1+n_cells_col-params.block_size)/static_cast<float>(step));
    int number_of_blocks=number_of_blocks_row*number_of_blocks_col;
    int block_hog_size=params.n_channels*(params.block_size*params.block_size);

    float *final_hog=new float[number_of_blocks*block_hog_size];
    int block_offset=0;
    for(k=0;k<number_of_blocks*block_hog_size;k++) final_hog[k]=0;

    n_blocks=0;
    //Starting cells for each block
    for(i_cell=0; i_cell<=n_cells_row-params.block_size; i_cell=i_cell+step)
    {
        for(j_cell=0; j_cell<=n_cells_col-params.block_size; j_cell=j_cell+step)
        {
            /*------------------------------------------------------------*/
            norm_value=0;
            i_block=0;
            block_offset=n_blocks*block_hog_size;
            for(i=0; i<params.block_size; i++)
            {
                for(j=0; j<params.block_size; j++)
                {
                    curr_cell_i=i+i_cell;
                    curr_cell_j=j+j_cell;
                    hog=list_of_hogs[curr_cell_i*n_cells_col+curr_cell_j];
                    for(k=0;k<params.n_channels;k++)
                    {
                        final_hog[block_offset+i_block]=hog[k];
                        norm_value=norm_value+hog[k]*hog[k];
                        //std::cout<<hog[k]<<" "<<norm_value<<std::endl;
                        i_block++;
                    }
                }                
            }
            /*------------------------------------------------------------*/
            //Normalizing and saturating the historgram
            norm_value=sqrt(norm_value+norm_e*norm_e);
            for(i=0; i<i_block;i++)
            {                
                final_hog[block_offset+i]=final_hog[block_offset+i]/(norm_value);
                if(std::isnan(final_hog[block_offset+i]))
                {
                    std::cout<<"ERROR at HOGDalal: A NAN occurs!!!!"<<std::endl;
                    exit(EXIT_FAILURE);
                }
                if(final_hog[block_offset+i]>0.2) final_hog[block_offset+i]=0.2;
                if(final_hog[block_offset+i]<1e-20) final_hog[block_offset+i]=0;
            }
            //std::cout<<std::endl;
            n_blocks++;
        }
    }
    //cleaning memory
    for(i=0;i<(int)list_of_hogs.size();i++)
    {
        delete[] list_of_hogs[i];
    }
    if(n_blocks!=number_of_blocks)
    {
        std::cout<<"ERROR: The number of blocks was misscalculated!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    return final_hog;

}
/*-----------------------------------------------------------------------------------------*/
void  Descriptor::drawHOGDalalTriggs(cv::Mat image, float *hog, float max_ang, HOGParams params)
{
    cv::Mat HOGImage=cv::Mat::zeros(image.size(), CV_8UC1);
    int step=params.block_size-ceil(params.block_size*params.th_intersection);

    int n_cells_row=static_cast<int>(round(image.rows/static_cast<float>(params.cell_size)));
    int n_cells_col=static_cast<int>(round(image.cols/static_cast<float>(params.cell_size)));
    int block_hog_size=params.n_channels*(params.block_size*params.block_size);
    int i_cell=0, j_cell=0;
    int block_offset=0;
    int n_blocks=0;
    int k=0;
    float ang_step=max_ang/params.n_channels;
    float length=params.cell_size;
    int xc=0, yc=0;
    int xf=0, yf=0, xr=0, yr=0;
    float theta=0;
    float val=0;
    //std::cout<<n_cells_row<<" "<<n_cells_col<<std::endl;
    //std::cout<<block_hog_size<<std::endl;
    for(i_cell=0; i_cell<=n_cells_row-params.block_size; i_cell=i_cell+step)
    {
        for(j_cell=0; j_cell<=n_cells_col-params.block_size; j_cell=j_cell+step)
        {
            block_offset=n_blocks*block_hog_size;
            xc=static_cast<int>((j_cell+ params.block_size*0.5)*params.cell_size);
            yc=static_cast<int>((i_cell+ params.block_size*0.5)*params.cell_size);

            xf=length;
            yf=0;

            for(k=0;k<params.n_channels;k++)
            {
                theta=ang_step*k;

                xr=cos(-theta)*xf-sin(-theta)*yf+xc;
                yr=sin(-theta)*xf+cos(-theta)*yf+yc;

                val=0;
                for(int c=0; c<params.block_size*params.block_size; c++)
                {
                    val+=hog[block_offset+c*params.n_channels+k];
                }

                //std::cout<<step<<"Lineas de "<<xc<<","<<yc<<" a"<<xf<<","<<yf<<std::endl<<val*200<<std::endl;

                if(xr<0) xr=0;
                if(yr<0) yr=0;
                if(xr>=image.cols) xr=image.cols-1;
                if(yr>=image.rows) yr=image.rows-1;
                line(HOGImage,cv::Point(xc,yc),cv::Point(xr,yr), (val+0.1)*200);
            }
            n_blocks++;
            //imshow("Parcial", HOGImage);
           // waitKey();
            line(HOGImage,cv::Point(xc,yc),cv::Point(xc,yc), 255);
            /*------------------------------------------------------------*/
        }
    }
    cv::imshow("HOG", HOGImage);    
}
#ifndef NO_VLFEAT
	/*-----------------------------------------------------------------------------------------*/
	int Descriptor::estimate_HOG_VL_size(cv::Size image_size, HOGParams params)
	{
		float *data=new float[image_size.width*image_size.height];
		VlHog *hog = vl_hog_new(VlHogVariantDalalTriggs, params.n_channels, VL_FALSE);
		vl_hog_put_image(hog, data , image_size.height, image_size.width, 1, params.cell_size) ;
		int hogWidth = vl_hog_get_width(hog) ;
		int hogHeight = vl_hog_get_height(hog) ;
		int hogDimension = vl_hog_get_dimension(hog) ;
		int size=hogWidth*hogHeight*hogDimension;
		vl_hog_delete(hog) ;
		delete[] data;
		return size;
	}
	/*-----------------------------------------------------------------------------------------*/
	int Descriptor::estimate_HOG_FZ_size(cv::Size image_size, HOGParams params)
	{
		float *data=NULL;
		VlHog *hog = vl_hog_new(VlHogVariantUoctti, params.n_channels, VL_FALSE);
		vl_hog_put_image(hog, data , image_size.height, image_size.width, 1, params.cell_size) ;
		int hogWidth = vl_hog_get_width(hog) ;
		int hogHeight = vl_hog_get_height(hog) ;
		int hogDimension = vl_hog_get_dimension(hog) ;
		int size=hogWidth*hogHeight*hogDimension;
		vl_hog_delete(hog) ;
		return size;
	}
	/*-----------------------------------------------------------------------------------------*/
	float *Descriptor::getHOGDalalTriggsVL(cv::Mat image, HOGParams params)
	{
		cv::Mat imageg;

		if(image.channels()==3) cvtColor(image, imageg,CV_BGR2BGRA);
		else image.copyTo(imageg);
		imageg.convertTo(imageg,CV_32F);
		float *data=(float*)imageg.data;
		VlHog *hog = vl_hog_new(VlHogVariantDalalTriggs, params.n_channels, VL_FALSE);
		vl_hog_put_image(hog, data , image.rows, image.cols, 1, params.cell_size) ;
		int hogWidth = vl_hog_get_width(hog) ;
		int hogHeight = vl_hog_get_height(hog) ;
		int hogDimension = vl_hog_get_dimension(hog) ;
		int size=hogWidth*hogHeight*hogDimension;
		void *hogArray = vl_malloc(size*sizeof(float)) ;
		vl_hog_extract(hog, (float*)hogArray) ;
		vl_hog_delete(hog) ;

		return (float*)hogArray;
	}
	/*-----------------------------------------------------------------------------------------*/
	float *Descriptor::getHOGDalalTriggsFZ(cv::Mat image, HOGParams params)
	{
		cv::Mat imageg;
		if(image.channels()==3) cvtColor(image, imageg,CV_BGR2BGRA);
		else image.copyTo(imageg);
		imageg.convertTo(imageg,CV_32F);
		float *data=(float*)imageg.data;
		VlHog *hog = vl_hog_new(VlHogVariantUoctti, params.n_channels, VL_FALSE);
		vl_hog_put_image(hog, data , image.rows, image.cols, 1, params.cell_size) ;
		int hogWidth = vl_hog_get_width(hog) ;
		int hogHeight = vl_hog_get_height(hog) ;
		int hogDimension = vl_hog_get_dimension(hog);
		int size=hogWidth*hogHeight*hogDimension;
		void *hogArray = vl_malloc(size*sizeof(float)) ;
		vl_hog_extract(hog, (float*)hogArray) ;
		vl_hog_delete(hog) ;
		return (float*)hogArray;
	}
	void Descriptor::HOG_VL_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, HOGParams params)
	{
	    cv::Mat imageg;
	    if(image.channels()==3) cvtColor(image,imageg,CV_BGR2GRAY);
	    else image.copyTo(imageg);
	    //--------------------------------------
	    int n_kp=0;
	    n_kp=keypoints.size();
	    int x=0, y=0, size=0;
	    int xf=0, yf=0;
	    int i_kp=0, i=0;
	    cv::Rect rect;
	    cv::Mat roi;
	    float *descriptor=NULL;
	    int n_width=60;
	    int n_height=60;

	    //std::cout<<"calculando tamaño"<<std::endl;
	    int des_size=estimate_HOG_VL_size(cv::Size(n_width,n_height), params);
	    descriptors=cv::Mat(n_kp,des_size, CV_32F);
	    //--------------------------------------
	    //std::cout<<"Calculando descriptores"<<std::endl;
	    for(i_kp=0; i_kp<n_kp;  i_kp++)
	    {
	        x=keypoints[i_kp].pt.x;
	        y=keypoints[i_kp].pt.y;
	        size=30;//6*(keypoints[i_kp].size);
	        x=std::max(x-size,0);
	        y=std::max(y-size,0);
	        xf=std::min(x+2*size-1,image.cols-1);
	        yf=std::min(y+2*size-1,image.rows-1);
	        rect=cv::Rect(x,y,xf-x+1, yf-y+1);
	        roi=imageg(rect);
	        //---------------------------------------------
	        resize(roi,roi,cv::Size(n_width, n_height));
	       // std::cout<<n_kp<<"-"<<i_kp<<"-"<<des_size<<std::endl;
	        descriptor=getHOGDalalTriggsVL(roi, params);
	        for(i=0;i<des_size;i++) descriptors.at<float>(i_kp,i)=descriptor[i];
	        delete[] descriptor;
	    }
	    //--------------------------------------
	}
	/*--------------------------------------------------------------------------------------*/
	void Descriptor::HOG_FZ_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, HOGParams params)
	{
	    cv::Mat imageg;
	    if(image.channels()==3) cvtColor(image,imageg,CV_BGR2GRAY);
	    else image.copyTo(imageg);
	    //--------------------------------------
	    int n_kp=0;
	    n_kp=keypoints.size();
	    int x=0, y=0, size=0;
	    int xf=0, yf=0;
	    int i_kp=0, i=0;
	    cv::Rect rect;
	    cv::Mat roi;
	    float *descriptor=NULL;
	    int n_width=60;
	    int n_height=60;
	    //std::cout<<"calculando tamaño"<<std::endl;
	    int des_size=estimate_HOG_VL_size(cv::Size(n_width,n_height), params);
	    descriptors=cv::Mat(n_kp,des_size, CV_32F);
	    //--------------------------------------
	    //std::cout<<"Calculando descriptores"<<std::endl;
	    for(i_kp=0; i_kp<n_kp;  i_kp++)
	    {
	        x=keypoints[i_kp].pt.x;
	        y=keypoints[i_kp].pt.y;
	        size=30;//6*(keypoints[i_kp].size);
	        x=std::max(x-size,0);
	        y=std::max(y-size,0);
	        xf=std::min(x+2*size-1,image.cols-1);
	        yf=std::min(y+2*size-1,image.rows-1);
	        rect=cv::Rect(x,y,xf-x+1, yf-y+1);
	        roi=imageg(rect);
	        //---------------------------------------------
	        resize(roi,roi,cv::Size(n_width, n_height));
	       // std::cout<<n_kp<<"-"<<i_kp<<"-"<<des_size<<std::endl;
	        descriptor=getHOGDalalTriggsFZ(roi, params);
	        for(i=0;i<des_size;i++) descriptors.at<float>(i_kp,i)=descriptor[i];
	        delete[] descriptor;
	    }
	}
	/*--------------------------------------------------------------------------------------*/
	void Descriptor::LBP_VL_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, LBPParams params)
	{
	    cv::Mat imageg;
	    if(image.channels()==3) cvtColor(image,imageg,CV_BGR2GRAY);
	    else image.copyTo(imageg);
	    //--------------------------------------
	    int n_kp=0;
	    n_kp=keypoints.size();
	    int x=0, y=0, size=0;
	    int xf=0, yf=0;
	    int i_kp=0, i=0;
	    cv::Rect rect;
	    cv::Mat roi;
	    float *descriptor=NULL;
	    int n_width=60;
	    int n_height=60;
	    //std::cout<<"calculando tamaño"<<std::endl;
	    int des_size=params.getDescriptorSize();
	    descriptors=cv::Mat(n_kp,des_size, CV_32F);
	    //--------------------------------------
	    //std::cout<<"Calculando descriptores"<<std::endl;
	    for(i_kp=0; i_kp<n_kp;  i_kp++)
	    {
	        x=keypoints[i_kp].pt.x;
	        y=keypoints[i_kp].pt.y;
	        size=30;//5*(keypoints[i_kp].size);
	        x=std::max(x-size,0);
	        y=std::max(y-size,0);
	        xf=std::min(x+2*size-1,imageg.cols-1);
	        yf=std::min(y+2*size-1,imageg.rows-1);
	        int min_dim=std::min(xf-x+1,yf-y+1); //to ensure square rois
	        rect=cv::Rect(x,y,min_dim, min_dim);
	        roi=imageg(rect);
	        resize(roi,roi,cv::Size(n_width, n_height));
	        //---------------------------------------------
	        //std::cout<<n_kp<<"-"<<i_kp<<"-"<<des_size<<std::endl;
	        descriptor=getLBP_VL_Descriptor(roi, params);
	        for(i=0;i<des_size;i++) descriptors.at<float>(i_kp,i)=descriptor[i];
	        delete[] descriptor;
	    }
	}
	/*--------------------------------------------------------------------------------------*/
	//An implementation of  LBP descriptor from VLFeat
	float *Descriptor::getLBP_VL_Descriptor(cv::Mat image, LBPParams params)
	{
	    //This is an implementatiof from VEDALDI
	    //It seems to be a bug because for some images a nan value is obtained!!!!
	    //this assumes that image is a gray image CV_8UC1
	    //image.cols==image.rows //mut be square and a multiple of num_cells to avoid division errors
	    //.data is uchar*, tener cuidado
	    VlLbp *lbp=vl_lbp_new(VlLbpUniform, VL_FALSE);
	    float *data=new float[image.cols*image.rows];
	    for(int i=0; i<image.cols*image.rows; i++) data[i]=(float)image.data[i];
	    int cell_size=floor(image.cols/params.num_cells);
	    if(floor(image.cols/cell_size)!=params.num_cells)
	    {
	        std::cout<<"ERROR: It possible that your size estimation is wrong!!";
	        exit(EXIT_FAILURE);
	    }
	    int d_size=params.getDescriptorSize();
	    float *descriptor=new float[d_size];
	    vl_lbp_process(lbp, descriptor, data, image.cols, image.rows, cell_size);

	    vl_lbp_delete(lbp);

	    delete[] data;

	    return descriptor;
	}
#endif // end vlfeat based functions
/*-----------------------------------------------------------------------------------------*/
float* smoothVector(float *h, int n, int k)
{
	float* hs=new float[n];
	float s=0;
	int nn=0,i=0,l=0;
	for(i=0;i<n;i++){
		s=0;
		nn=0;
		for(l=i-k;l<i+k;l++){
			if((l>=0)&&(l<n)){
				s+=h[l];
				nn++;
			}
		}
		hs[i]=s/nn;
	}
	return hs;
}
/*-----------------------------------------------------------------------------------------*/
float *Descriptor::getHELODescriptor(cv::Mat image, HELOParams params)
{
    //image is a binary image, where max_value=255;
    std::vector<float> sines(params.n_cells*params.n_cells);
    std::vector<float> cosines(params.n_cells*params.n_cells);
    std::vector<int> cell_count(params.n_cells*params.n_cells);
    std::vector<int> edge_count(params.n_cells*params.n_cells);
    int i=0, j=0;
    int cell_width=0, cell_height=0;
    int th_NEdge=0;
    //Estimating cell size
    cell_width=std::ceil(image.cols/static_cast<float>(params.n_cells));
    cell_height=std::ceil(image.rows/static_cast<float>(params.n_cells));
    th_NEdge=std::max(cell_width, cell_height)/2;
    //Initialization: all vectors to zero
    for(i=0; i<params.n_cells*params.n_cells; i++){
        sines[i]=0;
        cosines[i]=0;
        cell_count[i]=0;
        edge_count[i]=0;
    }
    //Sobel mask
    cv::Mat Gx=(cv::Mat_<float>(3,3)<<-1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gy;
    transpose(Gx,Gy);

    cv::Mat im_x=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat im_y=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat image_f;
    image.convertTo(image_f, CV_32F);
    filter2D(image_f, im_x, CV_32F, Gx);
    filter2D(image_f, im_y, CV_32F, Gy);

    //--------------------------------------------------
    int pos_x=0;
    int pos_y=0;
    float d_angle=0;

    int idx=0;
    for(i=0; i<image.rows; i++)
    {
        for(j=0; j<image.cols; j++)
        {
            pos_x=round((j/static_cast<float>(image.cols))*params.n_cells);
            pos_y=round((i/static_cast<float>(image.rows))*params.n_cells);
            if(isValid(pos_x, pos_y, 0, params.n_cells-1, 0, params.n_cells-1))
            {
                idx=pos_y*params.n_cells+pos_x;
                sines[idx]+=2*im_x.at<float>(i,j)*im_y.at<float>(i,j);
                cosines[idx]+=std::pow(im_x.at<float>(i,j),2.0)-std::pow(im_y.at<float>(i,j),2.0);
                if (image.at<uchar>(i,j)>0) edge_count[idx]++;
                cell_count[idx]++;
            }
        }
    }
    //--------------------------------------------------------------
    //smoothing sines and cosines
    for(int i=0; i<1; i++){
    	cv::GaussianBlur(sines, sines, cv::Size(3,3),0.5);
    	cv::GaussianBlur(cosines, cosines, cv::Size(3,3),0.5);
    }
    //--------------------------------------------------------------
    float *descriptor=new float[params.n_bins];
    for(i=0; i<params.n_bins; i++) descriptor[i]=0;

    int bin=0;
    /*------------------------------------------------------------*/
    for(i=0;i<params.n_cells*params.n_cells;i++)
    {
    	if(edge_count[i]>th_NEdge){
			d_angle=atan2(sines[i], cosines[i]);
			if(d_angle<0) d_angle+=2*PI;
			d_angle=d_angle*0.5;//it must vary between 0 and pi
			/*add pi/2 to get the actual orientation*/
			if(d_angle>=PI*0.5) d_angle-=PI*0.5;
			else d_angle+=PI*0.5;
			/*-------------------------------------*/
			bin=static_cast<int>(std::floor((d_angle/PI)*params.n_bins));
			if(bin==params.n_bins) bin=0;
            descriptor[bin]+=1;
        }
    }
    float* s_descriptor=smoothVector(descriptor, params.n_bins, 6);
    delete[] descriptor;
    //normalizamos el vector
    Preprocessing::normalizeVector(s_descriptor, params.n_bins, NORMALIZE_ROOT);
    /*for(int i=0; i<params.n_bins; i++){
    	std::cout<<descriptor[i]<<" ";
    }
    std::cout<<std::endl;
    */

    return s_descriptor;
}

/*-----------------------------------------------------------------------------------------*/
//This is Soft Histogram of Edge Local Orientations SHELO
float *Descriptor::getLocalHELODescriptor(cv::Mat image, HELOParams params, cv::Mat& im_draw)//S_HELO
{

	//----------------------------------------------------------------------
	bool draw=false;
	if(&im_draw!=&JMSR_EMPTY_MAT){
		draw=true;
	}
	//----------------------------------------------------------------------
	//image is a CV_U8C1 image
    float TH_MAG=0.1;
    int N_BLOCKS=params.n_blocks;
    int normalization_type=params.normalization;

    std::vector<float> sines(params.n_cells*params.n_cells);
    std::vector<float> cosines(params.n_cells*params.n_cells);
    std::vector<float> cell_magnitudes(params.n_cells*params.n_cells);
    std::vector<float> cell_count(params.n_cells*params.n_cells);
    std::vector<float*> histograms(N_BLOCKS*N_BLOCKS);

    int i=0, j=0;
    //----------------------------------------------------------------------
    //Initialization
    for(i=0; i<params.n_cells*params.n_cells; i++)
    {
        sines[i]=0;
        cosines[i]=0;
        cell_magnitudes[i]=0;
        cell_count[i]=0;
    }
    for(i=0; i<N_BLOCKS*N_BLOCKS; i++)
    {
        histograms[i]=new float[params.n_bins];
        for(j=0;j<params.n_bins;j++)
        {
            histograms[i][j]=0;
        }
    }
    //-----------------------------------------------------------------------
    //computing orientation based on Sobel
    cv::Mat Gx=(cv::Mat_<float>(3,3)<<-1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gy;
    transpose(Gx,Gy);

    cv::Mat im_x=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat im_y=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat image_f;
    image.convertTo(image_f, CV_32F);
    filter2D(image_f, im_x, CV_32F, Gx);
    filter2D(image_f, im_y, CV_32F, Gy);
    cv::Mat angles=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat magnitudes=cv::Mat::zeros(image.size(), CV_32F);
    //--------------------------------------------------
    //Computing angles
    for(i=0; i<image.rows; i++)
    {
        for(j=0; j<image.cols; j++)
        {
            angles.at<float>(i,j)=atan2(im_y.at<float>(i,j), im_x.at<float>(i,j));
            if(angles.at<float>(i,j)<0) angles.at<float>(i,j)+=2*PI;//varies between 0 and 2*PI
            magnitudes.at<float>(i,j)=sqrt(im_x.at<float>(i,j)*im_x.at<float>(i,j)+im_y.at<float>(i,j)*im_y.at<float>(i,j));
        }
    }
    //--------------------------------------------------
    float pos_x=0;
    float pos_y=0;
    int x_izq=0, x_der=0, y_der=0, y_izq=0;
    float w_x_izq=0, w_x_der=0, w_y_der=0, w_y_izq=0;
    float angle=0, mag=0;
    float d_angle=0;
    float max_mag=0;
    int idx=0;
    float weight=0;
    for(i=0; i<image.rows; i++)
    {
        for(j=0; j<image.cols; j++)
        {
            angle=angles.at<float>(i,j);
            mag=magnitudes.at<float>(i,j);
            pos_x=(j/static_cast<float>(image.cols))*params.n_cells;
            pos_y=(i/static_cast<float>(image.rows))*params.n_cells;
            Preprocessing::linearInterBIN(pos_x, &x_izq, &x_der, &w_x_izq, &w_x_der);
            Preprocessing::linearInterBIN(pos_y, &y_izq, &y_der, &w_y_izq, &w_y_der);
            //--------------------------------------------------------------------------------
            //Interpolacion bi-lineal en las celdas
            if(isValid(x_izq, y_izq, 0, params.n_cells-1, 0, params.n_cells-1))
            {
                idx=y_izq*params.n_cells+x_izq;
                weight=w_x_izq*w_y_izq;

                sines[idx]+=mag*2*sin(angle)*cos(angle);
                cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                cell_magnitudes[idx]+=mag*weight;
                if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                cell_count[idx]+=weight;
            }
            if(isValid(x_izq, y_der, 0, params.n_cells-1, 0, params.n_cells-1))
            {
                idx=y_der*params.n_cells+x_izq;
                weight=w_x_izq*w_y_der;

                sines[idx]+=mag*2*sin(angle)*cos(angle);
                cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                cell_magnitudes[idx]+=mag*weight;
                if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                cell_count[idx]+=weight;
            }
            if(isValid(x_der, y_izq, 0, params.n_cells-1, 0, params.n_cells-1))
            {
                idx=y_izq*params.n_cells+x_der;
                weight=w_x_der*w_y_izq;

                sines[idx]+=mag*2*sin(angle)*cos(angle);
                cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                cell_magnitudes[idx]+=mag*weight;
                if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                cell_count[idx]+=weight;
            }
            if(isValid(x_der, y_der, 0, params.n_cells-1, 0, params.n_cells-1))
            {
                idx=y_der*params.n_cells+x_der;
                weight=w_x_der*w_y_der;

                sines[idx]+=mag*2*sin(angle)*cos(angle);
                cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                cell_magnitudes[idx]+=mag*weight;
                if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                cell_count[idx]+=weight;
            }
            //--------------------------------------------------------------------------------
        }
    }
    //--------------------------------------------------
    //Promedio de magnitudes
    //for(i=0; i<params.n_cells*params.n_cells; i++) cell_magnitudes[i]/=max_mag;
    //Es posible agregar un factor de confianza, las magnitudes dan la certeza
    //--------------------------------------------------
    if(draw)
    {
        //int stepx=image.cols/params.n_cells;
        //int stepy=image.rows/params.n_cells;
    	//Preprocessing::drawGrid(im_draw, stepx, stepy, cv::Scalar(0,0,255));
        //image.copyTo(im_draw);
        //cvtColor(im_draw, *im_draw, CV_GRAY2BGR);
    	im_draw.create(image.size(), CV_8UC1);
    	im_draw.setTo(0);

    }

    //--------------------------------------------------
    //Creamos el descriptor obteniendo el ángulo por celda
    float *descriptor_aux;
    /*---------------------------------------------------*/
    float bin=0;
    int bin_izq=0, bin_der=0;
    float w_bin_izq=0, w_bin_der=0;
    float y_block=0, x_block=0;
    int cell_size_col=floor(image.cols/params.n_cells);
    int cell_size_row=floor(image.rows/params.n_cells);
    int i_cell=0, j_cell=0;
    float xi=0, yi=0, xc=0, yc=0, xf=0, yf=0, x_=0, y_=0;
    float certeza=0;
    /*---------------------------------------------------*/
    for(i=0;i<params.n_cells*params.n_cells;i++)
    {
        //cell positions-----------------------------------
        i_cell=floor(i/params.n_cells);
        j_cell=i-i_cell*params.n_cells;
        //-------------------------------------------------
        d_angle=atan2(sines[i], cosines[i]);
        if(d_angle<0) d_angle+=2*PI;
        d_angle=d_angle*0.5;//it must vary between 0 and pi
        /*add pi/2 to get the actual orientation*/
        if(d_angle>PI*0.5) d_angle-=PI*0.5;
        else d_angle+=PI*0.5;
        /*-----------------------------------------------*/
        bin=(d_angle/PI)*params.n_bins;
        Preprocessing::linearInterBIN(bin, &bin_izq, &bin_der, &w_bin_izq, &w_bin_der);
        if(bin_izq==params.n_bins) bin_izq=0;
        if(bin_izq<0) bin_izq=params.n_bins-1;
        if(bin_der==params.n_bins) bin_der=0;
        if(bin_der<0) bin_der=params.n_bins-1;
        certeza=cell_magnitudes[i]/max_mag;//se debe buscar magnitudes locales

        if(certeza>TH_MAG)
        {
            x_block=(j_cell/static_cast<float>(params.n_cells))*N_BLOCKS;
            y_block=(i_cell/static_cast<float>(params.n_cells))*N_BLOCKS;

            Preprocessing::linearInterBIN(x_block, &x_izq, &x_der, &w_x_izq, &w_x_der);
            Preprocessing::linearInterBIN(y_block, &y_izq, &y_der, &w_y_izq, &w_y_der);

            if(isValid(x_izq, y_izq, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_izq+y_izq*N_BLOCKS;
                weight=w_x_izq*w_y_izq;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            if(isValid(x_izq, y_der, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_izq+y_der*N_BLOCKS;
                weight=w_x_izq*w_y_der;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            if(isValid(x_der, y_izq, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_der+y_izq*N_BLOCKS;
                weight=w_x_der*w_y_izq;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            if(isValid(x_der, y_der, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_der+y_der*N_BLOCKS;
                weight=w_x_der*w_y_der;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            //dibujamos el vector de orientación local--------------------------------
            if(draw)
            {
                xi=(j_cell+0.2)*cell_size_col;
                xc=(j_cell+0.5)*cell_size_col;
                yc=(i_cell+0.5)*cell_size_row;
                yi=yc;
                xf=(j_cell+0.8)*cell_size_col;
                yf=yi;
                xi=xc-xi;
                yi=yc-yi;

                xf=xc-xf;
                yf=yc-yf;

                x_=cos(d_angle)*xi-sin(d_angle)*yi;
                y_=sin(d_angle)*xi+cos(d_angle)*yi;
                xi=x_+xc;
                yi=y_+yc;

                x_=cos(d_angle)*xf-sin(d_angle)*yf;
                y_=sin(d_angle)*xf+cos(d_angle)*yf;
                xf=x_+xc;
                yf=y_+yc;
                cv::line(im_draw, cv::Point(xi,yi), cv::Point(xf, yf),certeza*255,2);
                //rectangle(*im_draw, cv::Rect(j_cell*cell_size_col, i_cell*cell_size_row,cell_size_col,cell_size_row), cv::Scalar(200,200,200));
            }
            //--------------------------------------------------------------------------
        }

    }
    //------------------------------------------------------------------------------
    //normalizamos cada vector
    for(i=0;i<N_BLOCKS*N_BLOCKS;i++)
    {
		//We could analize  normalization by sum
        Preprocessing::normalizeVector(histograms[i], params.n_bins, normalization_type);
    }
    //------------------------------------------------------------------------------
    //Concatenamos los histogramas
    int des_final_size=params.n_bins*N_BLOCKS*N_BLOCKS;
    float* descriptor=new float[des_final_size];
    for(i=0;i<params.n_bins;i++)
    {
        for(j=0; j<N_BLOCKS*N_BLOCKS;j++)
        {
            descriptor[i+j*params.n_bins]=(histograms[j])[i];
        }
    }
    //square root normalization
    if(params.squared_root){
    	for(i=0; i<des_final_size;i++){
    		descriptor[i]=std::pow(descriptor[i],0.5);
    	}
    }
    //------------------------------------------------------------------------------
    //liberarmos memoria
    for(j=0; j<N_BLOCKS*N_BLOCKS;j++)
    {
        delete[] histograms[j];
    }

    //------------------------------------------------------------------------------
    return descriptor;
}
/*-----------------------------------------------------------------------------------------*/
//This is Soft Histogram of Edge Local Orientations SHELO with Mask
float *Descriptor::getLocalHELODescriptor_with_mask(cv::Mat image, HELOParams params, cv::Mat mask,
		bool draw, cv::Mat* im_draw, cv::Scalar color)//S_HELO
{
    //image is a CV_U8C1 image
    float TH_MAG=0.1;
    int N_BLOCKS=params.n_blocks;
    int normalization_type=params.normalization;

    std::vector<float> sines(params.n_cells*params.n_cells);
    std::vector<float> cosines(params.n_cells*params.n_cells);
    std::vector<float> cell_magnitudes(params.n_cells*params.n_cells);
    std::vector<float> cell_count(params.n_cells*params.n_cells);
    std::vector<float*> histograms(N_BLOCKS*N_BLOCKS);

    int i=0, j=0;
    //----------------------------------------------------------------------
    //Inicialization
    for(i=0; i<params.n_cells*params.n_cells; i++)
    {
        sines[i]=0;
        cosines[i]=0;
        cell_magnitudes[i]=0;
        cell_count[i]=0;
    }
    for(i=0; i<N_BLOCKS*N_BLOCKS; i++)
    {
        histograms[i]=new float[params.n_bins];
        for(j=0;j<params.n_bins;j++)
        {
            histograms[i][j]=0;
        }
    }
    //-----------------------------------------------------------------------
    //Usamos sobel para obtener las orientaciones
    cv::Mat Gx=(cv::Mat_<float>(3,3)<<-1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gy;
    transpose(Gx,Gy);

    cv::Mat im_x=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat im_y=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat image_f;
    image.convertTo(image_f, CV_32F);
    filter2D(image_f, im_x, CV_32F, Gx);
    filter2D(image_f, im_y, CV_32F, Gy);
    cv::Mat angles=cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat magnitudes=cv::Mat::zeros(image.size(), CV_32F);
    //--------------------------------------------------
    //Calculamos àngulos
    for(i=0; i<image.rows; i++)
    {
        for(j=0; j<image.cols; j++)
        {
            angles.at<float>(i,j)=atan2(im_y.at<float>(i,j), im_x.at<float>(i,j));
            if(angles.at<float>(i,j)<0) angles.at<float>(i,j)+=2*PI;//varies between 0 and 2*PI
            magnitudes.at<float>(i,j)=sqrt(im_x.at<float>(i,j)*im_x.at<float>(i,j)+im_y.at<float>(i,j)*im_y.at<float>(i,j));
        }
    }
    //--------------------------------------------------
    float pos_x=0;
    float pos_y=0;
    int x_izq=0, x_der=0, y_der=0, y_izq=0;
    float w_x_izq=0, w_x_der=0, w_y_der=0, w_y_izq=0;
    float angle=0, mag=0;
    float d_angle=0;
    float max_mag=0;
    int idx=0;
    float weight=0;
    for(i=0; i<image.rows; i++)
    {
        for(j=0; j<image.cols; j++)
        {
            if(mask.at<uchar>(i,j)==1)
            {
                angle=angles.at<float>(i,j);
                mag=magnitudes.at<float>(i,j);
                pos_x=(j/static_cast<float>(image.cols))*params.n_cells;
                pos_y=(i/static_cast<float>(image.rows))*params.n_cells;
                Preprocessing::linearInterBIN(pos_x, &x_izq, &x_der, &w_x_izq, &w_x_der);
                Preprocessing::linearInterBIN(pos_y, &y_izq, &y_der, &w_y_izq, &w_y_der);
                //--------------------------------------------------------------------------------
                //Interpolacion bi-lineal en las celdas
                if(isValid(x_izq, y_izq, 0, params.n_cells-1, 0, params.n_cells-1))
                {
                    idx=y_izq*params.n_cells+x_izq;
                    weight=w_x_izq*w_y_izq;

                    sines[idx]+=mag*2*sin(angle)*cos(angle);
                    cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                    cell_magnitudes[idx]+=mag*weight;
                    if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                    cell_count[idx]+=weight;
                }
                if(isValid(x_izq, y_der, 0, params.n_cells-1, 0, params.n_cells-1))
                {
                    idx=y_der*params.n_cells+x_izq;
                    weight=w_x_izq*w_y_der;

                    sines[idx]+=mag*2*sin(angle)*cos(angle);
                    cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                    cell_magnitudes[idx]+=mag*weight;
                    if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                    cell_count[idx]+=weight;
                }
                if(isValid(x_der, y_izq, 0, params.n_cells-1, 0, params.n_cells-1))
                {
                    idx=y_izq*params.n_cells+x_der;
                    weight=w_x_der*w_y_izq;

                    sines[idx]+=mag*2*sin(angle)*cos(angle);
                    cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                    cell_magnitudes[idx]+=mag*weight;
                    if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                    cell_count[idx]+=weight;
                }
                if(isValid(x_der, y_der, 0, params.n_cells-1, 0, params.n_cells-1))
                {
                    idx=y_der*params.n_cells+x_der;
                    weight=w_x_der*w_y_der;

                    sines[idx]+=mag*2*sin(angle)*cos(angle);
                    cosines[idx]+=mag*(cos(angle)*cos(angle)-sin(angle)*sin(angle));
                    cell_magnitudes[idx]+=mag*weight;
                    if(cell_magnitudes[idx]>max_mag) max_mag=cell_magnitudes[idx];
                    cell_count[idx]+=weight;
                }
            }
            //--------------------------------------------------------------------------------
        }
    }
    //--------------------------------------------------
    //Promedio de magnitudes
    //for(i=0; i<params.n_cells*params.n_cells; i++) cell_magnitudes[i]/=max_mag;
    //Es posible agregar un factor de confianza, las magnitudes dan la certeza
    //--------------------------------------------------
    if(draw)
    {
        //int stepx=image.cols/params.n_cells;
        //int stepy=image.rows/params.n_cells;
        image.copyTo(*im_draw);
        cvtColor(*im_draw, *im_draw, CV_GRAY2BGR);
        //Preprocessing::drawGrid(im_draw, stepx, stepy, cv::Scalar(0,0,255));
    }

    //--------------------------------------------------
    //Creamos el descriptor obteniendo el ángulo por celda
    float *descriptor_aux;
    /*---------------------------------------------------*/
    float bin=0;
    int bin_izq=0, bin_der=0;
    float w_bin_izq=0, w_bin_der=0;
    float y_block=0, x_block=0;
    int cell_size_col=floor(image.cols/params.n_cells);
    int cell_size_row=floor(image.rows/params.n_cells);
    int i_cell=0, j_cell=0;
    float xi=0, yi=0, xc=0, yc=0, xf=0, yf=0, x_=0, y_=0;
    float certeza=0;
    /*---------------------------------------------------*/
    for(i=0;i<params.n_cells*params.n_cells;i++)
    {
        //cell positions-----------------------------------
        i_cell=floor(i/params.n_cells);
        j_cell=i-i_cell*params.n_cells;
        //-------------------------------------------------
        d_angle=atan2(sines[i], cosines[i]);
        if(d_angle<0) d_angle+=2*PI;
        d_angle=d_angle*0.5;//it must vary between 0 and pi
        /*add pi/2 to get the actual orientation*/
        if(d_angle>PI*0.5) d_angle-=PI*0.5;
        else d_angle+=PI*0.5;
        /*-----------------------------------------------*/
        bin=(d_angle/PI)*params.n_bins;
        Preprocessing::linearInterBIN(bin, &bin_izq, &bin_der, &w_bin_izq, &w_bin_der);
        if(bin_izq==params.n_bins) bin_izq=0;
        if(bin_izq<0) bin_izq=params.n_bins-1;
        if(bin_der==params.n_bins) bin_der=0;
        if(bin_der<0) bin_der=params.n_bins-1;
        certeza=cell_magnitudes[i]/max_mag;//se debe buscar magnitudes locales

        if(certeza>TH_MAG)
        {
            x_block=(j_cell/static_cast<float>(params.n_cells))*N_BLOCKS;
            y_block=(i_cell/static_cast<float>(params.n_cells))*N_BLOCKS;

            Preprocessing::linearInterBIN(x_block, &x_izq, &x_der, &w_x_izq, &w_x_der);
            Preprocessing::linearInterBIN(y_block, &y_izq, &y_der, &w_y_izq, &w_y_der);

            if(isValid(x_izq, y_izq, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_izq+y_izq*N_BLOCKS;
                weight=w_x_izq*w_y_izq;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            if(isValid(x_izq, y_der, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_izq+y_der*N_BLOCKS;
                weight=w_x_izq*w_y_der;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            if(isValid(x_der, y_izq, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_der+y_izq*N_BLOCKS;
                weight=w_x_der*w_y_izq;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            if(isValid(x_der, y_der, 0, N_BLOCKS-1, 0, N_BLOCKS-1))
            {
                idx=x_der+y_der*N_BLOCKS;
                weight=w_x_der*w_y_der;

                descriptor_aux=histograms[idx];
                descriptor_aux[bin_izq]+=cell_magnitudes[i]*weight+w_bin_izq;
                descriptor_aux[bin_der]+=cell_magnitudes[i]*weight+w_bin_der;
            }

            //dibujamos el vector de orientación local--------------------------------
            if(draw)
            {
                xi=(j_cell+0.2)*cell_size_col;
                xc=(j_cell+0.5)*cell_size_col;
                yc=(i_cell+0.5)*cell_size_row;
                yi=yc;
                xf=(j_cell+0.8)*cell_size_col;
                yf=yi;
                xi=xc-xi;
                yi=yc-yi;

                xf=xc-xf;
                yf=yc-yf;

                x_=cos(d_angle)*xi-sin(d_angle)*yi;
                y_=sin(d_angle)*xi+cos(d_angle)*yi;
                xi=x_+xc;
                yi=y_+yc;

                x_=cos(d_angle)*xf-sin(d_angle)*yf;
                y_=sin(d_angle)*xf+cos(d_angle)*yf;
                xf=x_+xc;
                yf=y_+yc;
                line(*im_draw, cv::Point(xi,yi), cv::Point(xf, yf), color,2);

                //int degree_angle=(int)(d_angle*180.0/M_PI);
                //putText(*im_draw, Preprocessing::intToString(degree_angle), cv::Point(xi,yi),1,0.7,cv::Scalar(255,0,0));
                //imshow(name_w, im_draw);
                //std::cout<<(d_angle*180)/PI<<" -->" <<color_factor<<std::endl;
                //waitKey();
            }
            //--------------------------------------------------------------------------
        }

    }
    //------------------------------------------------------------------------------
    //normalizamos cada vector
    for(i=0;i<N_BLOCKS*N_BLOCKS;i++)
    {
                //We could analize  normalization by sum
        Preprocessing::normalizeVector(histograms[i], params.n_bins, normalization_type);
    }
    //------------------------------------------------------------------------------
    //Concatenamos los histogramas
    float* descriptor=new float[params.n_bins*N_BLOCKS*N_BLOCKS];
    for(i=0;i<params.n_bins;i++)
    {
        for(j=0; j<N_BLOCKS*N_BLOCKS;j++)
        {
            descriptor[i+j*params.n_bins]=(histograms[j])[i];
        }
    }
    //------------------------------------------------------------------------------
    //liberarmos memoria
    for(j=0; j<N_BLOCKS*N_BLOCKS;j++)
    {
        delete[] histograms[j];
    }
    //------------------------------------------------------------------------------
    return descriptor;
}
/*--------------------------------------------------------------------------------------*/
//deprecated
float *Descriptor::getSHELO_MS(cv::Mat image, SHELO_MSParams params)
{
    int des_size=params.getDescriptorSize();
    int des_size_level=params.getDescriptorSizeByLevel();
    cv::Mat image_g;
    double sigma=1;
    int filter_size=7;
    //std::cout<<"sizes--> "<<des_size<<" "<<des_size_level<<std::endl;
    float *final_descriptor=new float[des_size];
    float *descriptor=NULL;
    cv::Mat ed;
    for(int l=0; l<params.n_levels; l++)
    {
        cv::GaussianBlur(image, image_g, cv::Size(filter_size,filter_size), sigma);
        ed=Preprocessing::canny(image_g,-1,-1,0.5);
        descriptor=getLocalHELODescriptor(ed*255, HELOParams(params.n_cells_by_level[l],
                                                            params.n_bins,
                                                            params.n_blocks,
                                                            params.normalization));
        std::copy(descriptor,
                  descriptor+des_size_level,
                  final_descriptor+l*des_size_level);
        delete[] descriptor;
        sigma=sigma*2;
        filter_size=filter_size*2-1;
        image_g.release();
        ed.release();
    }
    return final_descriptor;
}
/*--------------------------------------------------------------------------------------*/
float *Descriptor::getSHELO_SP(cv::Mat image, SHELO_SPParams params)
{
    assert(image.type()==CV_8UC1);
	int des_size=params.getDescriptorSize();
	int des_size_level=0;
    cv::Mat image_g;
    float *final_descriptor=new float[des_size];
    float *descriptor=NULL;
    float *offset_final_descriptor=final_descriptor;
    for(int l=0; l<params.n_levels; l++)
    {
    	descriptor=getLocalHELODescriptor(image, params.sp_params[l]);
        des_size_level=params.getDescriptorSizeByLevel(l);
        std::copy(descriptor,
                  descriptor+des_size_level,
                  offset_final_descriptor);

        offset_final_descriptor=offset_final_descriptor+des_size_level;
        delete[] descriptor;
    }
    return final_descriptor;
}
/*--------------------------------------------------------------------------------------*/
//Function for detectin dense points
void Descriptor::computeDenseKeypoints(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, int stepx, int stepy, int npoints)
{
    int width=image.cols;
    int height=image.rows;
    int x=0, y=0;
    keypoints.clear();
    if(stepx==-1 || stepy==-1)
    {
        int np=static_cast<int>(round(sqrt(npoints)));
        stepx=width/(np+1);
        stepy=height/(np+1);
    }    
    for(x=stepx; x<width; x+=stepx)
    {
        for(y=stepy; y<height; y+=stepy)
        {
            keypoints.push_back(cv::KeyPoint(x,y,5));
        }
    }
}
/*--------------------------------------------------------------------------------------*/
void Descriptor::SHELO_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, HELOParams params)
{
    cv::Mat imageg;
    if(image.channels()==3) cvtColor(image,imageg,CV_BGR2GRAY);
    else image.copyTo(imageg);
    //--------------------------------------
    int n_kp=0;
    n_kp=keypoints.size();
    int x=0, y=0, size=0;
    int xf=0, yf=0;
    int i_kp=0, i=0;
    cv::Rect rect;
    cv::Mat roi;
    float *descriptor=NULL;
    int des_size=params.getDescriptorSize();
    descriptors=cv::Mat(n_kp,des_size, CV_32F);
    //--------------------------------------
    for(i_kp=0; i_kp<n_kp;  i_kp++)
    {
        x=keypoints[i_kp].pt.x;
        y=keypoints[i_kp].pt.y;
        size=3*(keypoints[i_kp].size);
        x=std::max(x-size,0);
        y=std::max(y-size,0);
        xf=std::min(x+2*size-1,image.cols-1);
        yf=std::min(y+2*size-1,image.rows-1);
        rect=cv::Rect(x,y,xf-x+1, yf-y+1);
        roi=imageg(rect);

        descriptor=getLocalHELODescriptor(roi, params);
        for(i=0;i<des_size;i++) descriptors.at<float>(i_kp,i)=descriptor[i];
        delete[] descriptor;

    }
}
/*--------------------------------------------------------------------------------------*/
void Descriptor::HOG_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, HOGParams params)
{
    cv::Mat imageg;
    if(image.channels()==3) cvtColor(image,imageg,CV_BGR2GRAY);
    else image.copyTo(imageg);
    //--------------------------------------
    int n_kp=0;
    n_kp=keypoints.size();
    int x=0, y=0, size=0;
    int xf=0, yf=0;
    int i_kp=0, i=0;
    cv::Rect rect;
    cv::Mat roi;
    float *descriptor=NULL;
    int n_width=60;
    int n_height=60;
    int des_size=params.getDescriptorSize(cv::Size(n_width, n_height));
    //std::cout<<"calculando tamaño"<<std::endl;
    //int des_size=estimate_HOG_VL_size(cv::Size(60,60), params);
    descriptors=cv::Mat(n_kp,des_size, CV_32F);
    //--------------------------------------
    //std::cout<<"Calculando descriptores"<<std::endl;
    for(i_kp=0; i_kp<n_kp;  i_kp++)
    {
        x=keypoints[i_kp].pt.x;
        y=keypoints[i_kp].pt.y;
        size=30;//6*(keypoints[i_kp].size);
        x=std::max(x-size,0);
        y=std::max(y-size,0);
        xf=std::min(x+2*size-1,image.cols-1);
        yf=std::min(y+2*size-1,image.rows-1);
        rect=cv::Rect(x,y,xf-x+1, yf-y+1);
        roi=imageg(rect);
        //---------------------------------------------
        resize(roi,roi,cv::Size(n_width, n_height));

        descriptor=getHOGDalalTriggs(roi, params);
        for(i=0;i<des_size;i++) descriptors.at<float>(i_kp,i)=descriptor[i];
        delete[] descriptor;
    }
    //--------------------------------------
}
/*--------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------*/
//This use LBP for faces, VLFeat does not work!!
void Descriptor::LBP_COLOR_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, LBPParams params)
{
    //Requires a color image
    //---------------------------------------for color images
    cv::Mat image_r, image_g, image_b;
    if(image.channels()==3)
    {
        std::vector<cv::Mat> images;
        images.clear();
        split(image, images);
        images[0].copyTo(image_b);
        images[1].copyTo(image_g);
        images[2].copyTo(image_r);
    }
    //---------------------------------------if it is a gray image
    else
    {
        image.copyTo(image_b);
        image.copyTo(image_g);
        image.copyTo(image_r);
    }
    //--------------------------------------
    int n_kp=0;
    n_kp=keypoints.size();
    int x=0, y=0, size=0;
    int xf=0, yf=0;
    int i_kp=0, i=0;
    cv::Rect rect;
    cv::Mat roi_r, roi_g, roi_b;
    //--------------------------------------
    float *descriptor_r=NULL;
    float *descriptor_g=NULL;
    float *descriptor_b=NULL;
    //std::cout<<"calculando tamaño"<<std::endl;
    int des_size=params.getDescriptorSize();
    descriptors=cv::Mat(n_kp,des_size*3, CV_32F);
    //--------------------------------------
    //std::cout<<"Calculando descriptores"<<std::endl;
    for(i_kp=0; i_kp<n_kp;  i_kp++)
    {
        x=keypoints[i_kp].pt.x;
        y=keypoints[i_kp].pt.y;
        size=30;//5*(keypoints[i_kp].size);
        x=std::max(x-size,0);
        y=std::max(y-size,0);
        xf=std::min(x+2*size-1,image.cols-1);
        yf=std::min(y+2*size-1,image.rows-1);
        int min_dim=std::min(xf-x+1,yf-y+1); //to ensure square rois
        rect=cv::Rect(x,y,min_dim, min_dim);

        roi_b=image_b(rect);
        roi_g=image_g(rect);
        roi_r=image_r(rect);

        descriptor_r=getLBP_Face(roi_r, params);
        descriptor_g=getLBP_Face(roi_g, params);
        descriptor_b=getLBP_Face(roi_b, params);

        for(i=0;i<des_size;i++)
        {
            descriptors.at<float>(i_kp,i)=descriptor_r[i];
            descriptors.at<float>(i_kp,des_size+i)=descriptor_g[i];
            descriptors.at<float>(i_kp,2*des_size+i)=descriptor_b[i];
        }
        delete[] descriptor_r;
        delete[] descriptor_g;
        delete[] descriptor_b;
        //std::cout<<x<<"-"<<y<<"-("<<size<<")-"<<xf-x+1<<"-"<<yf-y+1<<std::endl;
    }
}
/*--------------------------------------------------------------------------------------*/
//This is an implementation used for face recognition
void Descriptor::LBP_extractor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &descriptors, LBPParams params)
{
    cv::Mat imageg; //image must be a 8UC1-type image
    if(image.channels()==3) cvtColor(image,imageg,CV_BGR2GRAY);
    else image.copyTo(imageg);
    //--------------------------------------
    int n_kp=0;
    n_kp=keypoints.size();
    int x=0, y=0, size=0;
    int xf=0, yf=0;
    int i_kp=0, i=0;
    cv::Rect rect;
    cv::Mat roi;
    float *descriptor=NULL;
    //std::cout<<"calculando tamaño"<<std::endl;
    int des_size=params.getDescriptorSize();
    descriptors=cv::Mat(n_kp,des_size, CV_32F);
    //--------------------------------------
    //std::cout<<"Calculando descriptores"<<std::endl;
    for(i_kp=0; i_kp<n_kp;  i_kp++)
    {
        x=keypoints[i_kp].pt.x;
        y=keypoints[i_kp].pt.y;
        size=30;//5*(keypoints[i_kp].size);
        x=std::max(x-size,0);
        y=std::max(y-size,0);
        xf=std::min(x+2*size-1,image.cols-1);
        yf=std::min(y+2*size-1,image.rows-1);
        int min_dim=std::min(xf-x+1,yf-y+1); //to ensure square rois
        rect=cv::Rect(x,y,min_dim, min_dim);
        roi=imageg(rect);
        //-----------------------------------
        descriptor=getLBP_Face(roi, params);
        for(i=0;i<des_size;i++) descriptors.at<float>(i_kp,i)=descriptor[i];
        delete[] descriptor;
    }
}
/*----------------------------------------------------------------------------------*/
//lbp_extended is not my code
void Descriptor::lbp_extendend(cv::InputArray _src, cv::OutputArray _dst, int radius, int neighbors)
{
    //src is a gray image 8UC1
    //get matrices
    cv::Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    cv::Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++)
    {
        // sample points, se han cambiado los planos para generar un mov, anti-horario
        //sin(90+alpha)=cos(alpha)
        //cos(90+alpha)=-sin(alpha), esto esta en la formulas
        float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights bilineal
        float w1 = (1 - tx) * (1 - ty);// fy fx
        float w2 =      tx  * (1 - ty);//fy cx
        float w3 = (1 - tx) *      ty;// cy fx
        float w4 =      tx  *      ty; //cy  cx
        // iterate through your data
        for(int i=radius; i<src.rows-radius;i++)
        {
            for(int j=radius; j<src.cols-radius;j++)
            {
                // calculate interpolated value
                float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx);
                      t+= w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i-radius,j-radius) += ((t > src.at<uchar>(i,j)) ||
                                                   (std::abs(t-src.at<uchar>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}
/*----------------------------------------------------------------------------------*/
float *Descriptor::lbp_spatial_histogram(cv::InputArray _src, int numPatterns, int grid_x, int grid_y, bool normed)
{
    //TODO: Bilineal with respect to the cells
    //input is the lbl dst
    cv::Mat src = _src.getMat();
    src.convertTo(src,CV_32F);//working in float in orde to use the calcHist function
    // calculate LBP patch size
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    // allocate memory for the spatial histogram
    int size_des=numPatterns*grid_x*grid_y;
    float *all_histograms=new float[size_des];
    for(int i=0;i<size_des;i++) all_histograms[i]=0;
    // return matrix with zeros if no data is given
    if(src.empty())  return NULL;
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    int histSize = numPatterns;
    int min_val=0, max_val=255;//this only must be used for n=8;
    float range[] = {(float)min_val, (float)max_val} ;
    const float* histRange = {range};
    int k=0;
    float total=0;
    int pos_ini=0;
    for(int i = 0; i < grid_y; i++)
    {
        for(int j = 0; j < grid_x; j++)
        {
            //extract a cell
            cv::Mat src_cell = cv::Mat(src, cv::Range(i*height,(i+1)*height), cv::Range(j*width,(j+1)*width));
            //Computing histogram
            cv::Mat local_hist;
            // calc histogram //the input must be only a 8U or a 32F
            calcHist(&src_cell, 1, 0, cv::Mat(), local_hist, 1, &histSize, &histRange, true, false);
            local_hist.convertTo(local_hist,CV_32F);
            // normalize
            total=sum(local_hist)[0];

            pos_ini=resultRowIdx*numPatterns;
            for(k=0;k<numPatterns;k++)
            {
                all_histograms[pos_ini+k]=local_hist.at<float>(k,1);
                //the normalization is by sum
                if(normed)  all_histograms[pos_ini+k]=all_histograms[pos_ini+k]/(total+EPS);
            }
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return all_histograms;
}
/*----------------------------------------------------------------------------------*/
//This implementation is a simple LBP descriptor used for face recognition
float *Descriptor::getLBP_Face(cv::Mat image, LBPParams params)
{
    cv::Mat imageg;
    cv::Mat dst;
    if(image.channels()==3) cvtColor(image, imageg, CV_BGRA2GRAY);
    else image.copyTo(imageg);
    lbp_extendend(imageg,dst,params.radius,params.n_neighbors);
    float *descriptor=lbp_spatial_histogram(dst, params.quantize_value, params.num_cells, params.num_cells,true);
    return descriptor;
}
/*----------------------------------------------------------------------------------*/
//The same as getLBP_Face, but with a better name
float *Descriptor::getExtendedLBPDescriptor(cv::Mat image, LBPParams params)
{
    return getLBP_Face(image, params);
}
/*----------------------------------------------------------------------------------*/
// Simple LBPDescriptor JMSR
float *Descriptor::getSimpleLBPDescriptor(cv::Mat image, int n_bins, int normed)
{
    // It requires image to be of CV_U8C1

    float pos[8][2]={{-1,0},
                 {-1,-1},
                 {0,-1},
                 {+1,-1},
                 {+1,0},
                 {+1,+1},
                 {0,+1}};
    /*--------------------------------------------------------------------*/
    //Calculating the lbp code for each pixel
    int i=0, j=0;
    cv::Mat lbp_val=cv::Mat::zeros(image.size(), CV_8UC1);
    int neighbors=8;
    int p_neig=0;
    int t=0;
    for (p_neig=0; p_neig<neighbors; p_neig++)
    {
        for(i=1; i<image.rows-1; i++)
        {
            for(j=1; j<image.cols-1; j++)
            {
                t=image.at<uchar>(i+pos[p_neig][0], j+pos[p_neig][1]);
                lbp_val.at<uchar>(i,j)+=(t>(int)lbp_val.at<uchar>(i,j))<<p_neig;
            }
        }
    }

    lbp_val=lbp_val(cv::Rect(1,1,image.cols-2, image.rows-2));//to discar the frontier [were pad to 0]
    /*--------------------------------------------------------------------*/
    //Computing a histogram with "n_bins" bins
    int channels[]= {0};
    int hsize=n_bins;
    float hranges[]={0,256};
    const float* ranges[]={hranges};
    cv::Mat hist;
    cv::calcHist(&lbp_val, 1, channels, cv::Mat(), hist,1, &hsize, ranges);
    /*--------------------------------------------------------------------*/
    //normalizing
    hist.convertTo(hist,CV_32F);
    float *des=new float[n_bins];
    for(i=0;i<n_bins;i++)
    {
        des[i]=hist.at<float>(i,0);
    }    
    Preprocessing::normalizeVector(des,n_bins, normed);
    return des;
}
/*----------------------------------------------------------------------------------*/
unsigned char Descriptor::bin_lbp_8bits(unsigned char val)
{
    unsigned char table_8bits[256];
    for(unsigned int i=0; i<256; i++) table_8bits[i]=58;
    table_8bits[0]=0;   table_8bits[1]=1;   table_8bits[2]=2;
    table_8bits[3]=3;   table_8bits[4]=4;   table_8bits[6]=5;
    table_8bits[7]=6;   table_8bits[8]=7;   table_8bits[12]=8;
    table_8bits[14]=9;  table_8bits[15]=10; table_8bits[16]=11;
    table_8bits[24]=12; table_8bits[28]=13; table_8bits[30]=14;
    table_8bits[31]=15; table_8bits[32]=16; table_8bits[48]=17;
    table_8bits[56]=18; table_8bits[60]=19; table_8bits[62]=20;
    table_8bits[63]=21; table_8bits[64]=22; table_8bits[96]=23;
    table_8bits[112]=24;table_8bits[120]=25;table_8bits[124]=26;
    table_8bits[126]=27;table_8bits[127]=28;table_8bits[128]=29;
    table_8bits[129]=30;table_8bits[131]=31;table_8bits[135]=32;
    table_8bits[143]=33;table_8bits[159]=34;table_8bits[191]=35;
    table_8bits[192]=36;table_8bits[193]=37;table_8bits[195]=38;
    table_8bits[199]=39;table_8bits[207]=40;table_8bits[223]=41;
    table_8bits[224]=42;table_8bits[225]=43;table_8bits[227]=44;
    table_8bits[231]=45;table_8bits[239]=46;table_8bits[240]=47;
    table_8bits[241]=48;table_8bits[243]=49;table_8bits[247]=50;
    table_8bits[248]=51;table_8bits[249]=52;table_8bits[251]=53;
    table_8bits[252]=54;table_8bits[253]=55;table_8bits[254]=56;
    table_8bits[255]=57;
    return table_8bits[val];
}

/*----------------------------------------------------------------------------------*/
float *Descriptor::get_lbp_8bits(cv::Mat image, int radius, int normed)
{
    cv::Mat imageg;
    cv::Mat dst;
    if(image.channels()==3) cv::cvtColor(image, imageg, CV_BGR2GRAY);
    else image.copyTo(imageg);
    lbp_extendend(imageg,dst,radius,8);
    unsigned char size_des=59;

    float *descriptor=new float[size_des];
    for(unsigned char i=0; i<size_des;i++) descriptor[i]=0;

    int *data=reinterpret_cast<int*>(dst.data);
    unsigned char bin=0;
    for(int i=0; i<dst.rows*dst.cols;i++)
    {
        bin=bin_lbp_8bits(static_cast<unsigned char>(data[i]));
        descriptor[bin]++;
    }
    Preprocessing::normalizeVector(descriptor,size_des,normed);
    return descriptor;
}
/*----------------------------------------------------------------------------------*/
float *Descriptor::get_lbp_8bits_grid(cv::Mat image, LBPParams params, int normed)
{
    //there are 59 uniform patterns -> Ojala experimentations
    if (params.n_neighbors!=8 || params.quantize_value!=59)
    {
        std::cerr<<"Error: n_neighbors != 8 or quantize_value !=59"<<std::endl;
        exit(EXIT_FAILURE);
    }
    cv::Mat imageg;
    cv::Mat dst;
    //--------------------------------getting gray image
    if(image.channels()==3) cv::cvtColor(image, imageg, CV_BGR2GRAY);
    else image.copyTo(imageg);
    //--------------------------------computing LBP coding
    lbp_extendend(imageg,dst,params.radius,8);
    //--------------------------------computing histograms by soft computation
    float pos_x=0, pos_y=0;
    int x_izq=0, x_der=0, y_der=0, y_izq=0;
    float w_x_izq=0, w_x_der=0, w_y_der=0, w_y_izq=0;
    float weight=0;
    unsigned char val=0;
    int bin=0, idx=0;
    float **histogram=new float*[params.num_cells*params.num_cells];
    float *descriptor=new float[params.num_cells*params.num_cells*params.quantize_value];
    //--------------------------------set to 0 all histograms
    for(int i=0; i<params.num_cells*params.num_cells; i++)
    {
        histogram[i]=new float[params.quantize_value];
        for(bin=0; bin<params.quantize_value; bin++)
        {
            histogram[i][bin]=0;
        }
    }
    //--------------------------------compuitng local histograms by interpolation
    for(int i=0; i<dst.rows; i++)
    {
        for(int j=0; j<dst.cols; j++)
        {
            val=static_cast<unsigned char>(dst.at<int>(i,j));
            bin=(int)bin_lbp_8bits(val);
            pos_x=(j/static_cast<float>(image.cols))*params.num_cells;
            pos_y=(i/static_cast<float>(image.rows))*params.num_cells;
            Preprocessing::linearInterBIN(pos_x, &x_izq, &x_der, &w_x_izq, &w_x_der);
            Preprocessing::linearInterBIN(pos_y, &y_izq, &y_der, &w_y_izq, &w_y_der);
            if(isValid(x_izq, y_izq, 0, params.num_cells-1, 0, params.num_cells-1))
            {
                idx=y_izq*params.num_cells+x_izq;
                weight=w_x_izq*w_y_izq;
                histogram[idx][bin]+=val*weight;
            }
            if(isValid(x_izq, y_der, 0, params.num_cells-1, 0, params.num_cells-1))
            {
                idx=y_der*params.num_cells+x_izq;
                weight=w_x_izq*w_y_der;
                histogram[idx][bin]+=val*weight;
            }
            if(isValid(x_der, y_izq, 0, params.num_cells-1, 0, params.num_cells-1))
            {
                idx=y_izq*params.num_cells+x_der;
                weight=w_x_der*w_y_izq;
                histogram[idx][bin]+=val*weight;
            }

            if(isValid(x_der, y_der, 0, params.num_cells-1, 0, params.num_cells-1))
            {
                idx=y_der*params.num_cells+x_der;
                weight=w_x_der*w_y_der;
                histogram[idx][bin]+=val*weight;
            }
        }
    }
    //--------------------------- normalizing and concatenating all histograms
    for(int i=0; i<params.num_cells*params.num_cells; i++)
    {
        Preprocessing::normalizeVector(histogram[i],params.quantize_value, normed);
        std::copy(histogram[i],
                  histogram[i]+params.quantize_value,
                  descriptor+i*params.quantize_value);
        //---------------cleaning memory
        delete[] histogram[i];
    }
    delete[] histogram;
    //----------------------------
    return descriptor;
}
/*----------------------------------------------------------------------------------*/
float *Descriptor::get_color_lbp_8bits_grid(cv::Mat image, LBPParams params, int normed)
{
    if (params.n_neighbors!=8 || params.quantize_value!=59)
    {
        std::cerr<<"Error: n_neighbors != 8 or quantize_value !=59"<<std::endl;
        exit(EXIT_FAILURE);
    }
    if (image.channels()!=3)
    {
        std::cerr<<"The input must be a valid RGB image"<<std::endl;
        exit(EXIT_FAILURE);
    }
    cv::Mat oRGB_image;
    Preprocessing::BGR2oRGB(image, oRGB_image); //oRGB_image is CV_32FC3
    std::vector<cv::Mat> oRGB_channels;
    cv::split(oRGB_image,oRGB_channels);
    int size_by_channel=params.getDescriptorSize();
    float *descriptor_L=get_lbp_8bits_grid(oRGB_channels[0], params, normed);
    float *descriptor_C1=get_lbp_8bits_grid(oRGB_channels[1], params, normed);
    float *descriptor_C2=get_lbp_8bits_grid(oRGB_channels[2], params, normed);
    //-------------------------------------------concatenating descriptor by channel
    float *final_descriptor=new float[size_by_channel*3];
    std::copy(descriptor_L, descriptor_L+size_by_channel, final_descriptor);
    std::copy(descriptor_C1, descriptor_C1+size_by_channel, final_descriptor+size_by_channel);
    std::copy(descriptor_C2, descriptor_C2+size_by_channel, final_descriptor+2*size_by_channel);
    delete[] descriptor_L;
    delete[] descriptor_C1;
    delete[] descriptor_C2;
    return final_descriptor;
}

/*----------------------------------------------------------------------------------*/
// Compute GrayLayoutDescriptor is inherited from ColorLayout descriptor
float *Descriptor::getGrayLayoutDescriptor(cv::Mat _image, CLDParams params)
{
    cv::Mat subim=cv::Mat::zeros(params.num_cells_y, params.num_cells_x,CV_32FC1);
    float *des=new  float[params.getDescriptorSize()];
    int *n_pixels=new int[params.getDescriptorSize()];
    //-------------------------initializint to zero
    for(int i=0;i<params.getDescriptorSize(); i++) n_pixels[i]=0;
    //---------------------------------------------
    cv::Mat image;
    _image.convertTo(image,CV_8UC1);
    int sub_i=0, sub_j=0;
    //---------------------------------------------
    //getting the subim
    //std::cout<<image<<std::endl;
    for(int i=0; i<image.rows; i++)
    {
        for(int j=0; j<image.cols; j++)
        {
            //it is possible to apply bilinear interpolation
            sub_i=round((i/float(image.rows-1))*(params.num_cells_y-1));
            sub_j=round((j/float(image.cols-1))*(params.num_cells_x-1));

            subim.at<float>(sub_i, sub_j)+=image.at<uchar>(i,j);
            //in the case of bilinear interpolation, n_pixels will store weights
            n_pixels[sub_i*params.num_cells_x+sub_j]++;
        }
    }
    //---------------------------------------------
    for(int sub_i=0; sub_i<params.num_cells_y; sub_i++)
    {
        for(int sub_j=0; sub_j<params.num_cells_x; sub_j++)
        {
            subim.at<float>(sub_i, sub_j)/=n_pixels[sub_i*params.num_cells_x+sub_j];
        }
    }
    //std::cout<<subim<<std::endl;
    delete[] n_pixels;
    //---------------------------------------------
    //computing the discrete cosine transform on the subimage
    cv::Mat dct;
    cv::dct(subim, dct);
    for(int sub_i=0; sub_i<params.num_cells_y; sub_i++)
    {
        for(int sub_j=0; sub_j<params.num_cells_x; sub_j++)
        {
            //Faltaría generar una secuencia de zigzag
            des[sub_i*params.num_cells_x+sub_j]=dct.at<float>(sub_i, sub_j);
        }
    }
    //---------------------------------------------
    return des;
}
/*----------------------------------------------------------------------------------*/
//Discrete Cosine Transform
float *Descriptor::getDCTDescriptor(cv::Mat image, CLDParams params)
{
    cv::Mat imageg;
    cv::Mat imageg_f;
    cv::Mat dct;
    if(image.channels()==3) cv::cvtColor(image,imageg, CV_BGR2GRAY);
    else image.copyTo(imageg);
    resize(imageg,imageg,cv::Size(params.num_cells_x, params.num_cells_y));

    imageg.convertTo(imageg_f,CV_32F);
    cv::dct(imageg_f, dct);
    float *des=new float[params.getDescriptorSize()];
    for(int sub_i=0; sub_i<params.num_cells_y; sub_i++)
    {
        for(int sub_j=0; sub_j<params.num_cells_x; sub_j++)
        {
            //Faltaría generar una secuencia de zigzag
            des[sub_i*params.num_cells_x+sub_j]=dct.at<float>(sub_i, sub_j);
        }
    }
    return des;
}
/*-----------------------------------------------------------*/
float *Descriptor::getRGBColorHistogram(cv::Mat image, int k, int *size)
{
    //K puede ser 16, 32, 64, 128, 256
    cv::Mat reduced_image=cv::Mat(image.size(), image.type());
    *size=k*k*k;
    float* descriptor=new float[k*k*k];
    unsigned char n_bits=0;
    int b=0;
    int g=0;
    int r=0;
    int idx=0;
    int i=0, j=0;
    for(i=0; i<*size; i++) descriptor[i]=0;
    for(i=0; i<image.rows; i++)
    {
        for(j=0; j<image.cols; j++)
        {
            b=(int)image.at<cv::Vec3b>(i,j)[0];
            g=(int)image.at<cv::Vec3b>(i,j)[1];
            r=(int)image.at<cv::Vec3b>(i,j)[2];
            n_bits=8-log2(k);
            b=b>>n_bits;
            g=g>>n_bits;
            r=r>>n_bits;
            idx=(b*k+g)*k+r;
            descriptor[idx]++;
            reduced_image.at<cv::Vec3b>(i,j)[0]=(uchar)(b*255.0/k);
            reduced_image.at<cv::Vec3b>(i,j)[1]=(uchar)(g*255.0/k);
            reduced_image.at<cv::Vec3b>(i,j)[2]=(uchar)(r*255.0/k);
        }
    }    
    //imshow("reduced", reduced_image), waitKey();
//    for(i=0; i<*size; i++) cout<<descriptor[i]<<" ";
//    cout<<endl;
    Preprocessing::normalizeVector(descriptor, *size, NORMALIZE_ROOT_UNIT);
    return descriptor;
}
/*-----------------------------------------------------------*/
float *Descriptor::getRGBColorHistogram_grid3(cv::Mat query, int k, int *size)
{

    int n_rows=3, n_cols=3;
    int total_regions=n_rows*n_cols;
    int size_region=k*k*k;
    *size=size_region*total_regions;
    int cur_size_region=0;

    int region_width=(int)floor(query.cols/(float)n_cols);
    int region_height=(int)floor(query.rows/(float)n_rows);
    cv::Rect rect;
    rect.width=region_width;
    rect.height=region_height;
    float* final_des=new float[*size];
    float *des=NULL;
    int i_region=0;
    for(int i=0; i<n_rows; i++)
    {
        for(int j=0; j<n_cols;j++)
        {
            rect.x=i*rect.width;
            rect.y=j*rect.height;
            des=getRGBColorHistogram(query(rect),k,&cur_size_region);
            if(size_region!=cur_size_region)
            {
                std::cerr<<"Error: RGB Descriptor, incompatible region sizes!!"<<std::endl;
            }
            std::copy(des,des+size_region,final_des+i_region*size_region);
            i_region++;
            delete[] des;
        }
    }
    return final_des;
}

/*-----------------------------------------------------------*/
float* Descriptor::get_SHELO_2_for_image(cv::Mat _image, int *size_des, cv::Mat mask)
{
    float *descriptor=NULL;
    cv::Mat image;
    _image.copyTo(image);
    if(image.channels()==3) // to grayscale
    {
        cv::cvtColor(image,image,CV_BGR2GRAY);
    }
    HELOParams params(25,36,6);
    cv::GaussianBlur(image, image, cv::Size(7,7),1);
   //image=Preprocessing::anisotropicDiffusion(image,0.25,10,60);
    if(mask.empty())
    {
        descriptor=Descriptor::getLocalHELODescriptor(image,params);
    }
    else
    {
        descriptor=Descriptor::getLocalHELODescriptor_with_mask(image,params,mask);
    }
    *size_des=params.getDescriptorSize();
    return descriptor;
}
/*-----------------------------------------------------------*/
float* Descriptor::get_SHELO_2_for_image(const char* image_file, int *size_des, cv::Mat mask)
{
    float *descriptor=NULL;
    cv::Mat image=cv::imread(std::string(image_file));
    descriptor=get_SHELO_2_for_image(image, size_des, mask);
    return descriptor;
}
/*-----------------------------------------------------------*/
float* Descriptor::get_HOG_Dalal_for_image(cv::Mat _image, int *size_des)
{
    float* descriptor;
    cv::Mat image;
    _image.copyTo(image);
    if(image.channels()==3) // to grayscale
    {
        cv::cvtColor(image,image,CV_BGR2GRAY);
    }
    HOGParams params (18,5,9,0,cv::Size(200,200));
    //image=Preprocessing::anisotropicDiffusion(image,0.25,5,5);
    resize(image,image, params.getSizeParam());
    descriptor=Descriptor::getHOGDalalTriggs(image,params);
    *size_des=params.getDescriptorSize();
    return descriptor;
}
/*-----------------------------------------------------------*/
float* Descriptor::get_HOG_Dalal_for_image(const char* image_file, int *size_des)
{
    float *descriptor=NULL;
    cv::Mat image=cv::imread(std::string(image_file));
    descriptor=get_SHELO_2_for_image(image, size_des);
    return descriptor;
}



