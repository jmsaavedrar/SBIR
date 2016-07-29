/*
morphological.cpp includes morphological operations
This version just includes two thinning operations

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
#include "morphological.h"

#include <queue>
#include <vector>
#include <iostream>

Morphological::Morphological()
{
}
cv::Mat Morphological::thinning2(cv::Mat input, int iter, int* done_it)//it is the best
{
    /*input: a binary image CV_8UC1*/
    int i=0,j=0,it=0,cont=1;
    it=1;
    std::queue<cv::Point2i> deletedPoints;//=new std::queue<cv::Point2i*>();
    cv::Point2i point;
    int height=0;
    int width=0;
    height=input.size().height;
    width=input.size().width;
    int val_pixel=0;
    cv::Mat output=input.clone();
    //cout<<output<<endl;
    int _done_it=0;
    while((it<=iter)&&(cont>0))
    {
        cont=0;
        _done_it++;
        //First subiteration
        for(i=1;i<height-1;i++)
        {
            for(j=1;j<width-1;j++)
            {
                val_pixel=output.at<uchar>(i,j);
                if(val_pixel==1)//if is a foreground pixel //even
                {
                    if(evaluateGuoHall(output, i,j,1))//deleting condition
                    {
                        deletedPoints.push(cv::Point2i(j,i));
                        cont++;
                    }
                }
            }
        }
        //update buffer
        while(!(deletedPoints.empty()))
        {
            point=(cv::Point2i)deletedPoints.front();
            output.at<uchar>(point.y, point.x)=0;
            deletedPoints.pop();
        }
        cont=0;
        //Second subiteration
        for(i=1;i<height-1;i++)
        {
            for(j=1;j<width-1;j++)
            {
                val_pixel=output.at<uchar>(i,j);
                if(val_pixel==1)//if is a foreground pixel //even
                {
                    if(evaluateGuoHall(output, i,j,2))//deleting condition
                    {
                        deletedPoints.push(cv::Point2i(j,i));
                        cont++;
                    }
                }
            }
        }
        if(done_it!=nullptr) *done_it=_done_it;
        //update buffer
        while(!(deletedPoints.empty()))
        {
            point=(cv::Point2i)deletedPoints.front();
            output.at<uchar>(point.y, point.x)=0;
            deletedPoints.pop();
        }

        it++;//another iteration
    }//end while
    return output;
}

/*----------------------Evaluate the condition given by Guo and Hall for deleting a pixel THINNING--------------------*/
bool Morphological::evaluateGuoHall(cv::Mat input, int ib, int jb, int iter)//For thining2
{
    int *neighbors=new int[8];
    //int width=input.size().width;
    //uchar *buffer=input.data;
    /*neighbors[0]=(int)buffer[(ib)*width+jb+1];//top left corner
    neighbors[1]=(int)buffer[(ib-1)*width+jb+1];//clock-wise order
    neighbors[2]=(int)buffer[(ib-1)*width+jb];//
    neighbors[3]=(int)buffer[(ib-1)*width+jb-1];
    neighbors[4]=(int)buffer[(ib)*width+jb-1];//
    neighbors[5]=(int)buffer[(ib+1)*width+jb-1];
    neighbors[6]=(int)buffer[(ib+1)*width+jb];//
    neighbors[7]=(int)buffer[(ib+1)*width+jb+1];
    */
    neighbors[0]=(int)input.at<uchar>(ib,jb+1);
    neighbors[1]=(int)input.at<uchar>(ib-1,jb+1);
    neighbors[2]=(int)input.at<uchar>(ib-1,jb);
    neighbors[3]=(int)input.at<uchar>(ib-1,jb-1);
    neighbors[4]=(int)input.at<uchar>(ib,jb-1);
    neighbors[5]=(int)input.at<uchar>(ib+1,jb-1);
    neighbors[6]=(int)input.at<uchar>(ib+1,jb);
    neighbors[7]=(int)input.at<uchar>(ib+1,jb+1);

    int x=0,n1=0,n2=0;
    bool g1=false,g2=false,g3=false,g3p=false;
    bool rsp=false;
    x=0;
    n1=0;
    n2=0;
    x+=((!neighbors[0])&(neighbors[1]|neighbors[2]));
    x+=((!neighbors[2])&(neighbors[3]|neighbors[4]));
    x+=((!neighbors[4])&(neighbors[5]|neighbors[6]));
    x+=((!neighbors[6])&(neighbors[7]|neighbors[0]));
    n1+=(neighbors[0]|neighbors[1]);
    n1+=(neighbors[2]|neighbors[3]);
    n1+=(neighbors[4]|neighbors[5]);
    n1+=(neighbors[6]|neighbors[7]);
    n2+=(neighbors[1]|neighbors[2]);
    n2+=(neighbors[3]|neighbors[4]);
    n2+=(neighbors[5]|neighbors[6]);
    n2+=(neighbors[7]|neighbors[0]);
    int np=std::min(n1,n2);
    g1=(x==1);
    g2=((2<=np)&&(np<=3));
    if(iter==1)
    {
        g3=(((neighbors[1]|neighbors[2]|(!neighbors[7]))&neighbors[0])==0);
        rsp=(g1 && g2 && g3);
    }
    if(iter==2)
    {
        g3p=(((neighbors[5]|neighbors[6]|(!neighbors[3]))&neighbors[4])==0);
        rsp=(g1 && g2 && g3p);
    }
    delete[] neighbors;
    return rsp;
}
/*-------------------------------------------------------------------------------*/
void Morphological::thinSubIteration1(cv::Mat &pSrc, cv::Mat &pDst)
{
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    pSrc.copyTo(pDst);
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(pSrc.at<uchar>(i, j) == 1)
            {
                /// get 8 neighbors
                /// calculate C(p)
                int neighbor0 = (int) pSrc.at<uchar>( i-1, j-1);
                int neighbor1 = (int) pSrc.at<uchar>( i-1, j);
                int neighbor2 = (int) pSrc.at<uchar>( i-1, j+1);
                int neighbor3 = (int) pSrc.at<uchar>( i, j+1);
                int neighbor4 = (int) pSrc.at<uchar>( i+1, j+1);
                int neighbor5 = (int) pSrc.at<uchar>( i+1, j);
                int neighbor6 = (int) pSrc.at<uchar>( i+1, j-1);
                int neighbor7 = (int) pSrc.at<uchar>( i, j-1);
                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                        int(~neighbor3 & ( neighbor4 | neighbor5)) +
                        int(~neighbor5 & ( neighbor6 | neighbor7)) +
                        int(~neighbor7 & ( neighbor0 | neighbor1));
                if(C == 1)
                {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                            int(neighbor2 | neighbor3) +
                            int(neighbor4 | neighbor5) +
                            int(neighbor6 | neighbor7);

                    int N2 = int(neighbor1 | neighbor2) +
                            int(neighbor3 | neighbor4) +
                            int(neighbor5 | neighbor6) +
                            int(neighbor7 | neighbor0);
                    int N = std::min(N1,N2);
                    if ((N == 2) || (N == 3))
                    {
                        /// calculate criteria 3
                        int c3 = ( neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
                        if(c3 == 0)
                        {
                            pDst.at<uchar>(i,j) = 0.0;
                        }
                    }
                }
            }
        }
    }
}
/*-------------------------------------------------------------------------------*/
void Morphological::thinSubIteration2(cv::Mat &pSrc, cv::Mat &pDst)
{
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    pSrc.copyTo( pDst);
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if (pSrc.at<uchar>( i, j) == 1)
            {
                /// get 8 neighbors
                /// calculate C(p)
                int neighbor0 = (int) pSrc.at<uchar>( i-1, j-1);
                int neighbor1 = (int) pSrc.at<uchar>( i-1, j);
                int neighbor2 = (int) pSrc.at<uchar>( i-1, j+1);
                int neighbor3 = (int) pSrc.at<uchar>( i, j+1);
                int neighbor4 = (int) pSrc.at<uchar>( i+1, j+1);
                int neighbor5 = (int) pSrc.at<uchar>( i+1, j);
                int neighbor6 = (int) pSrc.at<uchar>( i+1, j-1);
                int neighbor7 = (int) pSrc.at<uchar>( i, j-1);
                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                        int(~neighbor3 & ( neighbor4 | neighbor5)) +
                        int(~neighbor5 & ( neighbor6 | neighbor7)) +
                        int(~neighbor7 & ( neighbor0 | neighbor1));
                if(C == 1)
                {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                            int(neighbor2 | neighbor3) +
                            int(neighbor4 | neighbor5) +
                            int(neighbor6 | neighbor7);
                    int N2 = int(neighbor1 | neighbor2) +
                            int(neighbor3 | neighbor4) +
                            int(neighbor5 | neighbor6) +
                            int(neighbor7 | neighbor0);
                    int N = std::min(N1,N2);
                    if((N == 2) || (N == 3))
                    {
                        int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
                        if(E == 0)
                        {
                            pDst.at<uchar>(i, j) = 0;
                        }
                    }
                }
            }
        }
    }
}
/*-------------------------------------------------------------------------------*/
cv::Mat Morphological::thinning_Zhang_Sue(cv::Mat input, int* done_it)
{
    bool bDone = false;
    int rows = input.rows;
    int cols = input.cols;

    cv::Mat output;
    input.copyTo(output);
    cv::Mat p_enlarged_src = cv::Mat(rows + 2, cols + 2, CV_8UC1);

    for(int i = 0; i < (rows+2); i++)
    {
        p_enlarged_src.at<uchar>(i, 0) = 0;
        p_enlarged_src.at<uchar>( i, cols+1) = 0;
    }
    for(int j = 0; j < (cols+2); j++)
    {
        p_enlarged_src.at<uchar>(0, j) = 0;
        p_enlarged_src.at<uchar>(rows+1, j) = 0;
    }
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            p_enlarged_src.at<uchar>( i+1, j+1) = input.at<uchar>(i,j);
        }
    }

    /// start to thin
    cv::Mat p_thinMat1 = cv::Mat::zeros(rows + 2, cols + 2, CV_8UC1);
    cv::Mat p_thinMat2 = cv::Mat::zeros(rows + 2, cols + 2, CV_8UC1);
    cv::Mat p_cmp = cv::Mat::zeros(rows + 2, cols + 2, CV_8UC1);
    int _done_it=0;
    while (bDone != true)
    {
        /// sub-iteration 1

        thinSubIteration1(p_enlarged_src, p_thinMat1);
        /// sub-iteration 2

        thinSubIteration2(p_thinMat1, p_thinMat2);
        /// compare
        compare(p_enlarged_src, p_thinMat2, p_cmp,cv::CMP_EQ);
        /// check
        int num_non_zero = countNonZero(p_cmp);
        if(num_non_zero == (rows + 2) * (cols + 2))
        {
            bDone = true;
        }
        /// copy
        p_thinMat2.copyTo(p_enlarged_src);
        _done_it++;
    }
    if(done_it!=nullptr) *done_it=_done_it;
    // copy result
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            output.at<uchar>(i,j) = p_enlarged_src.at<uchar>(i+1, j+1);

        }
    }
    return output;
}

