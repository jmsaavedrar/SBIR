/*
 * JDistance.cpp
 *
 *  Created on: Feb 2, 2015
 *  Author: José M. Saavedra
 *  Copyright Orand S.A.
 */

#include "JDistance.h"
#include <cmath>

JDistance::JDistance ()
{
}
//a generic distance using L2
float JDistance::getDistance(const float *v, const float *u, unsigned int size)
{
	float result=0;
	unsigned int last_group=size-3;//to ensure the last group has four members
	float diff_0 = 0, diff_1 = 0, diff_2 = 0, diff_3 = 0;
	unsigned int i = 0;
	while (i < last_group){
		diff_0 = v[i] - u[i];
		diff_1 = v[i + 1] - u[i + 1];
		diff_2 = v[i + 2] - u[i + 2];
		diff_3 = v[i + 3] - u[i + 3];
		result += diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3;
		i += 4;
	}
	while (i < size){
		diff_0 = v[i] - u[i];
		result += diff_0 * diff_0;
		i++;
	}
	return result; //sqrt?
}
JDistance::~JDistance ()
{
}
/*------------------------------------------------------------*/
JL1::JL1()
{
}
JL1::~JL1()
{
}
/*------------------------------------------------------------*/
float JL1::getDistance (const float *v, const float *u, unsigned int size)
{
  float result=0;
  unsigned int last_group=size-3;//to ensure the last group has four members
  float diff_0 = 0, diff_1 = 0, diff_2 = 0, diff_3 = 0;
  unsigned int i = 0;
  while (i < last_group)
    {
      diff_0 = std::abs(v[i] - u[i]);
      diff_1 = std::abs(v[i + 1] - u[i + 1]);
      diff_2 = std::abs(v[i + 2] - u[i + 2]);
      diff_3 = std::abs(v[i + 3] - u[i + 3]);
      result += diff_0  + diff_1  + diff_2 + diff_3;
      i += 4;
    }
  while (i < size)
    {
      diff_0 = std::abs(v[i] - u[i]);
      result += diff_0 ;
      i++;
    }
  return result;
}
/*------------------------------------------------------------*/
JL2::JL2()
{
}
JL2::~JL2()
{
}
/*------------------------------------------------------------*/
float JL2::getDistance (const float *v, const float *u, unsigned int size)
{
  float result=0;
  unsigned int last_group=size-3;//to ensure the last group has four members
  float diff_0 = 0, diff_1 = 0, diff_2 = 0, diff_3 = 0;
  unsigned int i = 0;
  while (i < last_group)
    {
      diff_0 = v[i] - u[i];
      diff_1 = v[i + 1] - u[i + 1];
      diff_2 = v[i + 2] - u[i + 2];
      diff_3 = v[i + 3] - u[i + 3];
      result += diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3;
      i += 4;
    }
  while (i < size)
    {
      diff_0 = v[i] - u[i];
      result += diff_0 * diff_0;
      i++;
    }
  return result; //sqrt??
}
/*------------------------------------------------------------*/
JHellinger::JHellinger()
{
}
JHellinger::~JHellinger()
{
}
/*------------------------------------------------------------*/
float JHellinger::getDistance (const float *v, const float *u, unsigned int size)
{
    float result=0;
    unsigned int last_group=size-3;//to ensure the last group has four members
    float diff_0 = 0, diff_1 = 0, diff_2 = 0, diff_3 = 0;
    unsigned int i = 0;
    while (i < last_group)
      {
        diff_0 = std::sqrt(v[i]) - std::sqrt(u[i]);
        diff_1 = std::sqrt(v[i + 1]) - std::sqrt(u[i + 1]);
        diff_2 = std::sqrt(v[i + 2]) - std::sqrt(u[i + 2]);
        diff_3 = std::sqrt(v[i + 3]) - std::sqrt(u[i + 3]);
        result += diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3;
        i += 4;
      }
    while (i < size)
      {
        diff_0 = std::sqrt(v[i]) - std::sqrt(u[i]);
        result += diff_0 * diff_0;
        i++;
      }
    return result;
}
/*------------------------------------------------------------*/
JChiSquare::JChiSquare()
{
}
JChiSquare::~JChiSquare()
{
}
/*------------------------------------------------------------*/
float JChiSquare::getDistance(const float *v, const float *u, unsigned int size)
{
  float result=0;
  float diff=0, sum=0;
  unsigned int i = 0;
  while (i < size)
    {
      sum=v[i]+u[i];
      sum=(sum==0)?1:sum;
      diff = v[i]-u[i];
      result+=(diff*diff)/sum;
      i++;
    }
  return result;
}
