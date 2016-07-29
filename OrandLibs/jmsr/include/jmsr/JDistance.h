/*
 * JDistance.h
 *
 *  Created on: Feb 2, 2015
 *  Author: José M. Saavedra
 */

#ifndef JMSR_JDISTANCE_H_
#define JMSR_JDISTANCE_H_

class JDistance
{
public:
  JDistance ();
  virtual float getDistance(const float *v, const float *u, unsigned int size);
  virtual ~JDistance ();
};
class JL1 : public JDistance
{
public:
  JL1();
  ~JL1();
  float getDistance(const float *v, const float *u, unsigned int size);
};
class JL2 : public JDistance
{
public:
  JL2();
  ~JL2();
  float getDistance(const float *v, const float *u, unsigned int size);
};
//class JMinkowski : public JDistance
//{
//  JMinkowski();
//  float getDistance(float *v, float *u, unsigned int size);
//};
class JHellinger : public JDistance
{
public:
  JHellinger();
  ~JHellinger();
  float getDistance(const float *v, const float *u, unsigned int size);
};
class JChiSquare : public JDistance
{
public:
  JChiSquare();
  ~JChiSquare();
  float getDistance(const float *v, const float *u, unsigned int size);
};

#endif /* JMSR_JDISTANCE_H_ */
