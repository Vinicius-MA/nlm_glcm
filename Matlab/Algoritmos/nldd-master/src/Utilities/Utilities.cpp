/*
 * Copyright (c) 2013, Marc Lebrun <marc.lebrun.ik@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file Utilities.cpp
 * @brief Set of usefull functions.
 *
 * @author Marc Lebrun <marc.lebrun.ik@gmail.com>
 **/

#include "Utilities.h"

#include <stdlib.h>
#include <iostream>
#ifdef  _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <algorithm>

using namespace std;


//! Convenient function to use the sort function provided by the vector library.
bool comparaisonFirst(
  const pair<float, unsigned> &i_pair1,
  const pair<float, unsigned> &i_pair2){
  return i_pair1.first < i_pair2.first;
}


//! Obtain and substract the baricenter of io_group3d.
void centerData(
  std::vector<float> &io_group3d,
  std::vector<float> &o_baricenter,
  const unsigned p_rows,
  const unsigned p_cols){

  const float inv = 1.f / (float) p_rows;
  for (unsigned j = 0; j < p_cols; j++) {
    float* iG = &io_group3d[j * p_rows];
    float sum = 0.f;
    float* iB = &o_baricenter[j];

    for (unsigned i = 0; i < p_rows; i++) {
      sum += iG[i];
    }
    *iB = sum * inv;

    for (unsigned i = 0; i < p_rows; i++) {
      iG[i] -= *iB;
    }
  }
}


//! Get (a,b) such that n = a * b, with a and b maximum, a < b
void getDivisors(
  const unsigned int i_n,
  unsigned int &o_a,
  unsigned int &o_b){

  o_a = (unsigned int) sqrtf(float(i_n));
  while (i_n % o_a != 0) {
    o_a--;
  }
  o_b = i_n / o_a;

  if (o_b < o_a) {
    int c = o_b;
    o_b = o_a;
    o_a = c;
  }
}


//! Return the PPCM of (a, b).
unsigned int getPpcm(
  const unsigned int i_a,
  const unsigned int i_b){

  if (i_a == 0 || i_b == 0) {
    return 0;
  }

  int a = i_a;
  int b = i_b;
  while (a != b) {
    if (a < b) {
      a += i_a;
    }
    else {
      b += i_b;
    }
  }

  return a;
}


//! Computes the 3-by-3 DCT transform matrix.  D*A
//! is the DCT of the columns of A and D'*A is the inverse DCT of
//! the columns of A (when A is 3-by-3).
void getDctMatrix(
  std::vector<float> &o_mat){

  //! Initialization
  const unsigned int N = 3;
  const float norm = sqrtf(2.f / 3.f);
  o_mat.resize(N * N);
  o_mat[0 * N + 0] = norm * cos((M_PI / 3.f) * 0.75f);
  o_mat[1 * N + 0] = norm * cos((M_PI / 3.f) * 0.75f);
  o_mat[2 * N + 0] = norm * cos((M_PI / 3.f) * 0.75f);
  o_mat[0 * N + 1] = norm * cos((M_PI / 3.f) * 0.5f);
  o_mat[1 * N + 1] = 0.f;
  o_mat[2 * N + 1] = -norm * cos((M_PI / 3.f) * 0.5f);
  o_mat[0 * N + 2] = norm * 0.5f;
  o_mat[1 * N + 2] = -norm;
  o_mat[2 * N + 2] = norm * 0.5f;
}


//! Compute a LUT of exp(-x), with a wanted precision and maximum value.
void fillExpLut(
  std::vector<float> &o_expLut,
  const float p_precision,
  const float p_max){

  //! Size of the lut
  const unsigned int size = (unsigned int) ceil(p_max) * (unsigned int) ceil(p_precision);
  o_expLut.resize(size);

  //! Fill the LUT
  for (unsigned int x = 0; x < size; x++) {
    o_expLut[x] = expf(-(float) x / p_precision);
  }
}


//! Returns exp(-x) according to a precomputed LUT obtained with fillExpLut.
float getExp(
  std::vector<float> const& i_expLut,
  const float p_value,
  const float p_precision,
  const float p_max){

  if (p_value >= p_max) {
    return 0.f;
  }

  unsigned int x = std::min((unsigned int) floor( (double) p_value * (float) p_precision),
                            (unsigned int) i_expLut.size() - 2);
  const float y1 = i_expLut[x    ];
  const float y2 = i_expLut[x + 1];

  return y1 + (y2 - y1) * (p_value * p_precision - x);
}


/**
 * @brief Compute the average standard deviation of a set of patches.
 *
 * @param i_Set(p_sP, p_nSimP): set of patches;
 * @param p_sP : size of a patch;
 * @param p_nSimP: number of patches in the set;
 * @param p_nChannels: number of channels of the image.
 *
 * @return the average standard deviation of the set
 **/
float computeStdDeviation(
  std::vector<float> const& i_Set,
  const unsigned int p_sP,
  const unsigned int p_nSimP,
  const unsigned int p_nChannels){

  float sigma = 0.f;

  for (unsigned int c = 0; c < p_nChannels; c++) {

    //! Initialization
    float mean = 0.f;
    float std = 0.f;

    //! Compute the sum and the square sum
    for (unsigned int n = 0; n < p_nSimP; n++) {
      for (unsigned int k = 0; k < p_sP; k++) {
        const float value = i_Set[k + c * p_sP + n * p_sP * p_nChannels];
        mean += value;
        std  += value * value;
      }
    }

    //! Sample standard deviation (Bessel's correction)
    sigma += (std - mean * mean / (float) (p_sP * p_nSimP)) / (float) (p_sP * p_nSimP - 1);
  }

  return sigma / (float) p_nChannels;
}










