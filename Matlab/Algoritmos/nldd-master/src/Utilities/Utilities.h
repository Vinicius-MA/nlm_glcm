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

#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include <vector>


//! Convenient function to use the sort function provided by the vector library.
bool comparaisonFirst(
  const std::pair<float, unsigned> &i_pair1,
  const std::pair<float, unsigned> &i_pair2);


/**
 * @brief Obtain and substract the baricenter of io_group3d.
 *
 * @param io_group3d(p_rows x p_cols) : data to center;
 * @param o_baricenter(p_cols): will contain the baricenter of io_group3d;
 * @param p_rows, p_cols: size of io_group3d.
 *
 * @return none.
 **/
void centerData(
  std::vector<float> &io_group3d,
  std::vector<float> &o_baricenter,
  const unsigned p_rows,
  const unsigned p_cols);


/**
 * @brief Get (a,b) such that n = a * b, with a and b maximum, (a < b).
 *
 * @param i_n : number to divide;
 * @param o_a : first divisor;
 * @param o_b : second divisor.
 *
 * @return none.
 **/
void getDivisors(
  const unsigned int i_n,
  unsigned int &o_a,
  unsigned int &o_b);


/**
 * @brief Return the PPCM of (a, b).
 *
 * @param i_a : first number,
 * @param i_b : second number.
 *
 * @return PPCM(a, b).
 **/
unsigned int getPpcm(
  const unsigned int i_a,
  const unsigned int i_b);


/**
 * @brief Computes the 3-by-3 DCT transform matrix.  D*A
 *       is the DCT of the columns of A and D'*A is the inverse DCT of
 *       the columns of A (when A is 3-by-3).
 *
 * @param o_mat : will contain the 3 \times 3 DCT transform matrix.
 *
 * @return none.
 **/
void getDctMatrix(
  std::vector<float> &o_mat);


/**
 * @brief Compute a LUT of exp(-x), with a wanted precision and maximum value.
 *
 * @param o_expLut : will contain the LUT of exp(-x);
 * @param p_precision : precision of the LUT;
 * @param p_max : maximum value in the LUT.
 *
 * @return none.
 **/
void fillExpLut(
  std::vector<float> &o_expLut,
  const float p_precision,
  const float p_max);


/**
 * @brief Returns exp(-x) according to a precomputed LUT obtained with fillExpLut.
 *
 * @param i_expLut : precomputed LUT of exp(-x);
 * @param p_value : input value;
 * @param p_precision : precision of the LUT;
 * @param p_max : maximum value.
 *
 * @return exp(-p_value).
 **/
float getExp(
  std::vector<float> const& i_expLut,
  const float p_value,
  const float p_precision,
  const float p_max);


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
  const unsigned int p_nChannels);


#endif // UTILITIES_H_INCLUDED
