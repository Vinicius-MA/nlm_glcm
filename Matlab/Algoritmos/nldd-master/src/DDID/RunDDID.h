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

#ifndef RUNDDID_H_INCLUDED
#define RUNDDID_H_INCLUDED

#include <vector>

#include "../Utilities/LibImages.h"

/**
 * @brief Stucture of parameters for DDID algorithm.
 *
 * @param sigma : standard deviation of the noise;
 * @param verbose : if true, print some information;
 * @param r : radius for the search windows;
 * @param sigmaS :
 * @param gammaR :
 * @param gammaF :
 **/
struct ddidParams{
  float sigma;
  bool verbose;
  int r;
  float sigmaS;
  float gammaR;
  float gammaF;
};

/**
 * @brief Apply the DDID denoising algorithm.
 *
 * @param i_imNoisy : input noisy image;
 * @param o_imFinal : will contains the denoised image;
 * @param p_imSize : size of i_imNoisy;
 * @param p_sigma : standard deviation of the noise to remove;
 * @param p_verbose : if true, print some information.
 *
 * @return EXIT_FAILURE in case of problems, EXIT_SUCCESS otherwise.
 **/
int runDDID(
  std::vector<float> const& i_imNoisy,
  std::vector<float> &o_imFinal,
  const ImageSize &p_imSize,
  const float p_sigma,
  const bool p_verbose);


/**
 * @brief Process the core function of DDID algorithm.
 *
 * @param i_guide : guide image;
 * @param i_im : input noisy image;
 * @param o_im : output estimated image;
 * @param p_imSize : size of images;
 * @param p_params : see ddidParams.
 *
 * @return none.
 **/
void processDDID(
  std::vector<float> const& i_guide,
  std::vector<float> const& i_im,
  std::vector<float> &o_im,
  const ImageSize &p_imSize,
  const ddidParams &p_params);

/**
 * @brief Run NlBayes to have a good guide image, and then only the third
 *       iteration of DDID.
 *
 * @param i_imNoisy : input noisy image;
 * @param o_imFinal : will contains the denoised image;
 * @param p_imSize : size of i_imNoisy;
 * @param p_sigma : standard deviation of the noise to remove;
 * @param p_verbose : if true, print some information.
 *
 * @return EXIT_FAILURE in case of problems, EXIT_SUCCESS otherwise.
 **/
int runDDIDNlBayes(
  std::vector<float> const& i_imNoisy,
  std::vector<float> &o_imFinal,
  const ImageSize &p_imSize,
  const float p_sigma,
  const bool p_verbose);


/**
 * @brief Process the core function of DDID algorithm.
 *
 * @param i_guide : guide image;
 * @param i_im : input noisy image;
 * @param o_im : output estimated image;
 * @param p_imSize : size of images;
 * @param p_params : see ddidParams.
 *
 * @return none.
 **/
void processDDID(
  std::vector<float> const& i_guide,
  std::vector<float> const& i_im,
  std::vector<float> &o_im,
  const ImageSize &p_imSize,
  const ddidParams &p_params);


#endif // RUNDDID_H_INCLUDED
