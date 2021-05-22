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
#ifndef LIB_IMAGES_H_INCLUDED
#define LIB_IMAGES_H_INCLUDED

#include <vector>


/**
 * @brief Structure containing size informations of an image.
 *
 * @param width     : width of the image;
 * @param height    : height of the image;
 * @param nChannels : number of channels in the image;
 * @param wh        : equal to width * height. Provided for convenience;
 * @param whc       : equal to width * height * nChannels. Provided for convenience.
 **/
struct ImageSize
{
  unsigned width;
  unsigned height;
  unsigned nChannels;
  unsigned wh;
  unsigned whc;
};


/**
 * @brief Load image, check the number of channels.
 *
 * @param p_name : name of the image to read;
 * @param o_im : vector which will contain the image : R, G and B concatenated;
 * @param o_imSize : will contain the size of the image;
 * @param p_verbose : if true, print some informations.
 *
 * @return EXIT_SUCCESS if the image has been loaded, EXIT_FAILURE otherwise.
 **/
int loadImage(
  char* p_name,
  std::vector<float> &o_im,
  ImageSize &o_imSize,
  const bool p_verbose);


/**
 * @brief write image.
 *
 * @param p_name : path+name+extension of the image;
 * @param i_im : vector which contains the image;
 * @param p_imSize : size of the image;
 * @param p_min, p_max : range of data (usually [0, 255]).
 *
 * @return EXIT_SUCCESS if the image has been saved, EXIT_FAILURE otherwise
 **/
int saveImage(
  char* p_name,
  std::vector<float> const& i_im,
  const ImageSize &p_imSize,
  const float p_min,
  const float p_max);


/**
 * @brief add noise to img.
 *
 * @param i_im : original noise-free image;
 * @param o_imNoisy = im + noise;
 * @param p_sigma : standard deviation of the noise;
 * @param p_verbose : if true, print some informations.
 *
 * @return none.
 **/
void addNoise(
  std::vector<float> const& i_im,
  std::vector<float> &o_imNoisy,
  const float p_sigma,
  const bool p_verbose);


/**
 * @brief Compute PSNR and RMSE between i_im1 and i_im2
 *
 * @param i_im1 : pointer to an allocated array of pixels;
 * @param i_im2 : pointer to an allocated array of pixels;
 * @param o_psnr  : will contain the PSNR;
 * @param o_rmse  : will contain the RMSE;
 * @param p_imageName: name of the image;
 * @param p_verbose: if true, print values of PSNR and RMSE.
 *
 * @return EXIT_FAILURE if both images haven't the same size.
 **/
int computePsnr(
  std::vector<float> const& i_im1,
  std::vector<float> const& i_im2,
  float &o_psnr,
  float &o_rmse,
  const char* p_imageName,
  const bool p_verbose);


/**
 * @brief Transform the color space of an image, from RGB to YUV, or vice-versa.
 *
 * @param io_im: image on which the transform will be applied;
 * @param p_imSize: size of io_im;
 * @param p_isForward: if true, go from RGB to YUV, otherwise go from YUV to RGB.
 *
 * @return none.
 **/
void transformColorSpace(
  std::vector<float> &io_im,
  const ImageSize &p_imSize,
  const bool p_isForward);


/**
 * @brief Transform the color space of an image, from RGB to YUV, or vice-versa.
 *
 * @param io_im: image on which the transform will be applied;
 * @param p_imSize: size of io_im;
 * @param p_isForward: if true, go from RGB to YUV, otherwise go from YUV to RGB.
 *
 * @return none.
 **/
void transformColorSpaceDDID(
  std::vector<float> &io_im,
  const ImageSize &p_imSize,
  const bool p_isForward);


/**
* @brief Add boundaries by symetry around an image to increase its size until a multiple of p_N.
*
* @param i_im : image to symetrize;
* @param o_imSym : will contain i_im with symetrized boundaries;
* @param p_imSize : size of i_im;
* @param p_imSizeSym : will contain the size of o_imSym;
* @param p_N : parameter of the boundaries.
*
* @return EXIT_FAILURE in case of problems.
**/
int enhanceBoundaries(
  std::vector<float> const& i_im,
  std::vector<float> &o_imSym,
  const ImageSize &p_imSize,
  ImageSize &o_imSizeSym,
  const unsigned p_N);


/**
* @brief Remove boundaries added with enhanceBoundaries
*
* @param o_im : will contain the inner image;
* @param i_imSym : contains i_im with symetrized boundaries;
* @param p_imSize: size of o_im;
* @param p_imSizeSym : size of i_imSym.
*
* @return none.
**/
int removeBoundaries(
  std::vector<float> &o_im,
  std::vector<float> const& i_imSym,
  const ImageSize &p_imSize,
  const ImageSize &p_imSizeSym);


/**
 * @brief Add boundaries around image to avoid boundary problems for the big loop over pixels.
 *
 * @param i_im : image to enwrap;
 * @param o_im : image which contains i_im in its center, and symetry in its borders;
 * @param i_imSize : size of i_im;
 * @param o_imSize : will contain the size of o_im;
 * @param p_border : size of the border.
 *
 * @return none.
 **/
void setBoundaries(
  std::vector<float> const& i_im,
  std::vector<float> &o_im,
  const ImageSize &i_imSize,
  ImageSize &o_imSize,
  const unsigned int p_border);


/**
 * @brief Divide an image into sub-parts of equal sizes with a boundaries around each.
 *
 * @param i_im : image to divide;
 * @param o_imSub : will contain sub-images;
 * @param i_imSize : size of i_im;
 * @param o_imSizeSub : will contain the size of o_imSub;
 * @param p_boundaries : boundaries to add around each sub-images;
 * @param p_nbParts : number of sub-images.
 *
 * @return EXIT_FAILURE in case of problems, EXIT_SUCCESS otherwise.
 **/
int subDivide(
  std::vector<float> const& i_im,
  std::vector<std::vector<float> > &o_imSub,
  const ImageSize &i_imSize,
  ImageSize &o_imSizeSub,
  const unsigned int p_boundaries,
  const unsigned int p_nbParts);


/**
 * @brief Build an image from a set of sub-parts obtained with subDivide.
 *
 * @param i_imSub : set of sub-images;
 * @param o_im : will contain the reconstructed image;
 * @param p_imSizeSub : size of sub-images;
 * @param p_imSize : size that o_im will have;
 * @param p_boundaries : boundaries that were added around each sub-images.
 *
 * @return EXIT_FAILURE in case of problems, EXIT_SUCCESS otherwise.
 **/
int subBuild(
  std::vector<std::vector<float> > const& i_imSub,
  std::vector<float> &o_im,
  const ImageSize &p_imSizeSub,
  const ImageSize &p_imSize,
  const unsigned int p_boundaries);




#endif // LIB_IMAGES_H_INCLUDED
