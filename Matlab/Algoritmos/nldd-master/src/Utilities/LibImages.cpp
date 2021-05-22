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
 * @file LibImages.cpp
 * @brief Usefull functions on images
 *
 * @author Marc Lebrun <marc.lebrun.ik@gmail.com>
 **/

#include "LibImages.h"
#include "Utilities.h"
#include "mt19937ar.h"

#include <iostream>
#include <sstream>
#include <stdlib.h>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#ifdef _MSC_VER
#include <time.h>
#include <process.h>
#else
#include <unistd.h>
#endif

 using namespace std;



//! add noise to img.
void addNoise(
  std::vector<float> const& i_im,
  std::vector<float> &o_imNoisy,
  const float p_sigma,
  const bool p_verbose){

  if (p_verbose) {
  cout << "Add noise [sigma = " << p_sigma << "] ...";
  }

  //! Initialization
  o_imNoisy = i_im;
  mt_init_genrand((unsigned long int) time (NULL) + (unsigned long int) getpid());

  //! Add noise
  for (unsigned int k = 0; k < i_im.size(); k++) {
    const double a = mt_genrand_res53();
    const double b = mt_genrand_res53();

    o_imNoisy[k] += p_sigma * (float) (sqrtl(-2.0l * log(a)) * cos(2.0l * M_PI * b));
  }

  if (p_verbose) {
    cout << "done." << endl;
  }
}


//! Compute PSNR and RMSE between i_im1 and i_im2.
int computePsnr(
  std::vector<float> const& i_im1,
  std::vector<float> const& i_im2,
  float &o_psnr,
  float &o_rmse,
  const char* p_imageName,
  const bool p_verbose){

  if (i_im1.size() != i_im2.size()) {
    cout << "Can't compute PSNR & RMSE: images have different sizes: " << endl;
    cout << "i_im1 : " << i_im1.size() << endl;
    cout << "i_im2 : " << i_im2.size() << endl;
    return EXIT_FAILURE;
  }

  float sum = 0.f;
  for (unsigned int k = 0; k < i_im1.size(); k++) {
    sum += (i_im1[k] - i_im2[k]) * (i_im1[k] - i_im2[k]);
  }

  o_rmse = sqrtf(sum / (float) i_im1.size());
  o_psnr = 20.f * log10f(255.f / o_rmse);

  if (p_verbose) {
    cout << p_imageName << endl;
    cout << "PSNR = " << o_psnr << endl;
    cout << "RMSE = " << o_rmse << endl;
  }

  return EXIT_SUCCESS;
}


//! Transform the color space of an image, from RGB to YUV, or vice-versa.
void transformColorSpace(
  std::vector<float> &io_im,
  const ImageSize &p_imSize,
  const bool p_isForward){

  //! If the image as only one channel, do nothing
  if (p_imSize.nChannels == 1) {
    return;
  }

  //! Initialization
  vector<float> imTmp(p_imSize.whc);

  //! Only 3 channels
  if (p_imSize.nChannels == 3) {
    //! Initializations
    float* iTr       = &imTmp[0];
    const float* iIr = &io_im[0];
    float* iTg       = &imTmp[p_imSize.wh];
    const float* iIg = &io_im[p_imSize.wh];
    float* iTb       = &imTmp[p_imSize.wh * 2];
    const float* iIb = &io_im[p_imSize.wh * 2];

    //! RGB to YUV
    if (p_isForward) {
      const float a = 1.f / sqrtf(3.f);
      const float b = 1.f / sqrtf(2.f);
      const float c = 2.f * a * sqrtf(2.f);

      for (unsigned int k = 0; k < p_imSize.wh; k++) {
        iTr[k] = a * (iIr[k] + iIg[k] + iIb[k]);
        iTg[k] = b * (iIr[k] - iIb[k]);
        iTb[k] = c * (0.25f * iIr[k] - 0.5f * iIg[k] + 0.25f * iIb[k]);
      }
    }

    //! YUV to RGB
    else {
      const float a = 1.f / sqrtf(3.f);
      const float b = 1.f / sqrtf(2.f);
      const float c = a / b;

      for (unsigned int k = 0; k < p_imSize.wh; k++) {
        iTr[k] = a * iIr[k] + b * iIg[k] + c * 0.5f * iIb[k];
        iTg[k] = a * iIr[k] - c * iIb[k];
        iTb[k] = a * iIr[k] - b * iIg[k] + c * 0.5f * iIb[k];
      }
    }
  }

  //! 4 channels
  else {
    //! Initializations
    float* iTgr       = &imTmp[0];
    const float* iIgr = &io_im[0];
    float* iTr        = &imTmp[p_imSize.wh];
    const float* iIr  = &io_im[p_imSize.wh];
    float* iTb        = &imTmp[p_imSize.wh * 2];
    const float* iIb  = &io_im[p_imSize.wh * 2];
    float* iTgb       = &imTmp[p_imSize.wh * 3];
    const float* iIgb = &io_im[p_imSize.wh * 3];

    //! RGB to YUV
    if (p_isForward) {
      const float a = 0.5f;
      const float b = 1.f / sqrtf(2.f);

      for (unsigned int k = 0; k < p_imSize.wh; k++) {
        iTgr[k] = a * (iIgr[k] + iIr[k] + iIb[k] + iIgb[k]);
        iTr [k] = b * (iIr[k] - iIb[k]);
        iTb [k] = a * (-iIgr[k] + iIr[k] + iIb[k] - iIgb[k]);
        iTgb[k] = b * (-iIgr[k] + iIgb[k]);
      }
    }

    //! YUV to RGB
    else {
      const float a = 0.5f;
      const float b = 1.f / sqrtf(2.f);

      for (unsigned int k = 0; k < p_imSize.wh; k++) {
        iTgr[k] = a * iIgr[k] - a * iIb[k] - b * iIgb[k];
        iTr [k] = a * iIgr[k] + b * iIr[k] + a * iIb [k];
        iTb [k] = a * iIgr[k] - b * iIr[k] + a * iIb [k];
        iTgb[k] = a * iIgr[k] - a * iIb[k] + b * iIgb[k];
      }
    }
  }

  io_im = imTmp;
}


//! Transform the color space of an image, from RGB to YUV, or vice-versa.
void transformColorSpaceDDID(
  std::vector<float> &io_im,
  const ImageSize &p_imSize,
  const bool p_isForward){

  //! The image must have 3 channels
  if (p_imSize.nChannels == 1) {
    return;
  }

  //! Initialization
  vector<float> imTmp(p_imSize.whc);
  vector<float> M;
  getDctMatrix(M);

  //! Initializations
  float* iTr       = &imTmp[0];
  const float* iIr = &io_im[0];
  float* iTg       = &imTmp[p_imSize.wh];
  const float* iIg = &io_im[p_imSize.wh];
  float* iTb       = &imTmp[p_imSize.wh * 2];
  const float* iIb = &io_im[p_imSize.wh * 2];

  //! RGB to YUV
  if (p_isForward) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < p_imSize.wh; k++) {
      iTr[k] = M[0 * 3 + 0] * iIr[k] + M[1 * 3 + 0] * iIg[k] + M[2 * 3 + 0] * iIb[k];
      iTg[k] = M[0 * 3 + 1] * iIr[k] + M[1 * 3 + 1] * iIg[k] + M[2 * 3 + 1] * iIb[k];
      iTb[k] = M[0 * 3 + 2] * iIr[k] + M[1 * 3 + 2] * iIg[k] + M[2 * 3 + 2] * iIb[k];
    }
  }

  //! YUV to RGB
  else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < p_imSize.wh; k++) {
      iTr[k] = M[0 * 3 + 0] * iIr[k] + M[0 * 3 + 1] * iIg[k] + M[0 * 3 + 2] * iIb[k];
      iTg[k] = M[1 * 3 + 0] * iIr[k] + M[1 * 3 + 1] * iIg[k] + M[1 * 3 + 2] * iIb[k];
      iTb[k] = M[2 * 3 + 0] * iIr[k] + M[2 * 3 + 1] * iIg[k] + M[2 * 3 + 2] * iIb[k];
    }
  }

  io_im = imTmp;
}


//! Add boundaries by symetry around an image to increase its size until a multiple of p_N.
int enhanceBoundaries(
  std::vector<float> const& i_im,
  std::vector<float> &o_imSym,
  const ImageSize &p_imSize,
  ImageSize &o_imSizeSym,
  const unsigned p_N){

  //! Check the size of the input image
  if (i_im.size() != p_imSize.whc) {
    cout << "Sizes are not consistent !!!" << endl;
    return EXIT_FAILURE;
  }

  //! Initialization
  unsigned int dh = 0;
  unsigned int dw = 0;

  //! Obtain size of boundaries
  if (p_imSize.height % p_N != 0) {
    dh = p_N - p_imSize.height % p_N;
  }
  if (p_imSize.width % p_N != 0) {
    dw = p_N - p_imSize.width % p_N;
  }

  //! Update size of o_imSym
  o_imSizeSym.height    = p_imSize.height + dh;
  o_imSizeSym.width     = p_imSize.width  + dw;
  o_imSizeSym.nChannels = p_imSize.nChannels;
  o_imSizeSym.wh        = o_imSizeSym.width * o_imSizeSym.height;
  o_imSizeSym.whc       = o_imSizeSym.wh * o_imSizeSym.nChannels;
  o_imSym.resize(o_imSizeSym.whc);

  //! Particular case
  if (dh == 0 && dw == 0) {
    o_imSym = i_im;
    return EXIT_SUCCESS;
  }

  //! Determine Left, Right, Up and Down boundaries
  const unsigned int dhU = dh / 2;
  const unsigned int dhD = dh - dhU;
  const unsigned int dwL = dw / 2;
  const unsigned int dwR = dw - dwL;

  for (unsigned int c = 0; c < p_imSize.nChannels; c++) {

    //! For convenience
    const unsigned int wi  = p_imSize.width;
    const unsigned int ws  = o_imSizeSym.width;
    const unsigned int dci = c * p_imSize.wh;
    const unsigned int dcs = c * o_imSizeSym.wh;

    //! Center of the image
    for (unsigned int i = 0; i < p_imSize.height; i++) {
      float* oS       = &o_imSym[dcs + (i + dhU) * ws + dwL];
      const float* iI = &i_im   [dci + i * wi];

      for (unsigned int j = 0; j < p_imSize.width; j++) {
        oS[j] = iI[j];
      }
    }

    //! Up
    for (unsigned int i = 0; i < dhU; i++) {
      const float* iI = &i_im   [dci + (dhU - i) * wi];
      float* oS       = &o_imSym[dcs + i * ws + dwL];

      for (unsigned int j = 0; j < p_imSize.width; j++) {
        oS[j] = iI[j];
      }
    }

    //! Down
    for (unsigned int i = 0; i < dhD; i++) {
      const float *iI = &i_im   [dci + (p_imSize.height - i - 2) * wi];
      float* oS       = &o_imSym[dcs + (p_imSize.height + i + dhU) * ws + dwL];

      for (unsigned int j = 0; j < p_imSize.width; j++) {
        oS[j] = iI[j];
      }
    }

    //! Left
    for (unsigned int i = 0; i < o_imSizeSym.height; i++) {
      float* oS = &o_imSym[dcs + i * ws];

      for (int j = 0; j < (int) dwL; j++) {
        oS[j] = oS[2 * dwL - j];
      }
    }

    //! Right
    for (unsigned int i = 0; i < o_imSizeSym.height; i++) {
      float* oS = &o_imSym[dcs + i * ws + dwL];

      for (int j = 0; j < (int) dwR; j++) {
        oS[j + wi] = oS[wi - j - 2];
      }
    }
  }

  return EXIT_SUCCESS;
}


//! Remove boundaries added with enhanceBoundaries
int removeBoundaries(
  std::vector<float> &o_im,
  std::vector<float> const& i_imSym,
  const ImageSize &p_imSize,
  const ImageSize &p_imSizeSym){

  //! Check if sizes are consistent
  if (i_imSym.size() != p_imSizeSym.whc) {
    cout << "Sizes are not consistent !!! " << endl;
    return EXIT_FAILURE;
  }
  if (p_imSize.width > p_imSizeSym.width || p_imSize.height > p_imSizeSym.height
    || p_imSize.nChannels != p_imSizeSym.nChannels) {
    cout << "Inner image is too big" << endl;
    return EXIT_FAILURE;
  }
  if (o_im.size() != p_imSize.whc) {
    o_im.resize(p_imSize.whc);
  }

  //! Obtain size of boundaries
  const unsigned int dhU = (p_imSizeSym.height - p_imSize.height) / 2;
  const unsigned int dwL = (p_imSizeSym.width  - p_imSize.width ) / 2;

  //! Get the inner image
  for (unsigned int c = 0; c < p_imSize.nChannels; c++) {
    const float* iS = &i_imSym[c * p_imSizeSym.wh + dhU * p_imSizeSym.width + dwL];
    float* oI       = &o_im   [c * p_imSize.wh];

    for (unsigned int i = 0; i < p_imSize.height; i++) {
      for (unsigned int j = 0; j < p_imSize.width; j++) {
        oI[i * p_imSize.width + j] = iS[i * p_imSizeSym.width + j];
      }
    }
  }

  return EXIT_SUCCESS;
}


//! Add boundaries around image to avoid boundary problems for the big loop over pixels.
void setBoundaries(
  std::vector<float> const& i_im,
  std::vector<float> &o_im,
  const ImageSize &i_imSize,
  ImageSize &o_imSize,
  const unsigned int p_border){

  //! Get the new size
  o_imSize.width     = i_imSize.width + 2 * p_border;
  o_imSize.height    = i_imSize.height + 2 * p_border;
  o_imSize.nChannels = i_imSize.nChannels;
  o_imSize.wh        = o_imSize.width * o_imSize.height;
  o_imSize.whc       = o_imSize.wh * o_imSize.nChannels;
  o_im.assign(o_imSize.whc, 0.f);

  //! For convenience
  const unsigned wI = i_imSize.width;
  const unsigned wO = o_imSize.width;
  const unsigned hO = o_imSize.height;

  //! Get the new (bigger) image
  for (unsigned c = 0; c < i_imSize.nChannels; c++) {
    const float* iI = &i_im[c * i_imSize.wh];

    //! Inner image
    float* oI = &o_im[c * o_imSize.wh + p_border * wO + p_border];
    for (unsigned i = 0; i < i_imSize.height; i++) {
      for (unsigned j = 0; j < i_imSize.width; j++) {
        oI[i * wO + j] = iI[i * wI + j];
      }
    }

    //! Left and right
    oI = &o_im[c * o_imSize.wh + p_border * wO];
    for (unsigned i = 0; i < i_imSize.height; i++) {
      for (unsigned j = 0; j < p_border; j++) {
        oI[i * wO + j] = iI[i * wI + p_border - j];
        oI[i * wO + wO - p_border + j] = iI[i * wI + wI - j - 2];
      }
    }

    //! Top and bottom
    oI = &o_im[c * o_imSize.wh];
    for (unsigned i = 0; i < p_border; i++) {
      for (unsigned j = 0; j < o_imSize.width; j++) {
        oI[i * wO + j] = oI[(2 * p_border - i) * wO + j];
        oI[(i + hO - p_border) * wO + j] = oI[(hO - p_border - i - 1) * wO + j];
      }
    }
  }
}


//! Divide an image into sub-parts of equal sizes with a boundaries around each.
int subDivide(
  std::vector<float> const& i_im,
  std::vector<std::vector<float> > &o_imSub,
  const ImageSize &i_imSize,
  ImageSize &o_imSizeSub,
  const unsigned int p_boundaries,
  const unsigned int p_nbParts){

  //! Add boundaries around the image
  unsigned int a, b;
  getDivisors(p_nbParts, a, b);
  const unsigned int d = getPpcm(a, b);
  ImageSize imSizeBig;
  vector<float> imBig;
  if (enhanceBoundaries(i_im, imBig, i_imSize, imSizeBig, d) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  //! Get the size of sub-images
  vector<float> imNew;
  ImageSize imSizeNew;
  setBoundaries(imBig, imNew, imSizeBig, imSizeNew, p_boundaries);
  o_imSizeSub.height    = imSizeBig.height / a + 2 * p_boundaries;
  o_imSizeSub.width     = imSizeBig.width  / b + 2 * p_boundaries;
  o_imSizeSub.nChannels = imSizeBig.nChannels;
  o_imSizeSub.wh        = o_imSizeSub.width * o_imSizeSub.height;
  o_imSizeSub.whc       = o_imSizeSub.wh * o_imSizeSub.nChannels;
  o_imSub.resize(p_nbParts);
  for (unsigned int n = 0; n < p_nbParts; n++) {
    o_imSub[n].resize(o_imSizeSub.whc);
  }

  //! For convenience
  const unsigned int ws = o_imSizeSub.width;
  const unsigned int wn = imSizeNew.width;

  //! Get all sub-images with the boundaries
  for (unsigned int c = 0; c < i_imSize.nChannels; c++) {
    for (unsigned int ni = 0; ni < a; ni++) {
      for (unsigned int nj = 0; nj < b; nj++) {
        const unsigned int di = ni * (o_imSizeSub.height - 2 * p_boundaries);
        const unsigned int dj = nj * (o_imSizeSub.width  - 2 * p_boundaries);
        float* oS = &o_imSub[ni * b + nj][c * o_imSizeSub.wh];
        const float* iN = &imNew[c * imSizeNew.wh + di * wn + dj];

        for (unsigned int i = 0; i < o_imSizeSub.height; i++) {
          for (unsigned int j = 0; j < o_imSizeSub.width; j++) {
            oS[i * ws + j] = iN[i * wn + j];
          }
        }
      }
    }
  }

  return EXIT_SUCCESS;
}


//! Build an image from a set of sub-parts obtained with subDivide.
int subBuild(
  std::vector<std::vector<float> > const& i_imSub,
  std::vector<float> &o_im,
  const ImageSize &p_imSizeSub,
  const ImageSize &p_imSize,
  const unsigned int p_boundaries){

  //! Initializations
  const unsigned int nbParts = i_imSub.size();
  unsigned int a, b;
  getDivisors(nbParts, a, b);
  const unsigned int hs = p_imSizeSub.height;
  const unsigned int ws = p_imSizeSub.width;
  const unsigned int hn = (hs - 2 * p_boundaries) * a;
  const unsigned int wn = (ws - 2 * p_boundaries) * b;
  ImageSize imSizeNew   = {wn, hn, p_imSize.nChannels, wn * hn, wn * hn * p_imSize.nChannels};
  o_im.resize(p_imSize.whc);
  vector<float> imNew(imSizeNew.whc);

  //! Get the image
  for (unsigned int c = 0; c < p_imSize.nChannels; c++) {
    for (unsigned int ni = 0; ni < a; ni++) {
      for (unsigned int nj = 0; nj < b; nj++) {
        const unsigned int di = ni * (hs - 2 * p_boundaries);
        const unsigned int dj = nj * (ws - 2 * p_boundaries);
        const float* iS = &i_imSub[ni * b + nj][c * p_imSizeSub.wh + p_boundaries * ws + p_boundaries];
        float* iN = &imNew[c * imSizeNew.wh + di * wn + dj];

        for (unsigned int i = 0; i < hs - 2 * p_boundaries; i++) {
          for (unsigned int j = 0; j < ws - 2 * p_boundaries; j++) {
            iN[i * wn + j] = iS[i * ws + j];
          }
        }
      }
    }
  }

  //! Extract image
  if (removeBoundaries(o_im, imNew, p_imSize, imSizeNew) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


