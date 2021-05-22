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
 * @file NlBayes.cpp
 * @brief NL-Bayes denoising functions
 *
 * @author Marc Lebrun <marc.lebrun.ik@gmail.com>
 **/

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

#include "NlBayes.h"
#include "LibMatrix.h"
#include "../Utilities/LibImages.h"
#include "../Utilities/Utilities.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

//! Initialize Parameters of the NL-Bayes algorithm.
void initializeNlbParameters(
  nlbParams &o_paramStep1,
  nlbParams &o_paramStep2,
  const float p_sigma,
  const ImageSize &p_imSize,
  const bool p_useArea1,
  const bool p_useArea2,
  const bool p_verbose){

  //! Standard deviation of the noise
  o_paramStep1.sigma = p_sigma;
  o_paramStep2.sigma = p_sigma;

  //! Size of patches
  if (p_imSize.nChannels == 1) {
    o_paramStep1.sizePatch = (p_sigma < 30.f ? 5 : 7);
    o_paramStep2.sizePatch = 5;
  }
  else {
    o_paramStep1.sizePatch = (p_sigma < 20.f ? 3 :
                             (p_sigma < 50.f ? 5 : 7));
    o_paramStep2.sizePatch = (p_sigma < 50.f ? 3 :
                             (p_sigma < 70.f ? 5 : 7));
  }

  //! Number of similar patches
  if (p_imSize.nChannels == 1) {
    o_paramStep1.nSimilarPatches = (p_sigma < 10.f ? 35 :
                                   (p_sigma < 30.f ? 45 :
                                   (p_sigma < 80.f ? 90 : 100)));
    o_paramStep2.nSimilarPatches = (p_sigma < 20.f ? 15 :
                                   (p_sigma < 40.f ? 25 :
                                   (p_sigma < 80.f ? 30 : 45)));
  }
  else {
    o_paramStep1.nSimilarPatches = o_paramStep1.sizePatch * o_paramStep1.sizePatch * 3;
    o_paramStep2.nSimilarPatches = o_paramStep2.sizePatch * o_paramStep2.sizePatch * 3;
  }

  //! Offset: step between two similar patches
  o_paramStep1.offSet = o_paramStep1.sizePatch / 2;
  o_paramStep2.offSet = o_paramStep2.sizePatch / 2;

  //! Use the homogeneous area detection trick
  o_paramStep1.useHomogeneousArea = p_useArea1;
  o_paramStep2.useHomogeneousArea = p_useArea2;

  //! Size of the search window around the reference patch (must be odd)
  o_paramStep1.sizeSearchWindow = o_paramStep1.nSimilarPatches / 2;
  if (o_paramStep1.sizeSearchWindow % 2 == 0) {
    o_paramStep1.sizeSearchWindow++;
  }
  o_paramStep2.sizeSearchWindow = o_paramStep2.nSimilarPatches / 2;
  if (o_paramStep2.sizeSearchWindow % 2 == 0) {
    o_paramStep2.sizeSearchWindow++;
  }

  //! Size of boundaries used during the sub division
  o_paramStep1.boundary = int(1.5f * float(o_paramStep1.sizeSearchWindow));
  o_paramStep2.boundary = int(1.5f * float(o_paramStep2.sizeSearchWindow));

  //! Parameter used to determine if an area is homogeneous
  o_paramStep1.gamma = 1.05f;
  o_paramStep2.gamma = 1.05f;

  //! Parameter used to estimate the covariance matrix
  if (p_imSize.nChannels == 1) {
    o_paramStep1.beta = (p_sigma < 15.f ? 1.1f :
                        (p_sigma < 70.f ? 1.f : 0.9f));
    o_paramStep2.beta = (p_sigma < 15.f ? 1.1f :
                        (p_sigma < 35.f ? 1.f : 0.9f));
  }
  else {
    o_paramStep1.beta = 1.f;
    o_paramStep2.beta = (p_sigma < 50.f ? 1.2f : 1.f);
  }

  //! Parameter used to determine similar patches
  o_paramStep2.tau = 16.f * o_paramStep2.sizePatch * o_paramStep2.sizePatch * p_imSize.nChannels;

  //! Print information?
  o_paramStep1.verbose = p_verbose;
  o_paramStep2.verbose = p_verbose;

  //! Is first step?
  o_paramStep1.isFirstStep = true;
  o_paramStep2.isFirstStep = false;

  //! Boost the paste trick
  o_paramStep1.doPasteBoost = true;
  o_paramStep2.doPasteBoost = true;
}


//! Main function to process the whole NL-Bayes algorithm.
int runNlBayes(
  std::vector<float> const& i_imNoisy,
  std::vector<float> &o_imBasic,
  std::vector<float> &o_imFinal,
  const ImageSize &p_imSize,
  const bool p_useArea1,
  const bool p_useArea2,
  const float p_sigma,
  const bool p_verbose){

  //! Only 1, 3 or 4-channels images can be processed.
  const unsigned int chnls = p_imSize.nChannels;
  if (! (chnls == 1 || chnls == 3 || chnls == 4)) {
    cout << "Wrong number of channels. Must be 1 or 3!!" << endl;
    return EXIT_FAILURE;
  }

  //! Number of available cores
  unsigned int nbThreads = 1;
#ifdef _OPENMP
    nbThreads = omp_get_max_threads();
    if (p_verbose) {
      cout << "Open MP is used" << endl;
    }
 #endif

  //! Initialization
  o_imBasic.resize(i_imNoisy.size());
  o_imFinal.resize(i_imNoisy.size());

  //! Parameters Initialization
  nlbParams paramStep1, paramStep2;
  initializeNlbParameters(paramStep1, paramStep2, p_sigma, p_imSize, p_useArea1,
                          p_useArea2, p_verbose);

  //! Step 1
  if (paramStep1.verbose) {
    cout << "1st Step...";
  }

  //! RGB to YUV
  vector<float> imNoisy = i_imNoisy;
  transformColorSpace(imNoisy, p_imSize, true);

  //! Divide the noisy image into sub-images in order to easier parallelize the process
  const unsigned nbParts = 2 * nbThreads;
  vector<vector<float> > imNoisySub(nbParts), imBasicSub(nbParts), imFinalSub(nbParts);
  ImageSize imSizeSub;
  if (subDivide(imNoisy, imNoisySub, p_imSize, imSizeSub, paramStep1.boundary, nbParts)
    != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  //! Process all sub-images
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, nbParts/nbThreads) \
            shared(imNoisySub, imBasicSub, imFinalSub, imSizeSub) \
            firstprivate (paramStep1)
 #endif
  for (int n = 0; n < (int) nbParts; n++) {
    processNlBayes(imNoisySub[n], imBasicSub[n], imFinalSub[n], imSizeSub, paramStep1);
  }

  //! Get the basic estimate
  if (subBuild(imBasicSub, o_imBasic, imSizeSub, p_imSize, paramStep1.boundary)
    != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  //! YUV to RGB
  transformColorSpace(o_imBasic, p_imSize, false);

  if (paramStep1.verbose) {
    cout << "done." << endl;
  }

  //! 2nd Step
  if (paramStep2.verbose) {
    cout << "2nd Step...";
  }

  //! Divide the noisy and basic images into sub-images in order to easier parallelize the process
  if (subDivide(i_imNoisy, imNoisySub, p_imSize, imSizeSub, paramStep2.boundary, nbParts)
    != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (subDivide(o_imBasic, imBasicSub, p_imSize, imSizeSub, paramStep2.boundary, nbParts)
    != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  //! Process all sub-images
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, nbParts/nbThreads) \
            shared(imNoisySub, imBasicSub, imFinalSub) \
            firstprivate (paramStep2)
#endif
  for (int n = 0; n < (int) nbParts; n++) {
    processNlBayes(imNoisySub[n], imBasicSub[n], imFinalSub[n], imSizeSub, paramStep2);
  }

  //! Get the final result
  if (subBuild(imFinalSub, o_imFinal, imSizeSub, p_imSize, paramStep2.boundary)
    != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  if (paramStep2.verbose) {
    cout << "done." << endl << endl;
  }

  return EXIT_SUCCESS;
}


//! Generic step of the NL-Bayes denoising (could be the first or the second).
void processNlBayes(
  std::vector<float> const& i_imNoisy,
  std::vector<float> &io_imBasic,
  std::vector<float> &o_imFinal,
  const ImageSize &p_imSize,
  nlbParams &p_params){

  //! Parameters initialization
  const unsigned int sW       = p_params.sizeSearchWindow;
  const unsigned int sP       = p_params.sizePatch;
  const unsigned int sP2      = sP * sP;
  const unsigned int sPC      = sP2 * p_imSize.nChannels;
  const unsigned int nSP      = p_params.nSimilarPatches;
  unsigned int nInverseFailed = 0;
  const float threshold       = p_params.sigma * p_params.sigma * p_params.gamma *
                                (p_params.isFirstStep ? p_imSize.nChannels : 1.f);

  //! Allocate Sizes
  if (p_params.isFirstStep) {
    io_imBasic.resize(p_imSize.whc);
  }
  o_imFinal.resize(p_imSize.whc);

  //! Used matrices during Bayes' estimate
  vector<vector<float> > group3d(p_imSize.nChannels, vector<float> (nSP * sP2));
  vector<float> group3dNoisy(sW * sW * sPC), group3dBasic(sW * sW * sPC);
  vector<unsigned int> index(p_params.isFirstStep ? nSP : sW * sW);
  matParams mat;
  mat.group3dTranspose.resize(p_params.isFirstStep ? nSP * sP2 : sW * sW * sPC);
  mat.tmpMat          .resize(p_params.isFirstStep ? sP2 * sP2 : sPC * sPC);
  mat.baricenter      .resize(p_params.isFirstStep ? sP2 : sPC);
  mat.covMat          .resize(p_params.isFirstStep ? sP2 * sP2 : sPC * sPC);
  mat.covMatTmp       .resize(p_params.isFirstStep ? sP2 * sP2 : sPC * sPC);

  //! ponderation: weight sum per pixel
  vector<float> weight(i_imNoisy.size(), 0.f);

  //! Mask: non-already processed patches
  vector<bool> mask(p_imSize.wh, false);

  //! Only pixels of the center of the image must be processed (not the boundaries)
  for (unsigned int i = sW; i < p_imSize.height - sW; i++) {
    for (unsigned int j = sW; j < p_imSize.width - sW; j++) {
      mask[i * p_imSize.width + j] = true;
    }
  }

  for (unsigned int ij = 0; ij < p_imSize.wh; ij += p_params.offSet) {
    //! Only non-seen patches are processed
    if (mask[ij]) {
      //! Search for similar patches around the reference one
      unsigned int nSimP = p_params.nSimilarPatches;
      if (p_params.isFirstStep) {
        estimateSimilarPatchesStep1(i_imNoisy, group3d, index, ij, p_imSize, p_params);
      }
      else {
        nSimP = estimateSimilarPatchesStep2(i_imNoisy, io_imBasic, group3dNoisy,
                group3dBasic, index, ij, p_imSize, p_params);
      }

      //! Initialization
      bool doBayesEstimate = true;

      //! If we use the homogeneous area trick
      if (p_params.useHomogeneousArea) {
        if (p_params.isFirstStep) {
          doBayesEstimate = !computeHomogeneousAreaStep1(group3d, sP, nSP,
                             threshold, p_imSize);
        }
        else {
          doBayesEstimate = !computeHomogeneousAreaStep2(group3dNoisy, group3dBasic,
                             sP, nSimP, threshold, p_imSize);
        }
      }

      //! Else, use Bayes' estimate
      if (doBayesEstimate) {
        if (p_params.isFirstStep) {
          computeBayesEstimateStep1(group3d, mat, nInverseFailed, p_params);
        }
        else {
          computeBayesEstimateStep2(group3dNoisy, group3dBasic, mat, nInverseFailed,
                                    p_imSize, p_params, nSimP);
        }
      }

      //! Aggregation
      if (p_params.isFirstStep) {
        computeAggregationStep1(io_imBasic, weight, mask, group3d, index,
                                p_imSize, p_params);
      }
      else {
        computeAggregationStep2(o_imFinal, weight, mask, group3dBasic, index,
                                p_imSize, p_params, nSimP);
      }
    }
  }

  //! Weighted aggregation
  computeWeightedAggregation(i_imNoisy, io_imBasic, o_imFinal, weight, p_params, p_imSize);

  if (nInverseFailed > 0 && p_params.verbose) {
    cout << "nInverseFailed = " << nInverseFailed << endl;
  }
}


//! Estimate the best similar patches to a reference one.
void estimateSimilarPatchesStep1(
  std::vector<float> const& i_im,
  std::vector<std::vector<float> > &o_group3d,
  std::vector<unsigned> &o_index,
  const unsigned int p_ij,
  const ImageSize &p_imSize,
  const nlbParams &p_params){

  //! Initialization
  const unsigned int sW    = p_params.sizeSearchWindow;
  const unsigned int sP    = p_params.sizePatch;
  const unsigned int width = p_imSize.width;
  const unsigned int chnls = p_imSize.nChannels;
  const unsigned int wh    = width * p_imSize.height;
  const unsigned int ind   = p_ij - (sW - 1) * (width + 1) / 2;
  const unsigned int nSimP = p_params.nSimilarPatches;
  vector<pair<float, unsigned int> > distance(sW * sW);

  //! Compute distance between patches
  for (unsigned int i = 0; i < sW; i++) {
    for (unsigned int j = 0; j < sW; j++) {
      const unsigned int k = i * width + j + ind;
      float diff = 0.f;
      for (unsigned int p = 0; p < sP; p++) {
        for (unsigned int q = 0; q < sP; q++) {
          const float tmpValue = i_im[p_ij + p * width + q] - i_im[k + p * width + q];
          diff += tmpValue * tmpValue;
        }
      }

      //! Save all distances
      distance[i * sW + j] = make_pair(diff, k);
    }
  }

  //! Keep only the N2 best similar patches
  partial_sort(distance.begin(), distance.begin() + nSimP, distance.end(), comparaisonFirst);

  //! Register position of patches
  for (unsigned int n = 0; n < nSimP; n++) {
    o_index[n] = distance[n].second;
  }

  //! Register similar patches into the 3D group
  for (unsigned int c = 0; c < chnls; c++) {
    for (unsigned int p = 0, k = 0; p < sP; p++) {
      for (unsigned int q = 0; q < sP; q++) {
        for (unsigned int n = 0; n < nSimP; n++, k++) {
          o_group3d[c][k] = i_im[o_index[n] + p * width + q + c * wh];
        }
      }
    }
  }
}


//! Keep from all near patches the similar ones to the reference patch for the second step.
unsigned int estimateSimilarPatchesStep2(
  std::vector<float> const& i_imNoisy,
  std::vector<float> const& i_imBasic,
  std::vector<float> &o_group3dNoisy,
  std::vector<float> &o_group3dBasic,
  std::vector<unsigned int> &o_index,
  const unsigned int p_ij,
  const ImageSize &p_imSize,
  const nlbParams &p_params){

  //! Initialization
  const unsigned int width = p_imSize.width;
  const unsigned int chnls = p_imSize.nChannels;
  const unsigned int wh    = width * p_imSize.height;
  const unsigned int sP    = p_params.sizePatch;
  const unsigned int sW    = p_params.sizeSearchWindow;
  const unsigned int ind   = p_ij - (sW - 1) * (width + 1) / 2;
  vector<pair<float, unsigned int > > distance(sW * sW);

  //! Compute distance between patches
  for (unsigned int i = 0; i < sW; i++) {
    for (unsigned int j = 0; j < sW; j++) {
      const unsigned int k = i * width + j + ind;
      float diff = 0.0f;

      for (unsigned int c = 0; c < chnls; c++) {
        const unsigned int dc = c * wh;
        for (unsigned int p = 0; p < sP; p++) {
          for (unsigned int q = 0; q < sP; q++) {
            const float tmpValue = i_imBasic[dc + p_ij + p * width + q]
                                 - i_imBasic[dc + k    + p * width + q];
            diff += tmpValue * tmpValue;
          }
        }
      }

      //! Save all distances
      distance[i * sW + j] = make_pair(diff, k);
    }
  }

  //! Keep only the nSimilarPatches best similar patches
  partial_sort(distance.begin(), distance.begin() + p_params.nSimilarPatches, distance.end(),
    comparaisonFirst);

  //! Save index of similar patches
  const float threshold = (p_params.tau > distance[p_params.nSimilarPatches - 1].first ?
                           p_params.tau : distance[p_params.nSimilarPatches - 1].first);
  unsigned int nSimP = 0;

  //! Register position of similar patches
  for (unsigned int n = 0; n < distance.size(); n++) {
    if (distance[n].first < threshold) {
      o_index[nSimP++] = distance[n].second;
    }
  }

  //! Save similar patches into 3D groups
  for (unsigned int c = 0, k = 0; c < chnls; c++) {
    for (unsigned int p = 0; p < sP; p++) {
      for (unsigned int q = 0; q < sP; q++) {
        for (unsigned int n = 0; n < nSimP; n++, k++) {
          o_group3dNoisy[k] = i_imNoisy[c * wh + o_index[n] + p * width + q];
          o_group3dBasic[k] = i_imBasic[c * wh + o_index[n] + p * width + q];
        }
      }
    }
  }

  return nSimP;
}


//! Detect if we are in an homogeneous area. In this case, compute the mean.
int computeHomogeneousAreaStep1(
  std::vector<std::vector<float> > &io_group3d,
  const unsigned int p_sP,
  const unsigned int p_nSimP,
  const float p_threshold,
  const ImageSize &p_imSize){

  //! Initialization
  const unsigned int N = p_sP * p_sP * p_nSimP;

  //! Compute the standard deviation of the set of patches
  float stdDev = 0.f;
  for (unsigned int c = 0; c < p_imSize.nChannels; c++) {
    stdDev += computeStdDeviation(io_group3d[c], p_sP * p_sP, p_nSimP, 1);
  }

  //! If we are in an homogeneous area
  if (stdDev < p_threshold) {
    for (unsigned int c = 0; c < p_imSize.nChannels; c++) {
      float mean = 0.f;

      for (unsigned int k = 0; k < N; k++) {
        mean += io_group3d[c][k];
      }

      mean /= (float) N;

      for (unsigned int k = 0; k < N; k++) {
        io_group3d[c][k] = mean;
      }
    }
    return 1;
  }
  else {
    return 0;
  }
}


//! Detect if we are in an homogeneous area. In this case, compute the mean.
int computeHomogeneousAreaStep2(
  std::vector<float> const& i_group3dNoisy,
  std::vector<float> &io_group3dBasic,
  const unsigned int p_sP,
  const unsigned int p_nSimP,
  const float p_threshold,
  const ImageSize &p_imSize){

  //! Parameters
  const unsigned int sP2 = p_sP * p_sP;
  const unsigned int sPC = sP2 * p_imSize.nChannels;

  //! Compute the standard deviation of the set of patches
  const float stdDev = computeStdDeviation(i_group3dNoisy, sP2, p_nSimP, p_imSize.nChannels);

  //! If we are in an homogeneous area
  if (stdDev < p_threshold) {
    for (unsigned int c = 0; c < p_imSize.nChannels; c++) {
      float mean = 0.f;

      for (unsigned int n = 0; n < p_nSimP; n++) {
        for (unsigned int k = 0; k < sP2; k++) {
          mean += io_group3dBasic[n * sPC + c * sP2 + k];
        }
      }

      mean /= float(sP2 * p_nSimP);

      for (unsigned int n = 0; n < p_nSimP; n++) {
        for (unsigned int k = 0; k < sP2; k++) {
          io_group3dBasic[n * sPC + c * sP2 + k] = mean;
        }
      }
    }
    return 1;
  }
  else {
    return 0;
  }
}


//! Compute the Bayes estimation for the 1st step.
 void computeBayesEstimateStep1(
  std::vector<std::vector<float> > &io_group3d,
  matParams &i_mat,
  unsigned int &io_nInverseFailed,
  nlbParams &p_params){

  //! Parameters
  const unsigned int chnls = io_group3d.size();
  const unsigned int nSimP = p_params.nSimilarPatches;
  const unsigned int sP2   = p_params.sizePatch * p_params.sizePatch;
  const float valDiag      = p_params.beta * p_params.sigma * p_params.sigma;

  //! Bayes estimate
  for (unsigned int c = 0; c < chnls; c++) {

      //! Center data around the baricenter
    centerData(io_group3d[c], i_mat.baricenter, nSimP, sP2);

    //! Compute the covariance matrix of the set of similar patches
    covarianceMatrix(io_group3d[c], i_mat.covMat, nSimP, sP2);

    //! Bayes' Filtering
    if (inverseMatrix(i_mat.covMat, sP2) == EXIT_SUCCESS) {
      productMatrix(i_mat.group3dTranspose, i_mat.covMat, io_group3d[c], sP2, sP2, nSimP);
      for (unsigned int k = 0; k < sP2 * nSimP; k++) {
        io_group3d[c][k] -= valDiag * i_mat.group3dTranspose[k];
      }
    }
    else {
      io_nInverseFailed++;
    }

    //! Add baricenter
    for (unsigned int j = 0, k = 0; j < sP2; j++) {
      for (unsigned int i = 0; i < nSimP; i++, k++) {
        io_group3d[c][k] += i_mat.baricenter[j];
      }
    }
  }
}


//! Compute the Bayes estimation for the 2nd step.
void computeBayesEstimateStep2(
  std::vector<float> &i_group3dNoisy,
  std::vector<float> &io_group3dBasic,
  matParams &i_mat,
  unsigned int &io_nInverseFailed,
  const ImageSize &p_imSize,
  nlbParams p_params,
  const unsigned int p_nSimP){

  //! Parameters initialization
  const float diagVal     = p_params.beta * p_params.sigma * p_params.sigma;
  const unsigned int sPC  = p_params.sizePatch * p_params.sizePatch * p_imSize.nChannels;

  //! Center 3D groups around their baricenter
  centerData(io_group3dBasic, i_mat.baricenter, p_nSimP, sPC);
  centerData(i_group3dNoisy, i_mat.baricenter, p_nSimP, sPC);

  //! Compute the covariance matrix of the set of similar patches
  covarianceMatrix(io_group3dBasic, i_mat.covMat, p_nSimP, sPC);

  //! Bayes' Filtering
  for (unsigned int k = 0; k < sPC; k++) {
    i_mat.covMat[k * sPC + k] += diagVal;
  }

  //! Compute the estimate
  if (inverseMatrix(i_mat.covMat, sPC) == EXIT_SUCCESS) {
    productMatrix(io_group3dBasic, i_mat.covMat, i_group3dNoisy, sPC, sPC, p_nSimP);
    for (unsigned int k = 0; k < sPC * p_nSimP; k++) {
      io_group3dBasic[k] = i_group3dNoisy[k] - diagVal * io_group3dBasic[k];
    }
  }
  else {
    io_nInverseFailed++;
  }

  //! Add baricenter
  for (unsigned int j = 0, k = 0; j < sPC; j++) {
    for (unsigned int i = 0; i < p_nSimP; i++, k++) {
      io_group3dBasic[k] += i_mat.baricenter[j];
    }
  }
}


//! Aggregate estimates of all similar patches contained in the 3D group.
void computeAggregationStep1(
  std::vector<float> &io_im,
  std::vector<float> &io_weight,
  std::vector<bool> &io_mask,
  std::vector<std::vector<float> > const& i_group3d,
  std::vector<unsigned int> const& i_index,
  const ImageSize &p_imSize,
  const nlbParams &p_params){

  //! Parameters initializations
  const unsigned int chnls  = p_imSize.nChannels;
  const unsigned int width  = p_imSize.width;
  const unsigned int height = p_imSize.height;
  const unsigned int sP     = p_params.sizePatch;
  const unsigned int nSimP  = p_params.nSimilarPatches;

  //! Aggregate estimates
  for (unsigned int n = 0; n < nSimP; n++) {
    const unsigned int ind = i_index[n];
    for (unsigned int c = 0; c < chnls; c++) {
      const unsigned int ij = ind + c * width * height;
      for (unsigned int p = 0; p < sP; p++) {
        for (unsigned int q = 0; q < sP; q++) {
          io_im[ij + p * width + q] += i_group3d[c][(p * sP + q) * nSimP + n];
          io_weight[ij + p * width + q]++;
        }
      }
    }

    //! Use Paste Trick
    io_mask[ind] = false;

    if (p_params.doPasteBoost) {
      io_mask[ind - width ] = false;
      io_mask[ind + width ] = false;
      io_mask[ind - 1		] = false;
      io_mask[ind + 1		] = false;
    }
  }
}


//! Aggregate estimates of all similar patches contained in the 3D group.
void computeAggregationStep2(
  std::vector<float> &io_im,
  std::vector<float> &io_weight,
  std::vector<bool> &io_mask,
  std::vector<float> const& i_group3d,
  std::vector<unsigned int> const& i_index,
  const ImageSize &p_imSize,
  const nlbParams &p_params,
  const unsigned int p_nSimP){

  //! Parameters initializations
  const unsigned int chnls = p_imSize.nChannels;
  const unsigned int width = p_imSize.width;
  const unsigned int wh    = width * p_imSize.height;
  const unsigned int sP    = p_params.sizePatch;

  //! Aggregate estimates
  for (unsigned int n = 0; n < p_nSimP; n++) {
    const unsigned int ind = i_index[n];
    for (unsigned int c = 0, k = 0; c < chnls; c++) {
      const unsigned int ij = ind + c * wh;
      for (unsigned int p = 0; p < sP; p++) {
        for (unsigned int q = 0; q < sP; q++, k++) {
          io_im[ij + p * width + q] += i_group3d[k * p_nSimP + n];
          io_weight[ij + p * width + q]++;
        }
      }
    }

    //! Apply Paste Trick
    io_mask[ind] = false;

    if (p_params.doPasteBoost) {
      io_mask[ind - width ] = false;
      io_mask[ind + width ] = false;
      io_mask[ind - 1     ] = false;
      io_mask[ind + 1     ] = false;
    }
  }
}


//! Compute the final weighted aggregation.
void computeWeightedAggregation(
  std::vector<float> const& i_imNoisy,
  std::vector<float> &io_imBasic,
  std::vector<float> &io_imFinal,
  std::vector<float> const& i_weight,
  const nlbParams &p_params,
  const ImageSize &p_imSize){

  for (unsigned int c = 0, k = 0; c < p_imSize.nChannels; c++) {

    for (unsigned int ij = 0; ij < p_imSize.wh; ij++, k++) {

      //! To avoid weighting problem (particularly near boundaries of the image)
      if (i_weight[k] > 0.f) {
        if (p_params.isFirstStep) {
          io_imBasic[k] /= i_weight[k];
        }
        else {
          io_imFinal[k] /= i_weight[k];
        }
      }
      else {
        if (p_params.isFirstStep) {
          io_imBasic[k] = i_imNoisy[k];
        }
        else {
          io_imFinal[k] = io_imBasic[k];
        }
      }
    }
  }
}
