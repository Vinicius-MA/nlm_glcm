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

#include "RunDDID.h"
#include "../Utilities/Utilities.h"
#include "../NlBayes/NlBayes.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;



//! Apply the NlBayes and DDID denoising algorithms.
int runDDIDNlBayes(
  std::vector<float> const& i_imNoisy,
  std::vector<float> &o_imFinal,
  const ImageSize &p_imSize,
  const float p_sigma,
  const bool p_verbose){

  //! Parameters
  ddidParams params;
  params.sigma   = p_sigma / 255.f;
  params.r       = 15;
  params.sigmaS  = 7.f;
  params.gammaF  = 0.8f;
  params.gammaR  = 0.7f;
  
  //! Get the guide image
  vector<float> imBasic, imGuide;
  if (p_verbose) {
    cout << "NlBayes" << endl;
  }
  runNlBayes(i_imNoisy, imBasic, imGuide, p_imSize, true, false, p_sigma, false);

  //! For now, the image must be in [0, 1].
  vector<float> imNoisy(p_imSize.whc);
  for (unsigned int k = 0; k < p_imSize.whc; k++) {
    imNoisy[k] = i_imNoisy[k] / 255.f;
    imGuide[k] /= 255.f;
  }

  //! Obtain YUV image
  transformColorSpaceDDID(imNoisy, p_imSize, true);
  transformColorSpaceDDID(imGuide, p_imSize, true);

  //! Third step of DDID
  processDDID(imGuide, imNoisy, o_imFinal, p_imSize, params);

  //! Go back to RGB space
  transformColorSpaceDDID(o_imFinal, p_imSize, false);

  //! Go back to [0, 255]
  for (unsigned int k = 0; k < p_imSize.whc; k++) {
    o_imFinal[k] *= 255.f;
  }

  return EXIT_SUCCESS;
}



//! Process the core function of DDID algorithm.
void processDDID(
  std::vector<float> const& i_guide,
  std::vector<float> const& i_im,
  std::vector<float> &o_im,
  const ImageSize &p_imSize,
  const ddidParams &p_params){

  //! Tabulate exp(-x), faster than using directly function exp
  vector<float> expLut;
  const float lutMax = 30.f;
  const float lutPrecision = 1000.f;
  fillExpLut(expLut, lutPrecision, lutMax);

  //! Initializations
  const unsigned int sW = 2 * p_params.r + 1;
  const unsigned int nC = p_imSize.nChannels;
  const unsigned int nb = p_imSize.width;
  vector<float> H(sW * sW);
  for (int i = 0; i < (int) sW; i++) {
    float* iH = &H[i * sW];
    for (int j = 0; j < (int) sW; j++) {
      iH[j] = getExp(expLut, ((i - p_params.r) * (i - p_params.r) + (j - p_params.r) * (j - p_params.r))
                  / (2 * p_params.sigmaS * p_params.sigmaS), lutPrecision, lutMax);
    }
  }
  o_im.resize(p_imSize.whc);

  //! Add boundaries
  vector<float> imSym, guideSym;
  ImageSize imSizeSym;
  setBoundaries(i_im, imSym, p_imSize, imSizeSym, p_params.r);
  setBoundaries(i_guide, guideSym, p_imSize, imSizeSym, p_params.r);

  //! Parallelization config
#ifdef _OPENMP
  const unsigned int nbThreads = omp_get_num_procs();
  omp_set_num_threads(nbThreads);
#else
  const unsigned int nbThreads = 1;
#endif

  //! FFTW initialization
  fftwf_complex** Gn = new fftwf_complex*[nbThreads];
  fftwf_complex** Sn = new fftwf_complex*[nbThreads];

  for (unsigned int n = 0; n < nbThreads; n++) {
    Gn[n] = (fftwf_complex*) fftwf_malloc(nb * nC * sW * sW * sizeof(fftwf_complex));
    Sn[n] = (fftwf_complex*) fftwf_malloc(nb * nC * sW * sW * sizeof(fftwf_complex));
  }

  int nbTable[2]    = {(int) sW, (int) sW};
  int nembed [2]    = {(int) sW, (int) sW};
  fftwf_plan* planG = new fftwf_plan[nbThreads];
  fftwf_plan* planS = new fftwf_plan[nbThreads];
  for (unsigned int n = 0; n < nbThreads; n++) {
    planS[n] = fftwf_plan_many_dft(2, nbTable, nb * nC, Sn[n], nembed, 1, sW * sW,
                                   Sn[n], nembed, 1, sW * sW, FFTW_FORWARD, FFTW_ESTIMATE);
    planG[n] = fftwf_plan_many_dft(2, nbTable, nb * nC, Gn[n], nembed, 1, sW * sW,
                                   Gn[n], nembed, 1, sW * sW, FFTW_FORWARD, FFTW_ESTIMATE);
  }

#ifdef _OPENMP
#pragma omp parallel shared(planG, planS, H, imSym, guideSym, Gn, Sn)
{
#endif
  //! Vector initialization
  vector<float> g (nb * nC * sW * sW);
  vector<float> y (nb * nC * sW * sW);
  vector<float> k (nb * sW * sW);
  vector<float> gt(nb * nC);
  vector<float> st(nb * nC);
  vector<float> V (nb);

#ifdef _OPENMP
  #pragma omp for schedule(dynamic) nowait
#endif
  for (int i = 0; i < p_imSize.height; i++) {

#ifdef _OPENMP
      const unsigned int nCurThread = omp_get_thread_num();
#else
      const unsigned int nCurThread = 0;
#endif

    for (unsigned int j = 0; j < p_imSize.width; j++) {

      //! Spatial Domain: Bilateral Filter
      vector<float> d(sW * sW);
      for (unsigned int c = 0; c < nC; c++) {
        const float value = guideSym[c * imSizeSym.wh + (p_params.r + i) * imSizeSym.width + j + p_params.r];

        for (unsigned int p = 0; p < sW; p++) {
          float* iG = &g[j * nC * sW * sW + c * sW * sW + p * sW];
          float* iY = &y[j * nC * sW * sW + c * sW * sW + p * sW];
          float* iD = &d[p * sW];
          const float* iSg = &guideSym[c * imSizeSym.wh + (p + i) * imSizeSym.width + j];
          const float* iSi = &imSym   [c * imSizeSym.wh + (p + i) * imSizeSym.width + j];

          for (unsigned int q = 0; q < sW; q++) {
            iG[q]  = iSg[q];
            iY[q]  = iSi[q];
            iD[q] += (iSg[q] - value) * (iSg[q] - value);
          }
        }
      }

      //! Equation 4
      float* iK = &k[j * sW * sW];
      float sumK  = 0.f;
      float sumK2 = 0.f;
      const float normK = 1.f / (float) (p_params.gammaR * p_params.sigma * p_params.sigma);
      for (unsigned pq = 0; pq < sW * sW; pq++) {
        const float k = getExp(expLut, d[pq] * normK, lutPrecision, lutMax) * H[pq];
        iK[pq] = k;
        sumK  += k;
        sumK2 += k * k;
      }

      //! Equation 2 & 3
      float* iGt = &gt[j * nC];
      float* iSt = &st[j * nC];
      for (unsigned int c = 0; c < nC; c++) {

        const float* gc = &g[j * nC * sW * sW + c * sW * sW];
        const float* yc = &y[j * nC * sW * sW + c * sW * sW];
        float g = 0.f;
        float s = 0.f;
        for (unsigned int pq = 0; pq < sW * sW; pq++) {
          g += gc[pq] * iK[pq];
          s += yc[pq] * iK[pq];
        }

        iGt[c] = g / sumK;
        iSt[c] = s / sumK;

        //! Fourier Domain: Wavelet Shrinkage
        V[j] = p_params.sigma * p_params.sigma * sumK2;

        //! Equation 6 & 7
        const unsigned int sH = (sW - 1) / 2;
        const float gtc = iGt[c];
        const float stc = iSt[c];
        fftwf_complex* G = &Gn[nCurThread][j * sW * sW * nC + c * sW * sW];
        fftwf_complex* S = &Sn[nCurThread][j * sW * sW * nC + c * sW * sW];

        for (unsigned int p = 0; p <= sH; p++) {
          for (unsigned int q = 0; q <= sH; q++) {
            G[p * sW + q][0] = (gc[(sH + p) * sW + sH + q] - gtc) * iK[(sH + p) * sW + sH + q];
            G[p * sW + q][1] = 0.f;
            S[p * sW + q][0] = (yc[(sH + p) * sW + sH + q] - stc) * iK[(sH + p) * sW + sH + q];
            S[p * sW + q][1] = 0.f;

            if (q < sH) {
              G[p * sW + sH + q + 1][0] = (gc[(sH + p) * sW + q] - gtc) * iK[(sH + p) * sW + q];
              G[p * sW + sH + q + 1][1] = 0.f;
              S[p * sW + sH + q + 1][0] = (yc[(sH + p) * sW + q] - stc) * iK[(sH + p) * sW + q];
              S[p * sW + sH + q + 1][1] = 0.f;
            }

            if (p < sH) {
              G[(p + sH + 1) * sW + q][0] = (gc[p * sW + q + sH] - gtc) * iK[p * sW + q + sH];
              G[(p + sH + 1) * sW + q][1] = 0.f;
              S[(p + sH + 1) * sW + q][0] = (yc[p * sW + q + sH] - stc) * iK[p * sW + q + sH];
              S[(p + sH + 1) * sW + q][1] = 0.f;

              if (q < sH) {
                G[(p + sH + 1) * sW + q + sH + 1][0] = (gc[p * sW + q] - gtc) * iK[p * sW + q];
                G[(p + sH + 1) * sW + q + sH + 1][1] = 0.f;
                S[(p + sH + 1) * sW + q + sH + 1][0] = (yc[p * sW + q] - stc) * iK[p * sW + q];
                S[(p + sH + 1) * sW + q + sH + 1][1] = 0.f;
              }
            }
          }
        }
      }
    }

    //! Equation 6
    fftwf_execute(planG[nCurThread]);

    //! Equation 7
    fftwf_execute(planS[nCurThread]);

    for (unsigned int j = 0; j < p_imSize.width; j++) {
      for(unsigned int c = 0; c < nC; c++) {
        const fftwf_complex* G = &Gn[nCurThread][j * nC * sW * sW + c * sW * sW];
        const fftwf_complex* S = &Sn[nCurThread][j * nC * sW * sW + c * sW * sW];

        //! Equation 9
        float St = 0.f;
        for (unsigned int pq = 0; pq < sW * sW; pq++) {

          //! Equation 9
          const float K = getExp(expLut, p_params.gammaF * V[j] / (G[pq][0] * G[pq][0] + G[pq][1] * G[pq][1]),
                         lutPrecision, lutMax);

          //! Equation 8
          St += S[pq][0] * K;
        }
        St /= float(sW * sW);

        //! Save denoised pixel
        o_im[c * p_imSize.wh + i * p_imSize.width + j] = st[j * nC + c] + St;
      }
    }
  }
#ifdef _OPENMP
}
#endif

  //! Clean up memory
  for (unsigned int n = 0; n < nbThreads; n++) {
    fftwf_free(Gn[n]);
    fftwf_free(Sn[n]);
  }
  for (unsigned int n = 0; n < nbThreads; n++) {
    fftwf_destroy_plan(planG[n]);
    fftwf_destroy_plan(planS[n]);
  }
  fftwf_cleanup();
  delete[] Gn;
  delete[] Sn;
  delete[] planG;
  delete[] planS;
}

//! Apply the DDID denoising algorithm.
int runDDID(
  std::vector<float> const& i_imNoisy,
  std::vector<float> &o_imFinal,
  const ImageSize &p_imSize,
  const float p_sigma,
  const bool p_verbose){

#ifdef _OPENMP
  cout << "OPEN MP is used" << endl;
#else
  cout << "OPEN MP isn't used. You can improve the speed by using it : make OMP=1" << endl;
#endif

  //! Parameters
  ddidParams params;
  params.sigma     = p_sigma / 255.f;
  params.verbose   = p_verbose;
  params.r         = 15;
  params.sigmaS    = 7.f;

  //! For now, the image must be in [0, 1].
  vector<float> imNoisy(p_imSize.whc);
  for (unsigned int k = 0; k < p_imSize.whc; k++) {
    imNoisy[k] = i_imNoisy[k] / 255.f;
  }

  //! Obtain YUV image
  transformColorSpace(imNoisy, p_imSize, true);

  //! First step
  if (params.verbose) {
    cout << "1st step...";
  }
  vector<float> imGuide(p_imSize.whc);
  params.gammaF    = 4.f;
  params.gammaR    = 100.f;
  processDDID(imNoisy, imNoisy, imGuide, p_imSize, params);
  if (params.verbose) {
    cout << "done." << endl;
  }

  //! Second step
  if (params.verbose) {
    cout << "2nd step...";
  }
  vector<float> imBasic(p_imSize.whc);
  params.gammaF    = 0.4f;
  params.gammaR    = 8.7f;
  processDDID(imGuide, imNoisy, imBasic, p_imSize, params);
  if (params.verbose) {
    cout << "done." << endl;
  }

  //! Third step
  if (params.verbose) {
    cout << "3rd step...";
  }
  vector<float> imFinal(p_imSize.whc);
  params.gammaF    = 0.8f;
  params.gammaR    = 0.7f;
  processDDID(imBasic, imNoisy, imFinal, p_imSize, params);
  if (params.verbose) {
    cout << "done." << endl;
  }

  //! Go back to RGB space
  transformColorSpace(imFinal, p_imSize, false);

  //! Go back to [0, 255]
  o_imFinal.resize(p_imSize.whc);
  for (unsigned int k = 0; k < p_imSize.whc; k++) {
    o_imFinal[k] = imFinal[k] * 255.f;
  }

  return EXIT_SUCCESS;
}



























