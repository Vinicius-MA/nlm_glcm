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
#include "io_png.h"

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

//! Load image, check the number of channels.
int loadImage(
  char* p_name,
  std::vector<float> &o_im,
  ImageSize &o_imSize,
  const bool p_verbose){

  //! read input image
  if (p_verbose) {
    cout << endl << "Read input image...";
  }

  float *imTmp = NULL;
  size_t w, h, c;
  imTmp = read_png_f32(p_name, &w, &h, &c);
  if (!imTmp) {
    cout << "error :: " << p_name << " not found or not a correct png image" << endl;
    return EXIT_FAILURE;
  }
  if (p_verbose) {
    cout << "done." << endl;
  }

  //! test if image is really a color image and exclude the alpha channel
  if (c > 2) {
    unsigned int k = 0;
    while (k < w * h && imTmp[k] == imTmp[w * h + k] && imTmp[k] == imTmp[2 * w * h + k]) {
      k++;
    }
    c = (k == w * h ? 1 : 3);
  }

  //! Some image informations
  if (p_verbose) {
    cout << "image size :" << endl;
    cout << " - width          = " << w << endl;
    cout << " - height         = " << h << endl;
    cout << " - nb of channels = " << c << endl;
  }

  //! Initializations
  o_imSize.width     = w;
  o_imSize.height    = h;
  o_imSize.nChannels = c;
  o_imSize.wh        = w * h;
  o_imSize.whc       = w * h * c;
  o_im.resize(w * h * c);
  for (unsigned int k = 0; k < w * h * c; k++) {
    o_im[k] = imTmp[k];
  }

  return EXIT_SUCCESS;
}


//! write image.
int saveImage(
  char* p_name,
  std::vector<float> const& i_im,
  const ImageSize &p_imSize,
  const float p_min,
  const float p_max){

  //! Allocate Memory
  float* imTmp = new float[p_imSize.whc];

  //! Check for boundary problems
  for (unsigned int k = 0; k < p_imSize.whc; k++) {
    imTmp[k] = (i_im[k] < p_min ? p_min : (i_im[k] > p_max ? p_max : i_im[k]));
  }

  if (write_png_f32(p_name, imTmp, p_imSize.width, p_imSize.height, p_imSize.nChannels) != 0) {
    cout << "... failed to save png image " << p_name << endl;
    return EXIT_FAILURE;
  }

  //! Free Memory
  delete[] imTmp;

  return EXIT_SUCCESS;
}


