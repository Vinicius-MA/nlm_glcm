/*
 * Copyright (c) 2014, Marc Lebrun <marc.lebrun.ik@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <string.h>

#include "NlBayes/NlBayes.h"
#include "Utilities/LibImages.h"
using namespace std;

// c: pointer to original argc
// v: pointer to original argv
// o: option name after hyphen
// d: default value (if NULL, the option takes no argument)
// d: default value (if NULL, the option takes no argument)
static char *pick_option(int *c, char ***v, const char *o, const char *d)
{
   int argc = *c;
   char **argv = *v;
   int id = d ? 1 : 0;
   for (int i = 0; i < argc - id; i++)
      if (argv[i][0] == '-' && 0 == strcmp(argv[i]+1, o))
      {
         char *r = argv[i+id]+1-id;
         *c -= id+1;
         for (int j = i; j < argc - id; j++)
            (*v)[j] = (*v)[j+id+1];
         return r;
      }
   return (char*) d;
}

/**
 * @file   main.cpp
 * @brief  Main executable file
 *
 * @author MARC LEBRUN  <marc.lebrun.ik@gmail.com>
 */

int main(
  int argc,
  char **argv){ //! Check if there is the right number of input parameters

  //! Parameters
  char *noisy_file   = pick_option(&argc, &argv, "t", "");
  const bool verbose = pick_option(&argc, &argv, "v", NULL) ? true : false;

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " input output sigma [-t noisy_input] [-v]" << endl;
    cout << "    -t noisy_input : add noise (sigma) to input and write the noisy image" << endl;
    cout << "    -v             : verbose" << endl;
    return EXIT_FAILURE;
  }

  const float sigma  = atof(argv[3]);

  //! Declarations
  vector<float> im;
  ImageSize imSize;

  //! Read Image
  if (loadImage(argv[1], im, imSize, verbose) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  //! Add some noise
  vector<float> imNoisy;
  if(strcmp(noisy_file,"") != 0)
     addNoise(im, imNoisy, sigma, verbose);
  else
     imNoisy = im;

  //! Denoise the image
  std::vector<float> o_imBasic, o_imDenoised;
  int ret = runNlBayes(imNoisy, o_imBasic, o_imDenoised, imSize, true,
                       false, sigma, verbose);

  if(strcmp(noisy_file,"") != 0) {
     saveImage(noisy_file, imNoisy, imSize, 0.f, 255.f);

     //! Compute the PSNR
     float psnr, rmse;
     computePsnr(im, o_imDenoised, psnr, rmse, "Denoised: ", true);
  }

  //! Save the final denoised image
  return saveImage(argv[2], o_imDenoised, imSize, 0.f, 255.f);
}
