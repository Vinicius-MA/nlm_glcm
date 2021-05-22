#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
    
#include <mex.h>

#include "DDID/RunDDID.h"

using namespace std;

/**
 * @file   DDIDStep.cpp
 * @brief  DDIDStep that receives a noisy Matlab image, and performs one step of DDID
 * @author Martin Rais
 */

// Function call example:
// denoised_I = DDIDstep(guide_I, noisy_I, sigma2, r, sigmaS, gammaR, gammaF);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//! Check if there is the right number of input parameters
	if(nrhs!=7) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Usage: [denoised_I] = DDIDstep(guide_I, noisy_I, sigma2, r, sigmaS, gammaR, gammaF).");
	}
	if(nlhs!=1) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "An output is required.");
	}
	//! Validate data
	if(( !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||mxGetNumberOfElements(prhs[2])!=1 ) || ( !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ||mxGetNumberOfElements(prhs[3])!=1 ) ||
		( !mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) ||mxGetNumberOfElements(prhs[4])!=1 ) || ( !mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) ||mxGetNumberOfElements(prhs[5])!=1 ) ||
		( !mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) ||mxGetNumberOfElements(prhs[6])!=1 ) )
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar", "Inputs must be scalar.");
	}
	// This code validates that the input matrix is type double.
	if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble",
			"Input matrix must be type double.");
	}
  //! Parameters
	double sigma  = sqrt(mxGetScalar(prhs[2]));
	int r = (int) mxGetScalar(prhs[3]);
	double sigmaS = mxGetScalar(prhs[4]);
	double gammaR = mxGetScalar(prhs[5]);
	double gammaF = mxGetScalar(prhs[6]);

	//! DDID Parameters
	ddidParams params;
	params.sigma   = sigma / 255.f;
	params.r       = r;
	params.sigmaS  = sigmaS;
	params.gammaF  = gammaF;
	params.gammaR  = gammaR;

	int nrowsGuide = mxGetDimensions(prhs[0])[1];
	int ncolsGuide = mxGetDimensions(prhs[0])[0];
	int nChannelsGuide = (mxGetNumberOfDimensions(prhs[0]) == 3 ? mxGetDimensions(prhs[0])[2]  : 1);
    double *imgGuide = mxGetPr(prhs[0]);

	int nrowsNoisy = mxGetDimensions(prhs[1])[1];
	int ncolsNoisy = mxGetDimensions(prhs[1])[0];
	int nChannelsNoisy = (mxGetNumberOfDimensions(prhs[1]) == 3 ? mxGetDimensions(prhs[1])[2]  : 1);
    double *imgNoisy = mxGetPr(prhs[1]);

	if (ncolsGuide != ncolsNoisy || nrowsGuide != nrowsNoisy || nChannelsGuide != nChannelsNoisy)
	{
		mexErrMsgTxt("Guide and Noisy images should have same size and amount of channels");
	}
	// Create Matlab output data
	/* create the output matrix */
	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]), mxGetDimensions(prhs[0]), 
         mxDOUBLE_CLASS, mxREAL); 
	/* get a pointer to the real data in the output matrix */
	double *outMatrix = mxGetPr(plhs[0]);

  //! Declarations
  ImageSize imSize;
  imSize.height = nrowsNoisy;
  imSize.width = ncolsNoisy;
  imSize.nChannels = nChannelsNoisy;
  imSize.wh = nrowsNoisy * ncolsNoisy ;
  imSize.whc = imSize.wh * nChannelsNoisy;
  vector<float> imNoisy(imSize.whc);
  vector<float> imGuide(imSize.whc);

  cout << "Image size: " << imSize.height << "x" << imSize.width << "x" << imSize.nChannels << endl;
  cout << "Sigma: " << sigma << endl;
  cout << "R: " << r << endl;
  cout << "sigma_S: " << sigmaS << endl;
  cout << "gamma_R: " << gammaR<< endl;
  cout << "gamma_F: " << gammaF << endl;

  //! For now, the image must be in [0, 1].
  for (unsigned int i=0;i<imSize.whc;i++)
  {
	  imNoisy[i] = imgNoisy[i] / 255.f;
      imGuide[i] = imgGuide[i] / 255.f;
  }
  
  //! Obtain YUV image
  transformColorSpaceDDID(imNoisy, imSize, true);
  transformColorSpaceDDID(imGuide, imSize, true);
  
  //! Denoise the image
  vector<float> imFinal;
  
  processDDID(imGuide, imNoisy, imFinal, imSize, params);

  //! Go back to RGB space
  transformColorSpaceDDID(imFinal, imSize, false);

  //! Go back to [0, 255]
  for (unsigned int i=0;i<imSize.whc;i++)
  {
	  outMatrix[i] = imFinal[i]* 255.f;
  }
  
}
