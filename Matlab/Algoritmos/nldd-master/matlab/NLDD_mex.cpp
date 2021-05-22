#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
    
#include <mex.h>

#include "DDID/RunDDID.h"

using namespace std;

/**
 * @file   NLDD.cpp
 * @brief  NLDD that receives a Matlab image, adds noise to it and stores the result 
 * on a Matrix
 * @author Martin Rais
 */

// Function call example:
// denoised_I = NLDD(noisy_I, sigma, verbose);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{//! Check if there is the right number of input parameters
	if(nrhs!=3) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Usage: [denoised_I] = NLDD(imNoisy, sigma,verbose).");
	}
	if(nlhs!=1) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "An output is required.");
	}
	/* make sure the first input argument is scalar */
	if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||mxGetNumberOfElements(prhs[1])!=1 ) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar", "Input sigma must be a scalar.");
	}
    // Commented because no validation is performed on the verbose parameter
//	if( !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||mxGetNumberOfElements(prhs[2])!=1 ) 
//	{
//		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar", "Input sigma must be a scalar.");
//	}
	// This code validates that inMatrix, represented by prhs[1], is type double.
	if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble",
			"Input matrix must be type double.");
	}
  //! Parameters
	double sigma  = mxGetScalar(prhs[1]);//atof(argv[3]);
	bool verbose = (bool) mxGetScalar(prhs[2]);//		(bool) atoi(argv[4]);

	int ncols = mxGetDimensions(prhs[0])[1];
	int nrows = mxGetDimensions(prhs[0])[0];
	int nChannels = (mxGetNumberOfDimensions(prhs[0]) == 3 ? mxGetDimensions(prhs[0])[2]  : 1);
    double *img = mxGetPr(prhs[0]);
    
	// Create Matlab output data
	/* create the output matrix */
	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]), mxGetDimensions(prhs[0]), 
         mxDOUBLE_CLASS, mxREAL); 
	//Use the mxGetPr function to assign the outMatrix argument to plhs[0]
	//
	/* get a pointer to the real data in the output matrix */
	double *outMatrix = mxGetPr(plhs[0]);

  //! Declarations
  ImageSize imSize;
  imSize.height = ncols;
  imSize.width = nrows;
  imSize.nChannels = nChannels;
  imSize.whc = nrows * ncols * nChannels;
  imSize.wh = nrows * ncols ;
  vector<float> imNoisy(imSize.whc);
  for (int i=0;i<imSize.whc;i++)
	  imNoisy[i] = img[i];
  
  //! Denoise the image
  vector<float> imFinal;
  std::cout << imSize.height << " " << imSize.width << " " << imSize.nChannels << std::endl;
  std::cout << sigma << " " << verbose << std::endl;
  runDDIDNlBayes(imNoisy, imFinal, imSize, sigma, verbose);
  for (int i=0;i<imSize.whc;i++)
  {
	  outMatrix[i] = imFinal[i];
  }
  
}
