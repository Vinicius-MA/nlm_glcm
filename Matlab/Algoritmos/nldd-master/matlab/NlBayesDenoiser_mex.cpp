#include <vector>
#include <algorithm>
#include "NlBayes/NlBayes.h"
#include "Utilities/LibImages.h"
#include <cstdlib>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include <iostream>
#include <string>
    
#include <mex.h>


using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{//! Check if there is the right number of input parameters
	if(nrhs!=3) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Usage: [denoised_I] = NlBayes(imNoisy, sigma,verbose).");
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
	// This code validates that inMatrix, represented by prhs[1], is type double.
	if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) 
	{
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble",
			"Input matrix must be type double.");
	}
  //! Parameters
	double p_sigma  = mxGetScalar(prhs[1]);//atof(argv[3]);
	bool verbose = (bool) mxGetScalar(prhs[2]);//		(bool) atoi(argv[4]);

	int h = mxGetDimensions(prhs[0])[1];
	int w = mxGetDimensions(prhs[0])[0];
	int c = (mxGetNumberOfDimensions(prhs[0]) == 3 ? mxGetDimensions(prhs[0])[2]  : 1);
    double *imNoisy = mxGetPr(prhs[0]);
    
	// Create Matlab output data
	/* create the output matrix */
	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]), mxGetDimensions(prhs[0]), 
         mxDOUBLE_CLASS, mxREAL); 
	/* get a pointer to the real data in the output matrix */
	double *outMatrix = mxGetPr(plhs[0]);

  unsigned int wh = w * h;
  unsigned int whc = wh * c;
  ImageSize p_imSize = {w, h, c, wh, whc};
  std::vector<float> i_imNoisy(whc);
  for (int i=0;i<p_imSize.whc;i++)
	  i_imNoisy[i] = imNoisy[i];
  
  
  std::vector<float> o_imBasic, o_imDenoised;

  int ret = runNlBayes(i_imNoisy, o_imBasic, o_imDenoised, p_imSize, true,
                       false, p_sigma, verbose);

  for (int i=0;i<p_imSize.whc;i++)
  {
	  outMatrix[i] = o_imDenoised[i];
  }

}