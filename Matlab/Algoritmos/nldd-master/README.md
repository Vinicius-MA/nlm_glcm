# Non-Local Dual Denoising

This project contains the source code for NLDD, along with NL-Bayes and DDID. Wrappers for Matlab are provided.

## Compilation

To compile the standalone solution:

### Requirements
1. LibPng and FFTW are required. If not already installed, do so before proceeding. Remember that the libraries need to be compiled using the same target machine (either x86 or x64) as the compilation tool configuration. This could generate linker errors otherwise. 
2. CMake is required to generate the Makefile/Compilation solution. 

### Compilation:
1. Go to src/build folder
2. __On Unix__

    $ cmake ../src
    $make

 __On Windows__

 Edit the VS64CMake.bat example and change the paths. For each library, you need to give two paths: To the .h header files and the full path to the .lib. Finally the generator tool should be modified (the -G parameter) if different than "Visual Studio 10 Win64". For a full list of supported generators, execute cmake from the command prompt.


## Matlab
To compile the mex files for Matlab:
   Run the script matlab/compileMex.m from within Matlab 

Already generated MEX files can be found in the /matlab/mex folder if OpenMP is installed (supporting multithreaded execution) or in /matlab/mex_no_OMP if OpenMP has not been installed in the system. OpenMP installation is recommended.

To use a mex file from within Matlab, just call it as any standard Matlab function. Mind that it needs to be either in the current Matlab folder or in the Matlab path.

An example file is given in matlab/test.m. In Windows, besides the mex files included in the path, libfftw3-f.dll is required to run the tests.

Function calls from within Matlab:
[denoised_I] = DDIDstep(guide_I, noisy_I, sigma2, r, sigmaS, gammaR, gammaF);
[denoised_I] = NLDD(noisy_I, sigma, verbose);
[denoised_I] = NlBayes(noisy_I, sigma, verbose);

