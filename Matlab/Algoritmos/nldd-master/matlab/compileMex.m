% Compile mex files

% Code specific libs and include dirs
% For windows
if (ispc)
    FFTWPATH = 'C:/Libs/FFTW-3.3.4/';
    LIBFFTW = ['-I',FFTWPATH, ' ', FFTWPATH, 'libfftw3f-3.lib'];      % static link
end
% For Mac OS
if (ismac)
    LIBFFTW = [' -I/usr/local/include/ -I/usr/include ', ...
           ' /usr/local/lib/libfftw3f.a ' ]; % static link
end
% For Unix
if (isunix && ~ismac)
    LIBFFTW = [' -I/usr/local/include/ -I/usr/include ', ...
           ' /usr/lib/libfftw3f.a ' ]; % static link
end
if (~ispc && ~ismac && ~isunix)
    disp('Your OS is not supported');
    quit;
end

SLIBS = [ LIBFFTW , ' ' ];

% Code specific flags
FLAGS = ' -O3 -ftree-vectorize -funroll-loops -ansi -fPIC -fopenmp ';

% Put all the flags everywhere just in case
SFLAGS = [' "CXXFLAGS=\$CXXFLAGS ',FLAGS, '" ', ...
          ' "CFLAGS=\$CFLAGS '    ,FLAGS, '" ', ...
          ' "LDFLAGS=\$LDFLAGS '  ,FLAGS, '" '];

% common source code files and include directories          
COMMON_SRCS = [' ../src/DDID/RunDDID.cpp ../src/NlBayes/LibMatrix.cpp ../src/NlBayes/NlBayes.cpp ' ...
              ' ../src/Utilities/mt19937ar.c ../src/Utilities/LibImages.cpp ../src/Utilities/Utilities.cpp -I../src '];

% mex flags
MEXFLAGS = '-largeArrayDims -v ';

if (ispc)
    MEXFLAGS = [MEXFLAGS, 'COMPFLAGS="/openmp $COMPFLAGS"'];
end

% target dir
OUTDIR = 'mex/';

% compile
eval(['mex -output ',OUTDIR,'NLDD            NLDD_mex.cpp '          , ' ', COMMON_SRCS,' ',  SFLAGS,' ',  SLIBS,' ', MEXFLAGS ]);
eval(['mex -output ',OUTDIR,'NlBayesDenoiser NlBayesDenoiser_mex.cpp', ' ', COMMON_SRCS,' ',  SFLAGS,' ',  SLIBS,' ', MEXFLAGS ]);
eval(['mex -output ',OUTDIR,'DDIDstep        DDIDstep_mex.cpp '      , ' ', COMMON_SRCS,' ',  SFLAGS,' ',  SLIBS,' ', MEXFLAGS ]);
