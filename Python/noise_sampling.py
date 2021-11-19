import statistics
import time
from os.path import exists

import numpy as np
import openpyxl as xl
from skimage import io

import utils
from glcm_properties import Props
from nlm_glcm import nlm_glcm_filter
from nlm_lbp import nonlocal_means_lbp_original as nlm_lbp_filter

from skimage.restoration import denoise_nl_means

ORIGINAL = "original"
NOISY_OUT_FNAME = "noisy"
BM3D_OUT_FNAME = "bm3d"
DA3D_OUT_FNAME = "da3d"
DDID_OUT_FNAME = "ddid"
NLDD_OUT_FNAME = "nldd"
NLM_OUT_FNAME = "nlm"
NLM_LBP_OUT_FNAME = "nlmlbp"
NLM_GLCM_OUT_FNAME = "nlmglcm"

FILTERS_IN_PYTHON = 2
DISCONSIDER_IN_OUT = 2
out_fname = [
    ORIGINAL,
    NOISY_OUT_FNAME, BM3D_OUT_FNAME, DA3D_OUT_FNAME, DDID_OUT_FNAME,
    NLDD_OUT_FNAME, NLM_OUT_FNAME, NLM_LBP_OUT_FNAME, NLM_GLCM_OUT_FNAME
]

TERMINAL_OUT_CREATED_FILE = "created"
TERMINAL_OUT_OPENED_FILE = "opened"
SPREADSHEET_HEADER = [ "Rows"]
SPREADSHEET_HEADER.extend( out_fname[i] for i in range(1, len(out_fname)) )
SPREADSHEET_MEDIA = "Average"

class BaseImage:

    def __init__(self, filename, sigmaList, samples=10, folder='') :
        
        # Class' Objects
        self.im_original = None
        self.folder = folder
        self.filename, self.extension = filename.split(".")
        self.samples = samples
        self.sigmaList = sigmaList
        self.noisyImages = None
        self.noisyPsnr = None
        self.nlmLbpImages = None
        self.nlmLbpPsnr = None
        self.nlmGlcmImages = None
        self.nlmGlcmPsnr = None

    def open_original(self):
        
        self.im_original = _imread( 
                f'{self.folder}{self.filename}.{self.extension}'
            )
  
    def generate_noisy_samples( self, folder="" ) :
        """ generate_noisy_samples():
            generate both object's noisy images matrix and image file.
            - parameters:
                folder: name of folder to save images. """
        
        if ( self.im_original is None ) :
            self.open_original()
        
        # Initalize Matrix of Noisy Images
        self.noisyImages = np.zeros(
            [ len( self.sigmaList ), self.samples,
            self.im_original.shape[0], self.im_original.shape[1] ],
            dtype=np.uint8
        )

        # Initialize Noisy PSNR matrix
        self.noisyPsnr = np.zeros(
            [ len(self.sigmaList), self.samples ], dtype=np.float64
        )

        # Generate Noisy Images
        for ( k, sigma ) in enumerate( self.sigmaList ) :

            for i in range( self.samples ):

                # current noisy image file name
                fname = get_noisy_sample_filename( f'{self.filename}.{self.extension}', sigma, i+1 )
                fullFilePath = f'{folder}{fname}'
                
                if (  not( exists( fullFilePath )  ) ):
                    
                    printStr = TERMINAL_OUT_CREATED_FILE
                    
                    # add Gaussian Noise
                    im_noisy = utils.add_gaussian_noise( self.im_original, mean=0, sigma=sigma )

                    # save to file
                    io.imsave(folder+fname, im_noisy)                    

                else:

                    printStr = TERMINAL_OUT_OPENED_FILE

                    im_noisy = _imread( fullFilePath )
                
                # save to Class object
                self.noisyImages[k, i, :, :] = im_noisy                

                # calculate psnr
                psnr = utils.calculate_psnr( self.im_original, im_noisy )
                self.noisyPsnr[k, i] = psnr

                print( f">>>> {printStr} {folder + fname} - psnr: {psnr:#.03f}" )

    def generate_nlm_samples(self, window_radius=10, patch_radius=3, folder="" ):

        self._generate_filter_samples( filterName=NLM_OUT_FNAME, folder=folder,
            window_radius=window_radius, patch_radius=patch_radius
        )

    def generate_nlm_lbp_samples(self, window_radius = 10, patch_radius = 6, 
        lbp_method = 'uniform', lbp_n_points = 16, lbp_radius = 2, folder="" ):

        self._generate_filter_samples( filterName=NLM_LBP_OUT_FNAME, folder=folder,
            window_radius=window_radius, patch_radius=patch_radius,
            lbp_method=lbp_method, lbp_n_points=lbp_n_points, lbp_radius=lbp_radius
        )

    def generate_nlm_glcm_samples(self, window_radius = 10, patch_radius = 6, 
        distances = [10], angles = [0], levels=256, props=Props.all(),
        symmetric=True, folder="" ):

        self._generate_filter_samples( filterName=NLM_GLCM_OUT_FNAME, folder=folder,
            window_radius=window_radius, patch_radius=patch_radius,
            glcm_distances=distances, glcm_angles=angles, glcm_levels=levels,
            glcm_props=props, glcm_symmetric=symmetric
        )

    def generate_slices( self, outFolder="", origin=(0,0), shape=(100,100),
         startStr="detail", sample=1, inFolder=""
        ):

        for filterName in out_fname:
            
            self._generate_filter_slices(outFolder=outFolder, filterName=filterName,
            origin=origin, shape=shape, startStr=startStr, sample=sample, inFolder=inFolder
        )
        
    def generate_spreadsheet(self, fname=None, folder="", imageFolder=""):

        print( f'>>>> generate_spreadsheet:')
        
        if fname is None:
            fname = self.filename
        
        workbook = xl.Workbook()

        # mean header to final row
        meanRow = [ SPREADSHEET_MEDIA ]

        for ( k, sigma ) in enumerate( self.sigmaList ):
            
            currSheetname = f'sigma_{sigma:#03d}'

            # prepare workbook
            workbook.create_sheet( currSheetname )
            currSheet = workbook[currSheetname]
            currSheet.append( SPREADSHEET_HEADER )

            # initialize list of psnr means
            psnr_means = (len(out_fname)- DISCONSIDER_IN_OUT + 1 )*[0.]

            for row in range( self.samples ):
               
                # sample number and noisy psnr to current row
                currRow = [ row+1, self.noisyPsnr[k, row] ]
                # noisy psnr portion to psnr means list
                psnr_means[0] += self.noisyPsnr[k, row] / self.samples

                # get other filter images, if exists
                Y_others = np.zeros( [len(out_fname)-FILTERS_IN_PYTHON-DISCONSIDER_IN_OUT,
                    self.im_original.shape[0], self.im_original.shape[1] ],
                    dtype=np.uint8
                )
                for i in range(DISCONSIDER_IN_OUT, len(out_fname)-FILTERS_IN_PYTHON ):
                    filter_fullpath_file = (
                        imageFolder + _get_sample_filename( f"{self.filename}.{self.extension}",
                            sigma, row+1, out_fname[i]
                        )
                    )
                    if exists( filter_fullpath_file ):
                        Y_others[i-DISCONSIDER_IN_OUT, :, : ] = _imread( filter_fullpath_file )
                    else:
                        Y_others[i-DISCONSIDER_IN_OUT, :, :] = False
                    currPsnr = ( utils.calculate_psnr( self.im_original, Y_others[i-DISCONSIDER_IN_OUT,:,:] )
                        if ( True in ( Y_others[i-DISCONSIDER_IN_OUT, :, : ] > 0 ) )
                        else 0
                    )
                    currRow.append( currPsnr )
                    # mean update
                    psnr_means[i-DISCONSIDER_IN_OUT+1] += currPsnr / self.samples
                # adding NLM-LBP and NLM-GLCM to current row
                currRow.extend( [ self.nlmLbpPsnr[k, row], self.nlmGlcmPsnr[k, row] ] )                
                # appending current row to current sheet
                currSheet.append( currRow )

            # NLM-LBP and NLM-GLCM means to psnr_means
            psnr_means[ - FILTERS_IN_PYTHON : ] = [
                statistics.mean( self.nlmLbpPsnr[k, :] ),
                statistics.mean( self.nlmGlcmPsnr[k, :] )
            ]
            
            # average row extend filter means
            meanRow.extend( psnr_means )
            
            currSheet.append( meanRow )
            print( f"\PSNR mean:\t{meanRow}" )

            # saving workbook to file
            outname = f'{folder}{fname}.xlsx'
            workbook.save( outname )
            print( f">>>> file saved: {folder}{outname}")

    def set_filename(self, newfilename):
        
        self.filename = newfilename
    
    def _generate_filter_slices(self, outFolder="", filterName=NLM_GLCM_OUT_FNAME, 
         origin=(0,0), shape=(100,100), startStr="detail", sample=0, inFolder=""
        ):
        """ Execute after filtering is done """
        
        if ( self.im_original is None ):
            self.open_original()

        if ( filterName == ORIGINAL):
            images = np.zeros(
                [1, 1, self.im_original.shape[0], self.im_original.shape[1]],
                dtype=np.uint8
            )
            images[ 0, 0, :, :] = self.im_original
            sample=0
        elif ( filterName == NLM_GLCM_OUT_FNAME ):
            images = self.nlmGlcmImages
        elif ( filterName == NLM_LBP_OUT_FNAME ):
            images = self.nlmLbpImages
        elif ( filterName == NOISY_OUT_FNAME ):
            images = self.noisyImages
        else:
            images = np.zeros( [len(self.sigmaList), 1,
                self.im_original.shape[0], self.im_original.shape[1] ], dtype=np.uint8
            )

            for (k, sigma ) in enumerate( self.sigmaList ):
                
                fullFilePath = inFolder + _get_sample_filename(
                    f"{self.filename}.{self.extension}", sigma, sample, filterName
                )

                if not exists( fullFilePath ):
                    continue
                
                images[ k, 0, :, :] = _imread( fullFilePath )

                print( f">>>> opened: {fullFilePath}" )

            sample=0

        # define slices dimensions
        ( y0, x0 ) = origin
        ( dy, dx ) = shape

        # generate matrix of slices
        if ( filterName == ORIGINAL ):
            slices = np.zeros([ 1, dy, dx ], dtype=np.uint8 )
        else:
            slices = np.zeros([ len(self.sigmaList), dy, dx ], dtype=np.uint8 )

        for ( k, sigma ) in enumerate( self.sigmaList ):
            
            if ( filterName == ORIGINAL ):
                slices[ 0, :, :] = images[0, 0, y0:y0+dy, x0 : x0+dx]
            else:
                slices[ k, :, : ] = images[ k, sample, y0: y0+dy , x0 : x0+dx ]
            
            fname = _get_slice_filename(
                f"{self.filename}.{self.extension}", sigma, startStr, filterName
            )

            # save image to file
            if (filterName == ORIGINAL ):
                io.imsave( outFolder+fname, slices[0,:,:])
            else:
                io.imsave( outFolder+fname, slices[k,:,:])

            print( f">>>> slice saved: {outFolder + fname} shape:( {slices[0,:,:].shape[0]}, {slices[0,:,:].shape[1]} )")

    def _generate_filter_samples(self, folder="", window_radius=10, patch_radius=6, 
         filterName=NLM_GLCM_OUT_FNAME,
         lbp_method='uniform', lbp_n_points=16, lbp_radius=2,
         glcm_distances=[10], glcm_angles=[0], glcm_levels=256, glcm_props=Props.all(),
         glcm_symmetric=True, as_object=True
        ):

        if ( self.im_original is None ) :
            self.open_original()

        # Generate Matrix of Processed Images
        images = np.zeros( [ len( self.sigmaList ), self.samples, 
            self.im_original.shape[0], self.im_original.shape[1] ],
            dtype=np.uint8
        )

        # Generate Processed PSNR matrix
        psnrs = np.zeros(
            [ len(self.sigmaList), self.samples ], dtype=np.float64
        )

        sigma_time = 0

        # Generate Processed Images
        for ( k, sigma ) in enumerate( self.sigmaList ) :

            sample_time = 0

            for i in range( self.samples ):

                # current processed image file name
                fname = _get_sample_filename(f'{self.filename}.{self.extension}', sigma, i+1, filterName)
                fullFilePath = f'{folder}{fname}'

                if( not( exists( fullFilePath) ) ):

                    printStr = TERMINAL_OUT_CREATED_FILE
                    
                    # recover Noisy Image
                    im_noisy = self.noisyImages[k, i, :, :]

                    start_time = time.time()
                    
                    if ( filterName == NLM_GLCM_OUT_FNAME):
                        im_proc =  (
                            nlm_glcm_filter(im_noisy, window_radius, patch_radius, sigma,
                                glcm_distances, glcm_angles, glcm_levels,
                                glcm_props, glcm_symmetric )
                        )
                    elif ( filterName == NLM_LBP_OUT_FNAME ):
                        im_proc =  (
                            nlm_lbp_filter( im_noisy,
                                window_radius, patch_radius, sigma, lbp_method,
                                lbp_n_points, lbp_radius )
                        )
                    elif ( filterName == NLM_OUT_FNAME ):
                        im_proc = (
                            denoise_nl_means( im_noisy, patch_size=patch_radius, 
                                patch_distance=window_radius, h=sigma,
                                preserve_range=True
                            )
                        )

                    diff = time.time() - start_time

                    # save to file
                    io.imsave( fullFilePath, im_proc.astype( np.uint8 ) )
                    
                else:

                    printStr = TERMINAL_OUT_OPENED_FILE

                    im_proc = io.imread( fullFilePath )
                        
                # save to Class object
                images[k, i, :, :] = im_proc 

                # calculate psnr
                psnr = utils.calculate_psnr( self.im_original, im_proc )
                psnrs[k, i] = psnr

                if 'diff' in locals():
                    
                    sample_time += diff
                    sigma_time += diff

                    print( f">>>> {printStr} {folder + fname} - psnr: {psnr}" + 
                        f" - time: { int( diff // 60 ):#02d}:{ int( diff % 60 ):#02d}"
                    )
                else:
                    print( f">>>> {printStr} {folder + fname} - psnr: {psnr}" )

                if (filterName == NLM_GLCM_OUT_FNAME and as_object ):
                    self.nlmGlcmImages = images
                    self.nlmGlcmPsnr = psnrs

                elif ( filterName == NLM_LBP_OUT_FNAME  and as_object ):
                    self.nlmLbpImages = images
                    self.nlmLbpPsnr = psnrs


            print( f">>>>\ttotal sample time: " +
                f"{ int( sample_time//60 ):#02d}:{ int( sample_time % 60 ):#02d}"
            )

        print( f">>>>\t\ttotal sigma time: " +
            f"{ int( sigma_time // 60 ):#02d}:{ int( sigma_time % 60 ):#02d}"
        )

def get_noisy_sample_filename ( filename, sigma, sample ):
        return _get_sample_filename( filename, sigma, sample, NOISY_OUT_FNAME )

def get_nlm_lbp_sample_filename( filename, sigma, sample ):
    return _get_sample_filename( filename, sigma, sample, NLM_LBP_OUT_FNAME )

def get_nlm_glcm_sample_filename( filename, sigma, sample ):
    return _get_sample_filename(filename, sigma, sample, NLM_GLCM_OUT_FNAME)

def get_noisy_slice_filename( filename, sigma, startStr="detail" ):
    return _get_slice_filename( filename, sigma, startStr, NOISY_OUT_FNAME )

def get_nlm_lbp_slice_filename( filename, sigma, startStr="detail" ):    
    return _get_slice_filename( filename, sigma, startStr, NLM_LBP_OUT_FNAME )

def get_nlm_glcm_slice_filename( filename, sigma, startStr="detail"):
    return _get_slice_filename(filename, sigma, startStr, NLM_GLCM_OUT_FNAME )

def _get_slice_filename( filename, sigma, startStr="detail", endStr="" ):
    """  """

    return f"{startStr}_" + filename.replace(".",
        f"_sigma{sigma:#03d}_{endStr}."
    )

def _get_sample_filename(filename, sigma, sample, endStr="" ):
    """ _get_sample_filename()
        Creates a new filename string adding the input information onto the
         filename string.
        - parameters:
            filename: original filename, used as base for output
            sigma: sigma to be inserted onto the output string;
            sample: index of the sampled file that will receive this function's
             output string;
            endStr: final part of the output that indicates what type of image
             is that (e.g 'noisy', 'nlmlbp')."""

    if endStr == ORIGINAL:
        return filename
    
    return filename.replace(".",
            f"_sigma{sigma:#03d}_{sample:#02d}_{endStr}."
        )

def _imread( fullFilePath ):

    image = io.imread( fullFilePath, as_gray=True )

    if( image.dtype != np.uint8):
        
        image = (255 * image).astype( np.uint8 )

    return image
