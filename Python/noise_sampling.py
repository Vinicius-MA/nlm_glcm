"""
    noise_sampling.py
    ------------
    Description
"""

import statistics
import time
from os.path import exists

import numpy as np
import openpyxl as xl
from skimage import io

import utils
from nlm_glcm import nlm_glcm_filter, Props
from nlm_lbp import nonlocal_means_lbp_original as nlm_lbp_filter

from skimage.restoration import denoise_nl_means
from skimage.metrics import structural_similarity as ssim

ORIGINAL = "original"
NOISY_OUT_FNAME = "noisy"
BM3D_OUT_FNAME = "bm3d"
DA3D_OUT_FNAME = "da3d"
DDID_OUT_FNAME = "ddid"
NLDD_OUT_FNAME = "nldd"
NLM_OUT_FNAME = "nlm"       # fast non-local means
NLM_LBP_OUT_FNAME = "nlmlbp"
NLM_GLCM_OUT_FNAME = "nlmglcm"

FILTERS_IN_PYTHON = 2
DISCONSIDER_IN_OUT = 2
OUT_FNAMES = [
    ORIGINAL,
    NOISY_OUT_FNAME, BM3D_OUT_FNAME, DA3D_OUT_FNAME, DDID_OUT_FNAME,
    NLDD_OUT_FNAME, NLM_OUT_FNAME, NLM_LBP_OUT_FNAME, NLM_GLCM_OUT_FNAME
]

TERMINAL_OUT_CREATED_FILE = "created"
TERMINAL_OUT_OPENED_FILE = "opened"
SPREADSHEET_HEADER = [ "Rows"]
SPREADSHEET_HEADER.extend( OUT_FNAMES[i] for i in range(1, len(OUT_FNAMES)) )
SPREADSHEET_MEDIA = "Average"
SPREADSHEET_IMPROV = "Improvement"

class BaseImage:
    """
        Description
    """

    def __init__(self, filename, sigma_list, samples=10, folder="", out_ext="tif") :
        # Class Objects
        self.im_original = None
        self.folder = folder
        self.filename, self.in_ext = filename.split(".")
        self.out_ext = out_ext.split(".")[-1]
        self.samples = samples
        self.sigma_list = sigma_list
        self.noisy_images = None
        self.nlmlbp_images = None
        self.nlmglcm_images = None
        self.nlmlbp_psnr = None
        self.nlmglcm_psnr = None

    def open_original(self):
        """
            Description
        """
        self.im_original = _imread( f'{self.folder}{self.filename}.{self.in_ext}' )

    def generate_noisy_samples( self, folder="" ) :
        """
            generate_noisy_samples():
             generate both object's noisy images matrix and image file.
                - parameters:
                 folder: name of folder to save images.
        """

        print( ">>>> generate_noisy_samples" )
        # open im_original if doesn't exists
        if self.im_original is None :
            self.open_original()
        # Initalize Matrix of Noisy Images
        self.noisy_images = np.zeros(
            [ len( self.sigma_list ), self.samples,
            self.im_original.shape[0], self.im_original.shape[1] ],
            dtype=np.uint8 )
        # Initialize Noisy PSNR matrix
        self.noisy_psnr = np.zeros( [ len(self.sigma_list), self.samples ], dtype=np.float64 )
        # Generate Noisy Images
        for ( k, sigma ) in enumerate( self.sigma_list ) :
            for i in range( self.samples ):
                # current noisy image file name
                fname = get_noisy_sample_filename( f"{self.filename}.{self.out_ext}", sigma, i )
                full_path = f"{folder}{fname}"
                # read file if already exists
                if exists( full_path ) :
                    print_str = TERMINAL_OUT_OPENED_FILE
                    im_noisy = _imread( full_path )
                # else add gaussian noise and save image
                else:
                    print_str = TERMINAL_OUT_CREATED_FILE
                    # add Gaussian Noise
                    im_noisy = utils.add_gaussian_noise( self.im_original, mean=0, sigma=sigma )
                    # save to file
                    io.imsave( full_path, im_noisy )
                # save to Class object
                self.noisy_images[k, i, :, :] = im_noisy
                # calculate psnr
                psnr = utils.calculate_psnr( self.im_original, im_noisy )
                # print out
                print( f"\t{print_str} {folder + fname}\tpsnr: {psnr:#.04f}" )

    def generate_nlm_samples(self, window_radius=10, patch_radius=3, folder="", fast_mode=True ):
        """
            Description
        """

        self._generate_filter_samples( filter_name=NLM_OUT_FNAME, folder=folder,
            nlm_fast=fast_mode, window_radius=window_radius, patch_radius=patch_radius )

    def generate_nlm_lbp_samples(self, window_radius=10, patch_radius=6,
         lbp_method="uniform", lbp_n_points=16, lbp_radius=2, folder=""
        ):
        """
            Description
        """

        self._generate_filter_samples( filter_name=NLM_LBP_OUT_FNAME, folder=folder,
            window_radius=window_radius, patch_radius=patch_radius,
            lbp_method=lbp_method, lbp_n_points=lbp_n_points, lbp_radius=lbp_radius )

    def generate_nlm_glcm_samples(self, window_radius=10, patch_radius=6, distances=[7],
         angles=[3*np.pi/4], levels=256, props=Props.best(), symmetric=True, folder=""
        ):
        """
            Description
        """

        self._generate_filter_samples( filter_name=NLM_GLCM_OUT_FNAME, folder=folder,
            window_radius=window_radius, patch_radius=patch_radius,
            glcm_distances=distances, glcm_angles=angles, glcm_levels=levels,
            glcm_props=props, glcm_symmetric=symmetric
        )

    def generate_slices( self, out_folder="", origin=(0,0), shape=(150,150),
         start_str="detail", sample=0, in_folder=""
        ):
        """
        Description
        """

        # interact through filter out names
        for filter_name in OUT_FNAMES:
            self._generate_filter_slices(out_folder=out_folder, filter_name=filter_name,
            origin=origin, shape=shape, start_str=start_str, sample=sample, in_folder=in_folder
        )

    def generate_spreadsheet( self, fname=None, sheet_folder="", image_folder="" ):
        """
            Description
        """

        # print out
        print( '>>>> generate_spreadsheet:')
        # check if fname exists
        if fname is None:
            fname = self.filename
        # create workbook object
        workbook = xl.Workbook()
        # PSNR LOOP
        for ( k, sigma ) in enumerate( self.sigma_list ):
            # current sheet name
            curr_sheet_name = f"sigma_{sigma:#03d}"
            # prepare workbook
            workbook.create_sheet( curr_sheet_name )
            curr_sheet = workbook[curr_sheet_name]
            # Fullfill PSNR and SSIM values
            curr_sheet.append( "PSNR [dB]" )
            curr_sheet.append( SPREADSHEET_HEADER)
            for row in range( self.samples ):
                # declare current image as noisy image and calculates current psnr
                curr_image = self.noisy_images[k, row, :, :]
                curr_psnr = utils.calculate_psnr( self.im_original, curr_image )
                # current row receives row number and current psnr
                curr_row = [ row+1, curr_psnr ]
                # get filter images, if exists
                filter_images = np.zeros( [len(OUT_FNAMES)-FILTERS_IN_PYTHON-DISCONSIDER_IN_OUT,
                    self.im_original.shape[0], self.im_original.shape[1] ], dtype=np.uint8 )
                for i in range(DISCONSIDER_IN_OUT, len(OUT_FNAMES)-FILTERS_IN_PYTHON ):
                    # current image full path
                    fullpath = image_folder + _get_sample_filename(
                        f"{self.filename}.{self.out_ext}", sigma, row, OUT_FNAMES[i] )
                    # open it if exists
                    if exists( fullpath ):
                        filter_images[i-DISCONSIDER_IN_OUT-1, :, :] = _imread( fullpath )
                    elif exists( fullpath.replace(self.out_ext, self.in_ext ) ):
                        fullpath = fullpath.replace(self.out_ext, self.in_ext )
                        filter_images[i-DISCONSIDER_IN_OUT-1, :, :] = _imread( fullpath )
                    # else, consider it False
                    else:
                        filter_images[i-DISCONSIDER_IN_OUT-1, :, :] = False
                    # declare current image, calculates psnr and ssim (if exists image)
                    curr_image = filter_images[i-DISCONSIDER_IN_OUT-1, :, :]
                    curr_psnr = ( utils.calculate_psnr( self.im_original, curr_image )
                        if ( True in ( curr_image > 0 ) )
                        else -1 )
                    curr_ssim = ( ssim( self.im_original, curr_image,
                        data_range=np.amax(curr_image) - np.amin(curr_image) )
                        if (True in (curr_image > 0) )
                        else -1 )
                    # append current psnr to current row
                    curr_row.append( curr_psnr )
                # adding NLM-LBP and NLM-GLCM to current row
                curr_row.extend( [ self.nlmlbp_psnr[k, row], self.nlmglcm_psnr[k, row] ] )
                # appending current row to current sheet
                curr_sheet.append( curr_row )
        # SSIM LOOP
        for ( k, sigma ) in enumerate( self.sigma_list ):
            curr_sheet_name = f'sigma_{sigma:#03d}'
            curr_sheet = workbook[curr_sheet_name]
            #   SSIM
            curr_sheet.append([""])
            curr_sheet.append(["SSIM"])
            curr_sheet.append( SPREADSHEET_HEADER )
            for row in range( self.samples ):            
                # sample number and noisy psnr to current row
                curr_image = self.noisy_images[k, row, :, :]
                curr_ssim = ssim( self.im_original, curr_image, data_range= np.amax(curr_image)-np.amin(curr_image) )
                
                curr_row = [ row+1, curr_ssim]

                print( f"\tNoisy image:\t\t\t\t\t\t\tpsnr: {self.noisy_psnr[k, row]:#.04f}\tssim: {curr_ssim:#.04f}")

                # get other filter images, if exists
                filter_images = np.zeros( [len(OUT_FNAMES)-DISCONSIDER_IN_OUT,
                    self.im_original.shape[0], self.im_original.shape[1] ],
                    dtype=np.uint8
                )
                for i in range(DISCONSIDER_IN_OUT, len(OUT_FNAMES) ):
                    fullpath = (
                        image_folder + _get_sample_filename( f"{self.filename}.{self.out_ext}",
                        sigma, row, OUT_FNAMES[i] ) )
                    # open it if exists
                    if exists( fullpath ):
                        filter_images[i-DISCONSIDER_IN_OUT-1, :, :] = _imread( fullpath )
                        print( f"\timread: {fullpath}", end="\t")
                    elif exists( fullpath.replace(self.out_ext, self.in_ext ) ):
                        fullpath = fullpath.replace(self.out_ext, self.in_ext )
                        filter_images[i-DISCONSIDER_IN_OUT-1, :, :] = _imread( fullpath )
                        print( f"\timread: {fullpath}", end="\t")
                    # else, consider it False
                    else:
                        filter_images[i-DISCONSIDER_IN_OUT-1, :, :] = False
                        print( f"\tskipped: {fullpath}", end="\t")
                    # declare current image, calculates psnr and ssim (if exists image)
                    curr_image = filter_images[i-DISCONSIDER_IN_OUT-1, :, :]
                    curr_ssim = ( ssim( self.im_original, curr_image, data_range= np.amax(curr_image)-np.amin(curr_image))
                        if ( True in ( curr_image > 0 ) )
                        else -1 )
                    curr_psnr = ( utils.calculate_psnr( self.im_original, curr_image )
                        if ( True in ( curr_image > 0 ) )
                        else -1 )
                    print( f"\tpsnr :{curr_psnr:#.04f}\tssim: {curr_ssim:#.04f}")
                    # append current ssim to current row
                    curr_row.append( curr_ssim )
                # appending current row to current sheet
                curr_sheet.append( curr_row )
            # saving workbook to file
            outname = f'{sheet_folder}{fname}.ods'
            workbook.save( outname )
        # print out
        print( f"\tfile saved: {outname}")

    def set_filename(self, newfilename):

        self.filename = newfilename

    def _generate_filter_slices(self, out_folder="", filter_name=NLM_GLCM_OUT_FNAME, 
         origin=(0,0), shape=(150,150), start_str="detail", sample=0, in_folder=""
        ):
        """ Execute after filtering is done """

        print( f">>>> _generate_filter_slices: {filter_name}, origin={origin}, shape={shape}, sample={sample}" )

        if ( self.im_original is None ):
            self.open_original()

        if ( filter_name == ORIGINAL):
            images = np.zeros(
                [1, 1, self.im_original.shape[0], self.im_original.shape[1]],
                dtype=np.uint8
            )
            images[ 0, 0, :, :] = self.im_original
            sample=0
        elif ( filter_name == NLM_GLCM_OUT_FNAME ):
            images = self.nlmGlcmImages
        elif ( filter_name == NLM_LBP_OUT_FNAME ):
            images = self.nlmlbp_images
        elif ( filter_name == NOISY_OUT_FNAME ):
            images = self.noisy_images
        else:
            images = np.zeros( [len(self.sigma_list), 1,
                self.im_original.shape[0], self.im_original.shape[1] ], dtype=np.uint8
            )

            for (k, sigma ) in enumerate( self.sigma_list ):
                
                full_path = in_folder + _get_sample_filename(
                    f"{self.filename}.{self.in_ext}", sigma, sample, filter_name
                )

                if not exists( full_path ):
                    print( f"\tskipping {full_path}!")
                    continue
                
                images[ k, 0, :, :] = _imread( full_path )

            sample=0

        # define slices dimensions
        ( y0, x0 ) = origin
        ( dy, dx ) = shape

        # generate matrix of slices
        if ( filter_name == ORIGINAL ):
            slices = np.zeros([ 1, dy, dx ], dtype=np.uint8 )
        else:
            slices = np.zeros([ len(self.sigma_list), dy, dx ], dtype=np.uint8 )

        for ( k, sigma ) in enumerate( self.sigma_list ):
            
            if ( filter_name == ORIGINAL ):
                slices[ 0, :, :] = images[0, 0, y0:y0+dy, x0 : x0+dx]
            else:
                slices[ k, :, : ] = images[ k, sample, y0: y0+dy , x0 : x0+dx ]
            
            fname = _get_slice_filename(
                f"{self.filename}.{self.out_ext}",sigma, sample, start_str, filter_name
            )

            # skip if slice already exists
            if ( exists( out_folder + fname) ):
                print( f"\tslice exists: {out_folder + fname}" )
                continue

            # save image to file
            if (filter_name == ORIGINAL ):
                io.imsave( out_folder+fname, slices[0,:,:])
            else:
                io.imsave( out_folder+fname, slices[k,:,:])

            print( f"\tslice saved: {out_folder + fname} shape:( {slices[0,:,:].shape[0]}, {slices[0,:,:].shape[1]} )")

    def _generate_filter_samples(self, folder="", window_radius=10, patch_radius=6,
         filter_name=NLM_GLCM_OUT_FNAME, nlm_fast=True,
         lbp_method='uniform', lbp_n_points=16, lbp_radius=2,
         glcm_distances=[10], glcm_angles=[0], glcm_levels=256, glcm_props=Props.all(),
         glcm_symmetric=True, as_object=True
        ):

        print( f">>>> _generate_filter_samples: {filter_name}, {window_radius}, {patch_radius}" )

        if self.im_original is None :
            self.open_original()

        # Generate Matrix of Processed Images
        images = np.zeros( [ len( self.sigma_list ), self.samples,
            self.im_original.shape[0], self.im_original.shape[1] ],
            dtype=np.uint8
        )

        # Generate Processed PSNR matrix
        psnrs = np.zeros(
            [ len(self.sigma_list), self.samples ], dtype=np.float64
        )

        sigma_time = 0

        # Generate Processed Images
        for ( k, sigma ) in enumerate( self.sigma_list ) :

            sample_time = 0

            for i in range( self.samples ):

                # current processed image file name (sample is passed as matriz index, not filenumbering)
                fname = _get_sample_filename(f'{self.filename}.{self.out_ext}', sigma, i, filter_name)
                full_path = f'{folder}{fname}'

                if( not( exists( full_path) ) ):

                    print_str = TERMINAL_OUT_CREATED_FILE
                    
                    # recover Noisy Image
                    im_noisy = self.noisy_images[k, i, :, :]

                    start_time = time.time()
                    
                    if ( filter_name == NLM_GLCM_OUT_FNAME):
                        print( f"\t{glcm_distances}\t{utils.list2str(glcm_angles)},{glcm_props},{glcm_symmetric}")
                        im_proc =  (
                            nlm_glcm_filter(im_noisy, window_radius, patch_radius, sigma,
                                glcm_distances, glcm_angles, glcm_levels,
                                glcm_props, glcm_symmetric )
                        )
                    elif ( filter_name == NLM_LBP_OUT_FNAME ):
                        im_proc =  (
                            nlm_lbp_filter( im_noisy, window_radius, patch_radius, sigma,
                                lbp_method, lbp_n_points, lbp_radius )
                        )
                    elif ( filter_name == NLM_OUT_FNAME ):
                        im_proc = (
                            denoise_nl_means( im_noisy, patch_size=patch_radius,
                                patch_distance=window_radius, h=sigma,
                                fast_mode=nlm_fast, preserve_range=True
                            )
                        )

                    diff = time.time() - start_time

                    # save to file
                    io.imsave( full_path, im_proc.astype( np.uint8 ) )
                    
                else:

                    print_str = TERMINAL_OUT_OPENED_FILE

                    im_proc = io.imread( full_path )
                        
                # save to Class object
                images[k, i, :, :] = im_proc 

                # calculate psnr
                psnr = utils.calculate_psnr( self.im_original, im_proc )
                psnrs[k, i] = psnr

                if 'diff' in locals():
                    
                    sample_time += diff
                    sigma_time += diff

                    print( f"\t{print_str} {folder + fname}\tpsnr: {psnr:#.04f}" + 
                        f" - time: { int( diff // 60 ):#02d}:{ int( diff % 60 ):#02d}"
                    )
                else:
                    print( f"\t{print_str} {folder + fname}\tpsnr: {psnr:#.04f}" )

                if (filter_name == NLM_GLCM_OUT_FNAME and as_object ):
                    self.nlmGlcmImages = images
                    self.nlmglcm_psnr = psnrs

                elif ( filter_name == NLM_LBP_OUT_FNAME  and as_object ):
                    self.nlmlbp_images = images
                    self.nlmlbp_psnr = psnrs


            print( f"\t\ttotal sample time: " +
                f"{ int( sample_time//60 ):#02d}:{ int( sample_time % 60 ):#02d}"
            )

        print( f"\t\t\ttotal sigma time: " +
            f"{ int( sigma_time // 60 ):#02d}:{ int( sigma_time % 60 ):#02d}"
        )

def get_noisy_sample_filename ( filename, sigma, sample ):
        return _get_sample_filename( filename, sigma, sample, NOISY_OUT_FNAME )

def get_nlm_lbp_sample_filename( filename, sigma, sample ):
    return _get_sample_filename( filename, sigma, sample, NLM_LBP_OUT_FNAME )

def get_nlm_glcm_sample_filename( filename, sigma, sample ):
    return _get_sample_filename(filename, sigma, sample, NLM_GLCM_OUT_FNAME)

def get_noisy_slice_filename( filename, sigma, start_str="detail" ):
    return _get_slice_filename( filename, sigma, start_str, NOISY_OUT_FNAME )

def get_nlm_lbp_slice_filename( filename, sigma, start_str="detail" ):    
    return _get_slice_filename( filename, sigma, start_str, NLM_LBP_OUT_FNAME )

def get_nlm_glcm_slice_filename( filename, sigma, start_str="detail"):
    return _get_slice_filename(filename, sigma, start_str, NLM_GLCM_OUT_FNAME )

def _get_slice_filename( filename, sigma, sample=0, start_str="detail", endStr="" ):
    """  """

    return _get_sample_filename( filename, sigma, sample, endStr, start_str+"_", False )

def _get_sample_filename(filename, sigma, sample, endStr="", start_str="", bypassOriginal=True ):
    """ _get_sample_filename()
        Creates a new filename string adding the input information onto the
         filename string.
        - parameters:
            filename: original filename, used as base for output
            sigma: sigma to be inserted onto the output string;
            sample: index of the sampled file that will receive this function's
             output string 
             (IMPORTANT: must pass the python matrix index (starts with 0, ends
              with n-1), not the file-related index (starts with 1,ends with n
              ). The conversion to filename is made inside this function);
            endStr: final part of the output that indicates what type of image
             is that (e.g 'noisy', 'nlmlbp')."""

    if endStr == ORIGINAL and bypassOriginal:
        return filename
    
    return start_str + filename.replace(".",
            f"_sigma{sigma:#03d}_{sample+1:#02d}_{endStr}."
        )

def _imread( full_path ):

    image = io.imread( full_path, as_gray=True )

    if( image.dtype != np.uint8):
        
        image = (255 * image).astype( np.uint8 )

    return image
