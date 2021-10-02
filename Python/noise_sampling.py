import numpy as np
from skimage import io
import nlm_lbp as nlmlbp
import utils
import openpyxl as xl
import statistics

NOISY_OUT_FNAME = "noisy"
NLMLBP_OUT_FNAME = "nlmlbp"
SPREADSHEET_HEADER = [ "Rows", "Noisy", "NLM-LBP" ]
SPREADSHEET_MEDIA = ["Média"]

class BaseImage:

    def __init__(self, filename, sigmaList, samples=10, folder=''):

        print(f'{filename} splitted: {filename.split(".") } ')
        
        # Parameters
        self.im_original = None
        self.folder = folder
        self.filename, self.extension = filename.split(".")
        self.samples = samples
        self.sigmaList = sigmaList

        self.open_original()
        self.init_matrices()
        
    """ init_matrices():
            initialize all matrices that can be used to store class data. """
    def init_matrices(self):

        ### FOR NOISY IMAGES ###
        
        # Generate Matrix of Noisy Images
        self.noisyImages = np.zeros(
            [ len( self.sigmaList ), self.samples,
            self.im_original.shape[0], self.im_original.shape[1] ],
            dtype=np.uint8
        )

        # Generate Noisy PSNR matrix
        self.psnrNoisy = np.zeros( [ len(self.sigmaList), self.samples ], dtype=np.float16 )

        ### FOR NLM LBP PROCESSED IMAGES ###
        
        # Generate Matrix of NLM-LBP Processed Images
        self.nlmLbpImages = np.zeros(
            [ len( self.sigmaList ), self.samples,
            self.im_original.shape[0], self.im_original.shape[1] ],
            dtype=np.uint8
        )

        # Generate NLM-LBP Processed PSNR matrix
        self.psnrNlmLbp = np.zeros( [ len(self.sigmaList), self.samples ], dtype=np.float16 )

    def open_original(self):
        
        self.im_original = ( 255 * io.imread( f'{self.folder}{self.filename}.{self.extension}', as_gray=True )
            ).astype( np.uint8 )
    
    """ generate_noisy_samples():
            generate both object's noisy images matrix and image file.
            - parameters:
                folder: name of folder to save images. """    
    def generate_noisy_samples( self, folder="" ) :

        if ( self.im_original is None ) :
            self.open_original()

        # Generate Noisy Images
        for ( k, sigma ) in enumerate( self.sigmaList ) :

            for i in range( self.samples ):

                # current noisy image file name
                fname = get_noisy_sample_filename( f'{self.filename}.{self.extension}', sigma, i+1 )
                
                # add Gaussian Noise
                im_noisy = utils.add_gaussian_noise( self.im_original, mean=0, sigma=sigma )
                
                # save to Class object
                self.noisyImages[k, i, :, :] = im_noisy

                # save to file
                io.imsave(folder+fname, im_noisy)

                # calculate psnr
                psnr = utils.calculate_psnr( self.im_original, im_noisy )
                self.psnrNoisy[k, i] = psnr

                print( f">>>> created {folder + fname} - psnr: {psnr:#.03f}" )

    def generate_nlmLbp_samples(self, window_radius = 10, path_radius = 6, 
        lbp_method = 'uniform', lbp_n_points = 16, lbp_radius = 2, folder="" ):

        if ( self.im_original is None ) :
            self.open_original()

        # Generate NLM LBP Images
        for ( k, sigma ) in enumerate( self.sigmaList ) :

            for i in range( self.samples ):

                # current noisy image file name
                fname = get_nlmlbp_sample_filename( f'{self.filename}.{self.extension}', sigma, i+1 )
                
                # recover Noisy Image
                im_noisy = self.noisyImages[k, i, :, :]
                
                # process current noisy image
                self.nlmLbpImages[k, i, :, :] = (
                    nlmlbp.nonlocal_means_lbp_original(
                        im_noisy, window_radius, path_radius, sigma,
                        lbp_method, lbp_n_points, lbp_radius
                    )
                )

                # save to file
                io.imsave(folder+fname, self.nlmLbpImages[k, i, :, :] )

                # calculate psnr
                psnr = utils.calculate_psnr( self.im_original, im_noisy )
                self.nlmLbpImages[k, i] = psnr

                print( f">>>> created {folder + fname} - psnr: {psnr:#.03f}" )

    def generate_spreadsheet(self, fname=None, folder=""):

        if fname is None:
            fname = self.filename
        
        workbook = xl.Workbook()

        for ( k, sigma ) in enumerate( self.sigmaList ):
            
            currSheetname = f'sigma_{sigma:#03d}'

            # prepare workbook
            workbook.create_sheet( currSheetname )
            currSheet = workbook[currSheetname]
            currSheet.append( SPREADSHEET_HEADER )

            for row in range(self.samples):

                currRow = [ row, self.psnrNoisy, self.psnrNlmLbp ]
                currSheet.append( currRow )

                print( f">>>> added to spreadsheet \n\t{currRow}" )
            
            currSheet.append(
                [
                    SPREADSHEET_MEDIA,
                    statistics.mean( self.psnrNoisy[k, :] ),
                    statistics.mean( self.psnrNlmLbp[k, :] )
                ]
            )

            print( f">>>> added to spreadsheet \n\t{currRow}" )

            outname = f'{folder}{fname}.xlsx'
            workbook.save( outname )

            print( f">>>> file saved: {outname}")

    def set_filename(self, newfilename):
        
        self.filename = newfilename

def get_noisy_sample_filename(filename, sigma, sample):
        return _get_sample_filename(filename, sigma, sample, NOISY_OUT_FNAME )

def get_nlmlbp_sample_filename( filename, sigma, sample ):
    return _get_sample_filename(filename, sigma, sample, NLMLBP_OUT_FNAME )

""" _get_sample_filename()
        Creates a new filename string adding the input information onto the
         filename string.
        - parameters:
            filename: original filename, used as base for output
            sigma: sigma to be inserted onto the output string;
            sample: index of the sampled file that will receive this function's
             output string;
            endStr: final part of the output that indicates what type of image
             is that (e.g 'noisy', 'nlmlnp'). """
def _get_sample_filename(filename, sigma, sample, endStr="" ):

    noisyFilename = filename.replace(".",
            f"_sigma{ str( sigma ).zfill( 3 ) }_{ str( sample ).zfill( 2 ) }_noisy."
        )

    return noisyFilename