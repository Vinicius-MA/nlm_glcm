import numpy as np
from skimage import io
import utils
import openpyxl as xl

class ImageGroup:

    def __init__(self, filename="", samples=10, sigmaList=[25]):

        # Parameters
        self.filename = filename
        self.samples = samples
        self.sigmaList = sigmaList
        
    """
        generate_noisy_samples
            
            generate both object's noisy images matrix and image file.

            parameters:
                folder: name of folder to save images
            
    """    
    def generate_noisy_samples(self, folder=""):

        
        self.im_original = ( 255 * io.imread( self.filename, as_gray=True )
            ).astype( np.uint8 )
        
        # Generate Matrix of Noisy Images
        self.noisyImages = np.zeros(
            [ len( self.sigmaList ), self.samples,
            self.im_original.shape[0], self.im_original.shape[1] ],
            dtype=np.uint8
        )

        # Generate Noisy Images
        for k, sigma in enumerate(self.sigmaList):

            for i in range( self.samples ):

                # current noisy image file name
                fname = get_noisy_sample_filename( self.filename, sigma, i )
                
                # add Gaussian Noise
                im_noisy = utils.add_gaussian_noise( self.im_original, mean=0, sigma=sigma )
                
                # save to Class object
                self.noisyImages[k, i, :, :] = im_noisy

                # save to file
                io.imsave(folder+fname, im_noisy)

                print(">>>> created " + fname)

    def set_filename(self, newfilename):
        
        self.filename = newfilename

def get_noisy_sample_filename(filename, sigma, sample):
        
        noisyFilename = filename.replace(".",
            f"_sigma{ str( sigma ).zfill( 3 ) }_{ str( sample ).zfill( 2 ) }_noisy."
        )

        return noisyFilename

imageGroup = ImageGroup('original.png')
imageGroup.generate_noisy_samples(folder='Python/nlm-lbp/')