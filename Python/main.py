import time

from noise_sampling import BaseImage


def main():

    indexes = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 19, 24]
    sigmaList = [10, 25, 50]
    samples = 10

    imageInFolder = "image-database/"
    imageNoisyFolder = "images-noisy/"
    imageOutFolder = "images-output/"
    slicesOutFolder = "slices/"
    spreadsheetFolder = "results-partial/"

    filenames = [f'HW_C{x:#03d}_120.jpg' for x in indexes]

    for fname in filenames:
    
            baseImage = BaseImage( filename=fname , sigmaList=sigmaList, samples=samples, folder=imageInFolder )
            baseImage.generate_noisy_samples( folder=imageNoisyFolder )

            # execute NLM filtering
            baseImage.generate_nlm_samples( folder=imageOutFolder )
            
            # execute NLM-LBP filtering
            baseImage.generate_nlm_lbp_samples( folder=imageOutFolder )
            
            # execute NLM_GLCM filtering
            baseImage.generate_nlm_glcm_samples( folder=imageOutFolder )
            
            # save data to spreadsheet
            baseImage.generate_spreadsheet( folder=spreadsheetFolder, imageFolder=imageOutFolder )
            
            # generate slices
            baseImage.generate_slices(outFolder=slicesOutFolder, inFolder=imageOutFolder)

if __name__ == "__main__":

    main()
