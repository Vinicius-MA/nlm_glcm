import time

from noise_sampling import BaseImage


def main():

    indexes = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 19, 24]
    sigmaList = [10, 25, 50]
    
    imageInFolder = "Banco de Imagens/"
    imageOutFolder = "Matlab/Resultados/Imagens Obtidas/"
    slicesOutFolder = "Matlab/Resultados/Slices/"
    spreadsheetFolder = "Python/resultados/"

    filenames = [f'HW_C{x:#03d}_120.jpg' for x in indexes]

    for fname in filenames:
    
            baseImage = BaseImage( fname , sigmaList, folder=imageInFolder )
            baseImage.generate_noisy_samples( folder=imageOutFolder )
            
            # execute NLM-LBP filtering
            baseImage.generate_nlm_lbp_samples( folder=imageOutFolder )
            
            # execute NLM_GLCM filtering
            baseImage.generate_nlm_glcm_samples( folder=imageOutFolder )
            
            # save data to spreadsheet
            baseImage.generate_spreadsheet( folder=spreadsheetFolder )
            
            # generate slices
            baseImage.generate_slices(outFolder=slicesOutFolder, inFolder=imageOutFolder)

if __name__ == "__main__":

    main()
