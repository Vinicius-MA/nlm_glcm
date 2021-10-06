from noise_sampling import BaseImage

def main():

    indexes = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 19, 24]
    sigmaList = [10, 25, 50]
    
    imageInFolder = 'Banco de Imagens/'
    imageInNoisyFolder = 'Matlab/Resultados/Imagens Obtidas/'
    imageOutFolder = 'Python/obtidas/'
    spreadsheetFolder = 'Python/resultados/'

    filenames = [f'HW_C{x:#03d}_120.jpg' for x in indexes]

    for fname in filenames:
    
            baseImage = BaseImage( f'{fname}', sigmaList, folder=imageInFolder)

            baseImage.generate_noisy_samples(folder = imageInNoisyFolder, opt='both' )
            baseImage.generate_nlmLbp_samples(folder = imageOutFolder)
            baseImage.generate_spreadsheet( folder = spreadsheetFolder)


if __name__ == "__main__":

    main()