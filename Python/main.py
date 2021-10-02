from noise_sampling import *

def main():

    indexes = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 19, 24]
    sigmas = [10,25]
    
    imageInFolder = 'Banco de Imagens/'
    imageOutFolder = 'Python/obtidas/'
    spreadsheetFolder = 'Python/resultados/'

    filenames = [f'HW_C{x:#03d}_120.jpg' for x in indexes]

    for fname in filenames:
    
        baseImage = BaseImage( f'{fname}', sigmas, folder=imageInFolder)

        baseImage.generate_noisy_samples(folder = imageOutFolder)
        baseImage.generate_nlmLbp_samples(folder = imageOutFolder)
        baseImage.generate_spreadsheet( folder = spreadsheetFolder)


if __name__ == "__main__":

    main()