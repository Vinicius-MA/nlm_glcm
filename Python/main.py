from noise_sampling import *

def main():

    filename = 'HW_C001_120.jpg'
    
    baseImage = BaseImage( f'{filename}', [10, 25, 50], folder='Banco de Imagens/')
    baseImage.generate_noisy_samples(folder='Python/resultados/')
    baseImage.generate_nlmLbp_samples(folder='Python/resultados/')
    baseImage.generate_spreadsheet()


if __name__ == "__main__":

    main()