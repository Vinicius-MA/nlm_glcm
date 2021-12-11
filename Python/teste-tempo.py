""" teste-tempo.py """

from noise_sampling import BaseImage


def main():
    """
    main(0)
    """

    indexes = [1, 4, 9, 13, 19]
    sigma_list = [25]
    samples = 1

    im_in_folder = "image-database/"
    im_noisy_folder = "images-noisy/"
    im_out_folder = "Python/testes-tempo/"
    #slices_out_folder = "slices/"
    sheet_folder = "Python/testes-tempo/"
    # generate file names
    filenames = [f'HW_C{x:#03d}_120.jpg' for x in indexes]
    # iterate through filenames
    for fname in filenames:
        # create BaseImage object
        base_image = BaseImage( filename=fname , sigma_list=sigma_list,
            samples=samples, folder=im_in_folder )
        # generate noisy samples
        base_image.generate_noisy_samples( folder=im_noisy_folder )
        # execute NLM filtering (non-Fast)
        base_image.generate_nlm_samples( folder=im_out_folder, patch_radius=6, fast_mode=False )
        # execute NLM-LBP filtering
        base_image.generate_nlm_lbp_samples( folder=im_out_folder )
        # execute NLM_GLCM filtering
        base_image.generate_nlm_glcm_samples( folder=im_out_folder )
        # save data to spreadsheet
        base_image.generate_spreadsheet( sheet_folder=sheet_folder, image_folder=im_out_folder )

if __name__ == "__main__":
    main()
