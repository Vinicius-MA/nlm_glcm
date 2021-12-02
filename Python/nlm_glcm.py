""" Non-Local Means with GLCM wighting improvement.
    Credits:
    ------------
     Vinícius Moraes Andreghetti <vinicius.andreghetti@usp.br>
    Special thanks to:
    ------------
     Gregory Petterson Zanelato <gregory.zanelato@usp.br>"""

from enum import Enum

import numpy as np
import skimage.feature as sft
from numba import njit, prange

import utils as ut


def nlm_glcm_filter(im_in, window_radius, patch_radius, sigma,
     distances, angles, levels=256, props=Props.all(), symmetric=True
    ) :
    """Performs the Non-Local Means with Gray-Level Co-Occurrence
     Matrix (GLCM) weighting improvement, on 2D grayscale images.

     Parameters:
     ---------------
     * im_in: 2D ndarray [np.uint8]
        im_in noisy image
     * window_radius: int
        Radius of the searching windows (Si) at which the pixels will
         be considered to calculate the central pixel's f_hat value.
     * patch_radius: int
        Radius of the patches (Li, Lj) that will be compared to
         compute distances.
     * sigma: float
        Estimative of the parameter sigma of the im_in's noise
         distribution.
     * distances: 1D ndarray [np.uint8]
        List of pixel pair distance offsets, in pixels, to be
         considered at the GLCM calculus.
     * angles: 1D ndarray [np.float64]
        List of pixel pair angles, in radians, to be considered at
         the GLCM calculus.
     * levels: int, optional
        The im_in image should contain integers in [0, `levels`-1],
         where levels indicate the number of gray-levels counted
         (typically 256 for an 8-bit image). This argument is
         required for 16-bit images or higher and is typically the
         maximum of the image. As the f_hat matrix is at least
         `levels` x `levels`, it might be preferable to use binning
         of the im_in image rather than large values for `levels`.
     * props: array_like [str], optional
        List of strings that indicates which GLCM properties will be
         computed at the filter. The possible values are:
            ['contrast', 'dissimilarity', 'homogeneity', 'energy',
            'correlation', 'ASM' ]
         Default value is: ['contrast']
     * symmetric: bool, optional
        If True, the GLCM matrix `G[:, :, d, theta]` is symmetric.
         This is accomplished by ignoring the order of value pairs,
         so both (i, j) and (j, i) are accumulated when (i, j) is
         encountered for a given offset. The default is False.
     * max_ram_gb: float, optional (is it correctly implemented?)
        Maximum amount of RAM memory to be available to store the
         glcm_patch array the (probably unnecessarily) biggest array
         in this algorithm. Accordingly to this value, the im_in
         image is broken into minor slices, that are processed
         independently.
     Returns:
     ---------------
     * f_hat: 2D ndarray [np.uint8]
        The final processed calculated image.
     References:
     ---------------
     [1] A. Buades, B. Coll and J. -. Morel, "A non-local algorithm
      for image denoising," 2005 IEEE Computer Society Conference on
      Computer Vision and Pattern Recognition (CVPR'05), 2005, pp.
      60-65 vol. 2, doi: 10.1109/CVPR.2005.38.
     [2] Haralick, RM.; Shanmugam, K., “Textural features for image
      classification” IEEE Transactions on systems, man, and
      cybernetics 6 (1973): 610-621. DOI:10.1109/TSMC.1973.4309314
     [3] Khellah, Fakhry. (2013). Application of Local Binary Pattern
      to Windowed Nonlocal Means Image Denoising. 8156. 21-30. 10.1007/978-3-642-41181-6_3.
     """

    print( "\tstarting nlm-glcm..." )
    # distances and angles as numpy arrays
    distances = np.array( distances, dtype=np.uint8 )
    angles = np.array( angles, dtype=np.float64 )
    # im_in image shape is (height, width)
    height = im_in.shape[0]
    width = im_in.shape[1]
    # padding im_in image and get new shape
    im_pad = np.pad( im_in,(patch_radius, patch_radius),
        mode='symmetric' ).astype( np.float64 )
    # make NLM kernel
    kernel = ut.make_kernel(patch_radius)
    # avoid zero division
    eps = 10e-7
    # alocate arrays memory
    f_hat = np.empty((height,width), dtype=np.uint8 )
    # get NLM smoothness parameter
    h_nlm = sigma*sigma
    h_glcm = h_nlm/(255*255)
    # get patch array from image
    im_patch = ut.image2patch( im_in, im_pad, patch_radius )
    # get GLCM array from patch array (calculates all glcm of all patches before calling process)
    d_patch = patch2glcm( im_in, im_patch, levels, distances, angles, props, symmetric, eps )
    # execute filtering loop
    f_hat = process( im_in, im_pad, d_patch, kernel,
        window_radius, patch_radius, h_nlm, h_glcm )
    return f_hat

@njit(nogil=True, parallel=True)
def process( im_in, im_pad, d_patch, kernel, window_radius, patch_radius, h_nlm, h_glcm ):
    """Description"""

    # input shape
    height, width = im_in.shape
    # initializing f_hat image
    f_hat=np.empty( (height, width), dtype=np.uint8 )
    # number of elements in each patch
    patch_size = (2*patch_radius) + 1
    # initialing patch windows
    w_1 = np.empty((patch_size,patch_size), dtype = np.float64)
    w_2 = np.empty_like( w_1, dtype = np.float64)
    # looping over each pixel
    for i in prange( height ):
        for j in prange( width ):
            # calculating indexing offset
            offset_y = i + patch_radius
            offset_x = j + patch_radius
            #Get NLM central patch in searching window (original im_in)
            w_1 = im_pad[
                offset_y-patch_radius:offset_y+patch_radius+1,
                offset_x-patch_radius:offset_x+patch_radius+1 ]
            #Calculate boundaries
            y_min = np.maximum( offset_y - window_radius, patch_radius )
            y_max = np.minimum( offset_y + window_radius + 1, height + patch_radius )
            x_min = np.maximum( offset_x - window_radius, patch_radius )
            x_max = np.minimum( offset_x + window_radius+1, width + patch_radius )
            # initialize variables
            wmax = 0
            average  = 0
            sweight = 0
            index_element = 0
            # number of samples in each patch
            patch_samples_size = ( (y_max - y_min )*( x_max - x_min ) ) - 1
            # NLM intensity weight vector  - 'w(i,j)intensity'
            w_i=np.zeros( (patch_samples_size), dtype=np.float64 )
            center_patches=np.zeros( (patch_samples_size), dtype=np.float64 )
            # GLCM descriptors for central patch
            d_1 = d_patch[i, j, :, :]
            # GLCM similarity weight vector - 'w(i,j)GLCM'
            w_glcm=np.zeros( (patch_samples_size), dtype=np.float64 )
            #Compare central patch with neighbors
            for r in prange( int(y_min), int(y_max) ):
                for s in prange( int(x_min), int(x_max) ):
                    # avoiding errors
                    if( r == offset_y and s == offset_x ):
                        continue
                    if( r >= im_in.shape[0] or s >= im_in.shape[1] ):
                        continue
                    # get NLM neighbor patch in search window (original im_in)
                    w_2 = im_pad[
                        r-patch_radius:r+patch_radius+1,
                        s-patch_radius:s+patch_radius+1 ]
                    #Calculate NLM distance weight
                    diff_i = np.subtract( w_1,w_2 )
                    d_i = ut.calc_distance( kernel, diff_i )
                    w_i[index_element] = ut.calc_weight( d_i, h_nlm )
                    center_patches[index_element] = im_pad[r,s]
                    # GLCM and descriptors for patch comparison
                    d_2 = d_patch[r, s, :, :]
                    # Calculate GLCM distance weight
                    diff_glcm = np.subtract(d_1, d_2)
                    d_glcm = np.sum( diff_glcm * diff_glcm)
                    w_glcm[index_element] = ut.calc_weight( d_glcm, h_glcm )
                    # increment index element
                    index_element += 1
            # central pixel receives maximum NLM weighting value
            wmax = np.max(w_i)
            # modulated weight w_m
            w_m = w_i * w_glcm
            # gets f_hat value (average of pixels in searching window)
            average = ( np.sum( w_m * center_patches) +
                wmax * im_pad[offset_y,offset_x] )
            # calculates Z(i) (normalize filter)
            sweight = np.sum(w_m) + wmax
            # avoid zero division
            if sweight > 0 :
                f_hat[i,j] = average / sweight
            else:
                f_hat[i,j] = im_in[i,j]
    return f_hat

def patch2glcm( im_in, im_patch, levels, distances, angles, props, symmetric=False, eps=10e-07 ):
    """Performs...

     Parameters:
     ---------------
     * im_in: 2D ndarray [np.uint8]
        Input original image
     * im_patch: 4D ndarray [np.uint8]
        Matrix of Patches gotten from original image
     * m: int
     * n: int
     * levels: int, optional
        The im_in image should contain integers in [0, `levels`-1],
         where levels indicate the number of gray-levels counted
         (typically 256 for an 8-bit image). This argument is
         required for 16-bit images or higher and is typically the
         maximum of the image. As the f_hat matrix is at least
         `levels` x `levels`, it might be preferable to use binning
         of the im_in image rather than large values for `levels`.
     * distances: 1D ndarray [np.uint8]
        List of pixel pair distance offsets, in pixels, to be
         considered at the GLCM calculus.
     * angles: 1D ndarray [np.float64]
        List of pixel pair angles, in radians, to be considered at
         the GLCM calculus.
     * props: array_like [str]
        List of strings that indicates which GLCM properties will be
         computed at the filter. The possible values are:
            ['contrast', 'dissimilarity', 'homogeneity', 'energy',
            'correlation', 'ASM' ]
     * symmetric: bool, optional
        If True, the GLCM matrix `G[:, :, d, theta]` is symmetric.
         This is accomplished by ignoring the order of value pairs,
         so both (i, j) and (j, i) are accumulated when (i, j) is
         encountered for a given offset. The default is False.
     Returns:
     ---------------
     * f_glcm: 4D ndarray [np.float64]
        Each element f_glcm[i,j,p,d,t] is the calculated, normalized
         value of property indexed by p, from a distance and angle
         indexed by d and a, calculated from the patch centered at
         the pixel (i,j)
     References:
     ---------------
     """

    # height and width are obtained from the original (not padded ) image
    height = im_in.shape[0]
    width = im_in.shape[1]
    # initialize properties vector
    f_glcm = np.empty( ( height, width, len( props ),
        distances.shape[0], angles.shape[0] ), np.float64 )
    # get properties limits to be normalized
    limits = greycoprops_limits( props, levels )
    # looping over each patch
    for i in range( height ):
        for j in range( width ):
            # define current patch
            patch = im_patch[ i, j, :, : ]
            # calculates current patch's GLCM
            glcm = sft.greycomatrix( patch, distances, angles, levels, symmetric, normed=True )
            # calculating normalized properties
            for p_idx, prop in enumerate( props ):
                f_i = sft.greycoprops( glcm, prop )
                f_glcm[ i, j, p_idx, :, : ] = (
                    ( f_i - limits[p_idx, 0] ) / ( limits[p_idx, 1] - limits[p_idx, 0] + eps ) )
    return f_glcm

def greycoprops_limits(props, levels=256 ):
    """Description"""

    # initialize output array
    f_hat = np.empty( (len(props), 2), dtype=np.float64 )
    # list of properties names
    already_normed = Props.get_list( [Props.HOMOGENEITY, Props.ENERGY, Props.ASM] )
    # looping through properties
    for p_idx, prop in enumerate(props):
        # already normalized properties
        if prop in already_normed :
            f_hat[p_idx, :] = [0, 1]
        # contrast limits
        elif prop in Props.get_list([Props.CONTRAST]) :
            f_hat[p_idx, :] = [0, (levels-1)**2 ]
        # dissimilarity limits
        elif prop in Props.get_list([Props.DISSIMILARITY]) :
            f_hat[p_idx, :] = [0, (levels-1) ]
        # correlation limits
        elif prop in Props.get_list([Props.CORRELATION]) :
            f_hat[p_idx, :] = [-1, 1]
    return f_hat
    already_normed = Props.get_list(
        [Props.HOMOGENEITY, Props.ENERGY, Props.ASM]
    )
    
    for pp, prop in enumerate(props):

       # already normalized properties
        if prop in already_normed:
           output[pp, :] = [0, 1]
        elif prop in Props.get_list([Props.CONTRAST]):
            output[pp, :] = [0, (levels-1)**2 ]
        elif prop in Props.get_list([Props.DISSIMILARITY]):
            output[pp, :] = [0, (levels-1) ]
        elif prop in Props.get_list([Props.CORRELATION]):
            output[pp, :] = [-1, 1]

    return output

