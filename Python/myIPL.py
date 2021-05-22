""" My Image Processing Library
 Defining functions to be used in this project """

import time

import cv2 as cv
import numpy as np
import skimage
import skimage.feature


def nlm_lbp(In, t, f, h, R=2, P=16):
    """
    
    IMPLEMENTATION OF NLM - LBP DENOISING

        In: image to be filtered
        t: radius of search window
        f: radius of similarity square neighborhood
        h: degree of filtering

        REFERENCES
        [1] KHELLAH, F. Application of local binary pattern to w_indowed 
            nonlocal means image denoising.In: SPRINGER.International 
            Conference on Image Analysis and Processing. [S.l.], 2013.p. 21–30.
        [2] Antoni Buades, Bartomeu Coll, Jean-Michel Morel. A review of image 
            denoising algorithms, w_ith a new one. SIAM Journal on Multiscale 
            Modeling and Simulation: A SIAM Interdisciplinary Journal, 2005, 
            4 (2), pp.490-530. <hal-00271141>
        [3] Jose Vicente Manjon-Herrera (2021). Non-Local Means Filter 
            (https://www.mathworks.com/matlabcentral/fileexchange/13176-non-
            local-means-filter), MATLAB Central File Exchange. Retrieved March
            23, 2021.
    """

    # Memory for output
    Y = np.zeros(In.shape)

    # Get input image dimensions
    [M, N] = In.shape

    # Symmetric padding input image
    In_pad = np.pad(In, pad_width=t, mode='symmetric')
    #           In_pad = np.pad(In, pad_width=f, mode='symmetric')

    # Calculates LBP
    print('calculating lbp')
    t0 = time.time()
    lbp = skimage.feature.local_binary_pattern(In_pad, P=P, R=R)
    print(f'lbp calculated in {time.time() - t0} s')

    # For each pixel of the input image
    t00 = time.time()
    for x in range(0, M):
        for y in range(0,N):

            print(f'\t({x},{y})')

            # target pixel (i) coordinates (in In_pad)
            [xi, yi] = [x+t, y+t]

            # Target intensity patch (in In_pad)
            Ni = In_pad[xi-f:xi+f, yi-f:yi+f]
            
            # Target LBP patch (in lbp)
            Li = lbp[xi-f:xi+f, yi-f:yi+f]
            # Li histogram
            t0 = time.time()
            Hi = np.histogram(Li, bins=2**P-1, range=(0,2**P-1))
            print(f'\t\tLi histogram calculated in {time.time() - t0} s')

            # Searching window (in In_pad)
            xj_min = max([xi-t, f])
            xj_max = min([xi+t, M+f])
            yj_min = max([yi-t, f])
            yj_max = min([yi+t, N+f])
            Si = In_pad[xj_min:xj_max, yj_min:yj_max]

            # Initialize Chi-squared weight (dissimilarity, at first) Matrix (LBP-weight)
            W_lbp = np.zeros((xj_max-xj_min, yj_max-yj_min), dtype=np.double)

            # Intensity-based weights matrix (Intensity-weight)
            W_i = np.zeros((xj_max-xj_min, yj_max-yj_min), dtype=np.double)

            for xj in range(xj_min, xj_max):
                for yj in range(yj_min, yj_max):

                    ### INTENSITY-BASED WEIGHT CALCULATION

                    # Comparison intensity patch (in In_pad)
                    Nj = In_pad[xj-f:xj+f, yj-f:yj+f]

                    # Intensity Euclidian distance
                    d_i = np.sum((np.double(Nj) - np.double(Ni))**2)

                    # Store Intensity weight into W_i
                    W_i[xj - xj_min][yj - yj_min] = np.exp(-d_i/(h**2))

                    ### LBP-BASED DISSIMILARITY CALCULATION

                    # Comparison LBP patch (in lbp)
                    Lj = lbp[xj-f:xj+f, yj-f:yj+f]
                    # Lj histogram
                    t0 = time.time()
                    Hj = np.histogram(Lj, bins=2**P-1, range=(0,2**P-1))
                    print(f'\t\t\tLj histogram calculated in {time.time() - t0} s')

                    # Computes the Chi-square LBP dissimilarity into the weight matrix (weight will be calculated)
                    t0 = time.time()
                    for n in range(len(Hi)):
                        if (Hi[0][n] + Hj[0][n]) != 0:
                            W_lbp[xj - xj_min][yj - yj_min] += ( Hi[0][n] - Hj[0][n] )**2 / ( Hi[0][n] + Hj[0][n] )
                    print(f'\t\t\tLBP distance calculated in {time.time() - t0} s')
            
            print(f'total elapsed time for ({x},{y}): {time.time() - t00} s')
            
            ### INTENSITY-BASED WEIGHT NORMALIZING
            Z_i = np.sum(W_i)
            W_i /= Z_i

            ### LBP-BASED WEIGHT CALCULATION / NORMALIZING

            # REGION SMOOTHNESS ESTIMATION
            hSi = np.std(W_lbp)

            # Turn dissimilarity matrix into weight matrix
            W_lbp = np.exp(-W_lbp/hSi)

            # Normalizing LBP-based weights
            Z_lbp = np.sum(W_lbp)
            W_lbp /= Z_lbp

            ### COMPUTING BOTH WEIGHTS THROUGH Si
            Y[x][y] = np.sum( (W_i * W_lbp) * Si)

    return Y

def nlm(In, t, f, h):
    """
    
    IMPLEMENTATION OF NLM - LBP DENOISING

        Inputs
            In: image to be filtered
            t: radius of search window
            f: radius of similarity square neighborhood
            h: degree of filtering
        Outputs:
            Y: nlm filtered image

        REFERENCES
        [1] Antoni Buades, Bartomeu Coll, Jean-Michel Morel. A review of image 
            denoising algorithms, w_ith a new one. SIAM Journal on Multiscale 
            Modeling and Simulation: A SIAM Interdisciplinary Journal, 2005, 
            4 (2), pp.490-530. <hal-00271141>
    """

    # Memory for output
    Y = np.zeros(In.shape)

    # Get input image dimensions
    [M, N] = In.shape

    # Symmetric padding input image
    In_pad = np.pad(In, pad_width=t, mode='symmetric')

    # For each pixel of the input image
    t000 = time.time()
    for x in range(0, M):
        for y in range(0,N):

            t00 = time.time()

            print(f'\t({x},{y})')

            # target pixel (i) coordinates (in In_pad)
            [xi, yi] = [x+t, y+t]

            # Target intensity patch (in In_pad)
            Ni = In_pad[xi-f:xi+f, yi-f:yi+f]

            # Searching window (in In_pad)
            xj_min = max([xi-t, f])
            xj_max = min([xi+t, M+f])
            yj_min = max([yi-t, f])
            yj_max = min([yi+t, N+f])
            Si = In_pad[xj_min:xj_max, yj_min:yj_max]

            # Intensity-based weights matrix (Intensity-weight)
            W_i = np.zeros((xj_max-xj_min, yj_max-yj_min), dtype=np.double)

            for xj in range(xj_min, xj_max):
                for yj in range(yj_min, yj_max):

                    ### INTENSITY-BASED WEIGHT CALCULATION

                    # Comparison intensity patch (in In_pad)
                    Nj = In_pad[xj-f:xj+f, yj-f:yj+f]

                    # Intensity Euclidian distance
                    d_i = np.sum((np.double(Nj) - np.double(Ni))**2)

                    # Store Intensity weight into W_i
                    W_i[xj - xj_min][yj - yj_min] = np.exp(-d_i/(h**2))
            
            print(f'total elapsed time for ({x},{y}): {time.time() - t00} s')
            
            ### INTENSITY-BASED WEIGHT NORMALIZING
            Z_i = np.sum(W_i)
            W_i /= Z_i

            ### COMPUTING BOTH WEIGHTS THROUGH Si
            Y[x][y] = np.sum( (W_i) * Si)
    
    print(f'TOTAL ELAPSED TIME: {time.time() - t000}')

    return Y

def im2patch(f, hw):
    """ Converte a imagem em uma matriz de patches.
        Inputs:
            f: imagem de entrada
            hw: meia largura do patch
        Outputs:
            F: matriz de patches
    """

    #   dimensão de entrada
    [M, N] = f.shape

    #   largura do patch
    w = 2*hw + 1

    #   replicar bordas
    ff = np.pad(f.astype('float16'), pad_width=hw, mode='symmetric')

    #   matriz de patches
    F = np.zeros([M, N, w, w], dtype=np.float16)

    for i in range(M):
        for j in range(N):
            
            patch = ff[i:i+w, j:j+w]
            F[i,j,:,:] = patch
    
    return F

def nlm_andre(v, d, s, h):
    """ Implementação do filtro Nonlocal-Means.
        Inputs:
            v: imagem ruidosa
            d: distância da janela de busca
            s: distância do patch
            f: função de similaridade
            h: parâmetro de filtragem
        Outputs:
            u: imagem filtrada
    """
    
    print(f'starting nlm_andre...')

    h = h*h

    #   tamanho da imagem de entrada
    [M, N] = v.shape
    #   imagem de saída
    u = np.zeros([M, N], dtype=np.float16)
    #   matriz da soma dos pesos
    w = np.zeros([M, N], dtype=np.float16)
    #   matriz dos pesos máximos
    z = np.zeros([M, N], dtype=np.float16)
    #   matriz de patches
    P = im2patch(v, s)

    # replicar bordas das matrizes da imagem e de patch
    v1 = np.pad(v, pad_width=d, mode='symmetric')
    P1 = np.pad(P, pad_width=[(d,d),(d,d),(0,0),(0,0)], mode='symmetric')

    #   offset das bordas
    x = [k+d for k in range(M)]
    y = [k+d for k in range(N)]
    
    t000 = time.time()
    for dx in range(-d, d):
        for dy in range(-d, d):

            t00 = time.time()

            # pular o deslocamento (dx, dy) = (0,0) -- não comparar patch central com ele mesmo
            if (dx != 0 or dy != 0):

                #   desloca matrizes
                #vxy = np.zeros(v.shape, dtype=np.uint8)
                #Pxy = np.zeros(P.shape, dtype=np.float16)                
                vxy = np.array([v1[kx+dx, ky+dy] for kx in x for ky in y]).reshape(v.shape)
                Pxy = np.array([P1[kx+dx, kx+dy, :, :] for kx in x for ky in y]).reshape(P.shape)

                """for nx, kx in enumerate(x):
                    for ny, ky in enumerate(y):
                        vxy[nx, ny] = v1[kx+dx, ky+dy]
                        Pxy[nx, ny] = P1[kx+dx, kx+dy, :, :]"""

                #   calcula as distâncias
                dxy = np.sum((P-Pxy)**2, axis=(2,3), dtype=np.float32)

                #   calcula os pesos
                wxy = np.exp(-dxy/h, dtype=np.float16)

                #   atualiza a matriz com os pesos máximos
                z = np.array([max([z[a,b], wxy[a,b]]) for a in range(M) for b in range(N)]).reshape(z.shape)
                """ for nx, kx in enumerate(x):
                    for ny, ky in enumerate(y):
                        z[nx, ny] = max(z[nx, ny], wxy[nx, ny]) """

                #   atualiza a imagem de saída com o peso dos patches do deslocamento atual
                u += wxy*vxy

                #   atualiza a matriz da soma dos pesos
                w += wxy
            
            print(f'\t tempo de 1 loop: {time.time()- t00} s')
    
    #   atribui peso máximo para pixel central
    u += v*z

    #   incrementa a matriz de soma dos pesos com os pesos do pixel central
    w += z

    # normaliza os valores obtidos com a soma dos pesos
    u /= w

    print(f'total time: {time.time() - t000} s')

    return u.astype('uint8')

def imresize(img, ratio, interpol = cv.INTER_AREA):
    """Return the resized (by ratio) image. It keeps the aspect ratio
        (x/y).
    """

    h, w = img.shape[:2]
    dim = (int(ratio * w), int(ratio * h))

    return cv.resize(img, dim, interpolation=interpol)

def _nl_means_denoising_3d(image, s, d, h, var):
    """
    Perform non-local means denoising on 3-D array
    Parameters
    ----------
    image : ndarray
        Input data to be denoised.
    s : int, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : np_floats, optional
        Cut-off distance (in gray levels).
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.
    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    dtype = np.float64

    n_pln, n_row, n_col = image.shape[0], image.shape[1], image.shape[2]
    offset = s // 2
    # padd the image so that boundaries are denoised as well
    padded = np.ascontiguousarray(
        np.pad(image, offset, mode='reflect'))
    result = np.empty_like(image)

    A = ((s - 1.) / 4.)
    range_vals = np.arange(-offset, offset + 1, dtype=dtype)
    xg_pln, xg_row, xg_col = np.meshgrid(range_vals, range_vals, range_vals,
                                         indexing='ij')
    w = np.ascontiguousarray(
        np.exp(-(xg_pln * xg_pln + xg_row * xg_row + xg_col * xg_col) /
               (2 * A * A)))
    w *= 1. / (np.sum(w) * h * h)

    var *= 2

    # Iterate over planes, taking padding into account
    for pln in range(n_pln):
        i_start = pln - min(d, pln)
        i_end = pln + min(d + 1, n_pln - pln)
        # Iterate over rows, taking padding into account
        for row in range(n_row):
            j_start = row - min(d, row)
            j_end = row + min(d + 1, n_row - row)
            # Iterate over columns, taking padding into account
            for col in range(n_col):
                k_start = col - min(d, col)
                k_end = col + min(d + 1, n_col - col)

                central_patch = padded[pln:pln+s, row:row+s, col:col+s]

                new_value = 0
                weight_sum = 0

                # Iterate over local 3d patch for each pixel
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        for k in range(k_start, k_end):
                            weight = patch_distance_3d(central_patch,
                                padded[i:i+s, j:j+s, k:k+s],
                                w, s, var)
                            # Collect results in weight sum
                            weight_sum += weight
                            new_value += weight * padded[i+offset,
                                                            j+offset,
                                                            k+offset]

                # Normalize the result
                result[pln, row, col] = new_value / weight_sum

    return np.asarray(result)

def denoise_nl_means(image, patch_size=7, patch_distance=11, h=0.1, sigma=0.):
    """Perform non-local means denoising on 2-D or 3-D grayscale images, and
    2-D RGB images.
    Parameters
    ----------
    image : 2D or 3D ndarray
        Input image to be denoised, which can be 2D or 3D, and grayscale
        or RGB (for 2D images only, see ``multichannel`` parameter).
    patch_size : int, optional
        Size of patches used for denoising.
    patch_distance : int, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : float, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches. A higher h results in a smoother image,
        at the expense of blurring features. For a Gaussian noise of standard
        deviation sigma, a rule of thumb is to choose the value of h to be
        sigma of slightly less.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    fast_mode : bool, optional
        If True (default value), a fast version of the non-local means
        algorithm is used. If False, the original version of non-local means is
        used. See the Notes section for more details about the algorithms.
    sigma : float, optional
        The standard deviation of the (Gaussian) noise.  If provided, a more
        robust computation of patch weights is computed that takes the expected
        noise variance into account (see Notes below).
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    result : ndarray
        Denoised image, of same shape as `image`.
    Notes
    -----
    The non-local means algorithm is well suited for denoising images with
    specific textures. The principle of the algorithm is to average the value
    of a given pixel with values of other pixels in a limited neighbourhood,
    provided that the *patches* centered on the other pixels are similar enough
    to the patch centered on the pixel of interest.
    In the original version of the algorithm [1]_, corresponding to
    ``fast=False``, the computational complexity is::
        image.size * patch_size ** image.ndim * patch_distance ** image.ndim
    Hence, changing the size of patches or their maximal distance has a
    strong effect on computing times, especially for 3-D images.
    However, the default behavior corresponds to ``fast_mode=True``, for which
    another version of non-local means [2]_ is used, corresponding to a
    complexity of::
        image.size * patch_distance ** image.ndim
    The computing time depends only weakly on the patch size, thanks to
    the computation of the integral of patches distances for a given
    shift, that reduces the number of operations [1]_. Therefore, this
    algorithm executes faster than the classic algorithm
    (``fast_mode=False``), at the expense of using twice as much memory.
    This implementation has been proven to be more efficient compared to
    other alternatives, see e.g. [3]_.
    Compared to the classic algorithm, all pixels of a patch contribute
    to the distance to another patch with the same weight, no matter
    their distance to the center of the patch. This coarser computation
    of the distance can result in a slightly poorer denoising
    performance. Moreover, for small images (images with a linear size
    that is only a few times the patch size), the classic algorithm can
    be faster due to boundary effects.
    The image is padded using the `reflect` mode of `skimage.util.pad`
    before denoising.
    If the noise standard deviation, `sigma`, is provided a more robust
    computation of patch weights is used.  Subtracting the known noise variance
    from the computed patch distances improves the estimates of patch
    similarity, giving a moderate improvement to denoising performance [4]_.
    It was also mentioned as an option for the fast variant of the algorithm in
    [3]_.
    When `sigma` is provided, a smaller `h` should typically be used to
    avoid oversmoothing.  The optimal value for `h` depends on the image
    content and noise level, but a reasonable starting point is
    ``h = 0.8 * sigma`` when `fast_mode` is `True`, or ``h = 0.6 * sigma`` when
    `fast_mode` is `False`.
    References
    ----------
    .. [1] A. Buades, B. Coll, & J-M. Morel. A non-local algorithm for image
           denoising. In CVPR 2005, Vol. 2, pp. 60-65, IEEE.
           :DOI:`10.1109/CVPR.2005.38`
    .. [2] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
           nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
           International Symposium on Biomedical Imaging: From Nano to Macro,
           2008, pp. 1331-1334.
           :DOI:`10.1109/ISBI.2008.4541250`
    .. [3] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
           Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
           :DOI:`10.5201/ipol.2014.120`
    .. [4] A. Buades, B. Coll, & J-M. Morel. Non-Local Means Denoising.
           Image Processing On Line, 2011, vol. 1, pp. 208-212.
           :DOI:`10.5201/ipol.2011.bcm_nlm`
    Examples
    --------
    >>> a = np.zeros((40, 40))
    >>> a[10:-10, 10:-10] = 1.
    >>> a += 0.3 * np.random.randn(*a.shape)
    >>> denoised_a = denoise_nl_means(a, 7, 5, 0.1)
    """

    image = image.astype(np.float16)

    kwargs = dict(s=patch_size, d=patch_distance, h=h, var=sigma * sigma)
    
    # 3-D grayscale
    return _nl_means_denoising_3d(image, **kwargs)

def patch_distance_3d(p1, p2, w, s, var):
    """
    Compute a Gaussian distance between two image patches.
    Parameters
    ----------
    p1 : 3-D array_like
        First patch.
    p2 : 3-D array_like
        Second patch.
    w : 3-D array_like
        Array of weights for the different pixels of the patches.
    s : Py_ssize_t
        Linear size of the patches.
    var_diff : np_floats
        The double of the expected noise variance.
    Returns
    -------
    distance : np_floats
        Gaussian distance between the two patches
    Notes
    -----
    The returned distance is given by
    .. math::  exp( -w ((p1 - p2)^2 - 2*var))
    """

    DISTANCE_CUTOFF = 5.0
    distance = 0

    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for k in range(s):
                tmp_diff = p1[i, j, k] - p2[i, j, k]
                distance += w[i, j, k] * (tmp_diff * tmp_diff - var)
    return np.exp(-max(0.0, distance))
