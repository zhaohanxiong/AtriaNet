from __future__ import division
import os
import cv2
import random
import numbers
import tflearn
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import rotate
from skimage.exposure import rescale_intensity
from scipy.ndimage.filters import gaussian_filter
from skimage.util import img_as_float, img_as_uint
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import uniform_filter,gaussian_filter

NR_OF_GREY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm
	
def get_all_paths(root):

    paths = []
    dirs = os.listdir(root)
    dirs.sort()
    for folder in dirs:
        if not folder.startswith('.'):  # skip hidden folders
            path = root + '/' + folder
            paths.append(path)
    return paths

# compute the barycenter of mask volume.
# mask: binary indicator
def find_centers(mask):

    x, y, z = mask.nonzero()

    midx = np.mean(x)
    midy = np.mean(y)
    midz = np.mean(z)

    return midx, midy, midz

def interpolate_data_z(ImageIn,factor):
	
	# this function interpolates the 3D data in the z direction depending on the factor
	Nx,Ny,Nz = ImageIn.shape
	x,y,z = np.linspace(1,Nx,Nx),np.linspace(1,Ny,Ny),np.linspace(1,Nz,Nz)
	interp_func = RegularGridInterpolator((x,y,z),ImageIn,method="linear")
	[xi,yi,zi] = np.meshgrid(x,y,np.linspace(1,Nz,factor*Nz),indexing='ij')
	ImageIn = interp_func( np.stack([xi,yi,zi],axis=3) )
	
	return(ImageIn)

def smooth3D_interpolate(data,threshold=20,factor=2):
	
	# this function interpolates the MRI and smoothes it in 3D
	data[data>=1] = 1
	data[data!=1] = 0
	data[data==1] = 50
	data = interpolate_data_z(data,factor)
	data = uniform_filter(data,5)
	data[data <  threshold] = 0
	data[data >= threshold] = 50
	data = data//50
	
	return(data)

# this function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):

	data = sitk.ReadImage( full_path_filename )
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
	data = sitk.GetArrayFromImage(data)

	return(data)

# this function encodes a 2D file into run-length-encoding format (RLE)
# 	the inpuy is a 2D binary image (1 = positive), the output is a string of the RLE
def run_length_encoding(input_mask):
	
	dots = np.where(input_mask.T.flatten()==1)[0]
	
	run_lengths,prev = [],-2
	
	for b in dots:
		if (b>prev+1): run_lengths.extend((b+1, 0))

		run_lengths[-1] += 1
		prev = b

	return(" ".join([str(i) for i in run_lengths]))

###############################################################################################################################################################
### Mutli-Label 2D Augmentation Functions
###############################################################################################################################################################

def multilabel_elastic_deformation_2d(volume, mask, alpha, sigma):

    shape = volume.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
	
    for i in range(0,mask.shape[2]):
        mask[:,:,i] = map_coordinates(mask[:,:,i], indices, order=1).reshape(shape)
	
    return map_coordinates(volume, indices, order=2).reshape(shape),mask

def multilabel_random_rotation_2d(volume, mask, max_angles):

    # rotate along y-axis
    angle = random.uniform(-max_angles[1], max_angles[1])
    volume3 = rotate(volume, angle, order=2, mode='nearest', reshape=False)
	
    for i in range(0,mask.shape[2]):
        mask[:,:,i] = rotate(mask[:,:,i], angle, order=1, mode='nearest', reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angles[0], max_angles[0])
    volume_rot = rotate(volume3, angle, order=2, mode='nearest', reshape=False)
	
    for i in range(0,mask.shape[2]):
        mask[:,:,i] = rotate(mask[:,:,i], angle, order=1, mode='nearest', reshape=False)
		
    return volume_rot, mask
	
def multilabel_random_scale_2d(volume, mask, max_scale_deltas):

    scalex = random.uniform(1 - max_scale_deltas[0], 1 + max_scale_deltas[0])
    scaley = random.uniform(1 - max_scale_deltas[1], 1 + max_scale_deltas[1])

    volume_zoom = zoom(volume, (scalex, scaley), order=2)
    volume_out  = cv2.resize(volume_zoom,(volume.shape[1],volume.shape[0]))

    for i in range(0,mask.shape[2]):
        mask_zoom   = zoom(mask[:,:,i], (scalex, scaley), order=1)
        mask[:,:,i] = cv2.resize(mask_zoom,(volume.shape[1],volume.shape[0]))

    return volume_out, mask

def multilabel_random_flip_2d(volume, mask):

    if random.choice([True, False]):
        volume = volume[::-1,:].copy()  # here must use copy(), otherwise error occurs
        mask   = mask[::-1,:,:].copy()
    else:
        volume = volume[:,::-1].copy()
        mask   = mask[:,::-1,:].copy()

    return volume, mask

###############################################################################################################################################################
### Online Augmentation
###############################################################################################################################################################
# augments data and label on the fly
def online_augmentation(data,label):
	#train_image,train_label,train_mean,train_sd = online_augmentation(train_Image,train_Label)
	print("\nPerforming Online Data Augmentation...\n")
	
	# initialize arrays for storing the data
	data_aug = np.zeros([data.shape[0],data.shape[1],data.shape[2],data.shape[3]]) # N x X x Y x 1
	label_aug = np.zeros([label.shape[0],label.shape[1],label.shape[2],label.shape[3]]) # N x X x Y x 4
	
	# loop through all the data
	for i in range(data.shape[0]):
	
		temp_img,temp_lab = data[i,:,:,0],label[i,:,:,:]
		
		# probably of augmentation
		if random.uniform(0,1) >= 0:

			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_random_scale_2d(temp_img,temp_lab,(0.2,0.2))
			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_random_rotation_2d(temp_img,temp_lab,(20,20))
			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_random_flip_2d(temp_img,temp_lab)
			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_elastic_deformation_2d(temp_img,temp_lab,temp_img.shape[0]*3,temp_img.shape[0]*0.10)
			
			temp_lab[temp_lab<0.5]  = 0
			temp_lab[temp_lab>=0.5] = 1

		# append data
		data_aug[i,:,:,0],label_aug[i,:,:,:] = temp_img,temp_lab
		
	# normalize the data
	data_mean,data_sd = np.mean(data_aug),np.std(data_aug)
	data_aug = (data_aug - data_mean)/data_sd
	
	return(data_aug,label_aug,data_mean,data_sd)

###############################################################################################################################################################
### CLAHE
###############################################################################################################################################################

def equalize_adapthist_3d(image, kernel_size=None,
                          clip_limit=0.01, nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.
    Parameters
    ----------
    image : (N1, ...,NN[, C]) ndarray
        Input image.
    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim`` (without color channel). If integer, it is broadcasted
        to each `image` dimension. By default, ``kernel_size`` is 1/8 of
        ``image`` height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    Returns
    -------
    out : (N1, ...,NN[, C]) ndarray
        Equalized image.
    See Also
    --------
    equalize_hist, rescale_intensity
    Notes
    -----
    * For color images, the following steps are performed:
       - The image is converted to HSV color space
       - The CLAHE algorithm is run on the V (Value) channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.
    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))

    if kernel_size is None:
        kernel_size = tuple([image.shape[dim] // 8 for dim in range(image.ndim)])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        ValueError('Incorrect value of `kernel_size`: {}'.format(kernel_size))

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit * nbins, nbins)
    image = img_as_float(image)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins=128):
    """Contrast Limited Adaptive Histogram Equalization.
    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size: int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit (higher values give more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.
    The number of "effective" greylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    ns = [int(np.ceil(image.shape[dim] / kernel_size[dim])) for dim in range(image.ndim)]

    steps = [int(np.floor(image.shape[dim] / ns[dim])) for dim in range(image.ndim)]

    bin_size = 1 + NR_OF_GREY // nbins
    lut = np.arange(NR_OF_GREY)
    lut //= bin_size

    map_array = np.zeros(tuple(ns) + (nbins,), dtype=int)

    # Calculate greylevel mappings for each contextual region

    for inds in np.ndindex(*ns):

        region = tuple([slice(inds[dim] * steps[dim], (inds[dim] + 1) * steps[dim]) for dim in range(image.ndim)])
        sub_img = image[region]

        if clip_limit > 0.0:  # Calculate actual cliplimit
            clim = int(clip_limit * sub_img.size / nbins)
            if clim < 1:
                clim = 1
        else:
            clim = NR_OF_GREY  # Large value, do not clip (AHE)

        hist = lut[sub_img.ravel()] #lut[sub_img.ravel().astype(np.int32)]
        hist = np.bincount(hist)
        hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
        hist = clip_histogram(hist, clim)
        hist = map_histogram(hist, 0, NR_OF_GREY - 1, sub_img.size)
        map_array[inds] = hist

    # Interpolate greylevel mappings to get CLAHE image

    offsets = [0] * image.ndim
    lowers = [0] * image.ndim
    uppers = [0] * image.ndim
    starts = [0] * image.ndim
    prev_inds = [0] * image.ndim

    for inds in np.ndindex(*[ns[dim] + 1 for dim in range(image.ndim)]):

        for dim in range(image.ndim):
            if inds[dim] != prev_inds[dim]:
                starts[dim] += offsets[dim]

        for dim in range(image.ndim):
            if dim < image.ndim - 1:
                if inds[dim] != prev_inds[dim]:
                    starts[dim + 1] = 0

        prev_inds = inds[:]

        # modify edges to handle special cases
        for dim in range(image.ndim):
            if inds[dim] == 0:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = 0
                uppers[dim] = 0
            elif inds[dim] == ns[dim]:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = ns[dim] - 1
                uppers[dim] = ns[dim] - 1
            else:
                offsets[dim] = steps[dim]
                lowers[dim] = inds[dim] - 1
                uppers[dim] = inds[dim]

        maps = []
        for edge in np.ndindex(*([2] * image.ndim)):
            maps.append(map_array[tuple([[lowers, uppers][edge[dim]][dim] for dim in range(image.ndim)])])

        slices = [np.arange(starts[dim], starts[dim] + offsets[dim]) for dim in range(image.ndim)]

        interpolate(image, slices[::-1], maps, lut)

    return image


def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.
    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).
    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.
    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = int(n_excess / hist.size)  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    hist[excess_mask] = clip_limit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = (hist >= upper) & (hist < clip_limit)
    mid = hist[mid_mask]
    n_excess -= mid.size * clip_limit - mid.sum()
    hist[mid_mask] = clip_limit

    prev_n_excess = n_excess

    while n_excess > 0:  # Redistribute remaining excess
        index = 0
        while n_excess > 0 and index < hist.size:
            under_mask = hist < 0
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under_mask[indices] = True
            under_mask = (under_mask) & (hist < clip_limit)
            hist[under_mask] += 1
            n_excess -= under_mask.sum()
            index += 1
        # bail if we have not distributed any excess
        if prev_n_excess == n_excess:
            break
        prev_n_excess = n_excess

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).
    It does so by cumulating the input histogram.
    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.
    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist).astype(float)
    scale = ((float)(max_val - min_val)) / n_pixels
    out *= scale
    out += min_val
    out[out > max_val] = max_val
    return out.astype(int)


def interpolate(image, slices, maps, lut):
    """Find the new grayscale level for a region using bilinear interpolation.
    Parameters
    ----------
    image : ndarray
        Full image.
    slices : list of array-like
       Indices of the region.
    maps : list of ndarray
        Mappings of greylevels from histograms.
    lut : ndarray
        Maps grayscale levels in image to histogram levels.
    Returns
    -------
    out : ndarray
        Original image with the subregion replaced.
    Notes
    -----
    This function calculates the new greylevel assignments of pixels within
    a submatrix of the image. This is done by linear interpolation between
    2**image.ndim different adjacent mappings in order to eliminate boundary artifacts.
    """

    norm = np.product([slices[dim].size for dim in range(image.ndim)])  # Normalization factor

    # interpolation weight matrices
    coeffs = np.meshgrid(*tuple([np.arange(slices[dim].size) for dim in range(image.ndim)]), indexing='ij')
    coeffs = [coeff.transpose() for coeff in coeffs]

    inv_coeffs = [np.flip(coeffs[dim], axis=image.ndim - dim - 1) + 1 for dim in range(image.ndim)]

    region = tuple([slice(int(slices[dim][0]), int(slices[dim][-1] + 1)) for dim in range(image.ndim)][::-1])
    view = image[region]

    im_slice = lut[view.astype(np.int32)]

    new = np.zeros_like(view, dtype=int)
    for iedge, edge in enumerate(np.ndindex(*([2] * image.ndim))):
        edge = edge[::-1]
        new += np.product([[inv_coeffs, coeffs][edge[dim]][dim] for dim in range(image.ndim)], 0) * maps[iedge][
            im_slice]

    new = (new / norm).astype(view.dtype)
    view[::] = new
    return image

###############################################################################################################################################################
### AtriaNet
###############################################################################################################################################################
# 3d convolution operation
def tflearn_conv_2d(net,nb_filter,kernel,stride,dropout=1.0,is_train=True):

    net = tflearn.layers.conv.conv_2d(net,nb_filter,kernel,stride,padding="same",activation="linear",bias=False,trainable=is_train)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.prelu(net)
    
    net = tflearn.layers.core.dropout(net,keep_prob=dropout)
    
    return(net)

# 3d deconvolution operation
def tflearn_deconv_2d(net,nb_filter,kernel,stride,dropout=1.0,is_train=True):

    net = tflearn.layers.conv.conv_2d_transpose(net,nb_filter,kernel,
                                                [net.shape[1].value*stride,net.shape[2].value*stride,nb_filter],
                                                [1,stride,stride,1],padding="same",activation="linear",bias=False,trainable=is_train)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.prelu(net)
    net = tflearn.layers.core.dropout(net,keep_prob=dropout)
    
    return(net)

# merging operation
def tflearn_merge_2d(layers,method):
    
    net = tflearn.layers.merge_ops.merge(layers,method,axis=3)
    
    return(net)

# detection CNN
def AtriaNet_ROI(x, y):

    fm 			= 16	# feature map scale
    kkk 		= 5		# kernel size
    keep_rate 	= 0.8   # dropout

    # level 0 input
    layer_0a_input	= tflearn.layers.core.input_data(shape=[None,x,y,1])

    # level 1 down
    layer_1a_conv 	= tflearn_conv_2d(net=layer_0a_input,nb_filter=fm,kernel=kkk,stride=1,is_train=True)
    layer_1a_stack	= tflearn_merge_2d([layer_0a_input]*fm,"concat")

    layer_1a_add	= tflearn_merge_2d([layer_1a_conv,layer_1a_stack],"elemwise_sum")
    layer_1a_down	= tflearn_conv_2d(net=layer_1a_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

    # level 2 down
    layer_2a_conv 	= tflearn_conv_2d(net=layer_1a_down,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)
    layer_2a_conv 	= tflearn_conv_2d(net=layer_2a_conv,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

    layer_2a_add	= tflearn_merge_2d([layer_1a_down,layer_2a_conv],"elemwise_sum")
    layer_2a_down	= tflearn_conv_2d(net=layer_2a_add,nb_filter=fm*4,kernel=2,stride=2,is_train=True)

    # level 3 down
    layer_3a_conv 	= tflearn_conv_2d(net=layer_2a_down,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

    layer_3a_add	= tflearn_merge_2d([layer_2a_down,layer_3a_conv],"elemwise_sum")
    layer_3a_down	= tflearn_conv_2d(net=layer_3a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 4 down
    layer_4a_conv 	= tflearn_conv_2d(net=layer_3a_down,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_4a_add	= tflearn_merge_2d([layer_3a_down,layer_4a_conv],"elemwise_sum")
    layer_4a_down	= tflearn_conv_2d(net=layer_4a_add,nb_filter=fm*16,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 5
    layer_5a_conv 	= tflearn_conv_2d(net=layer_4a_down,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_5a_add	= tflearn_merge_2d([layer_4a_down,layer_5a_conv],"elemwise_sum")
    layer_5a_up		= tflearn_deconv_2d(net=layer_5a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 4 up
    layer_4b_concat	= tflearn_merge_2d([layer_4a_add,layer_5a_up],"concat")
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_concat,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_4b_add	= tflearn_merge_2d([layer_4b_conv,layer_4b_concat],"elemwise_sum")
    layer_4b_up		= tflearn_deconv_2d(net=layer_4b_add,nb_filter=fm*4,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 3 up
    layer_3b_concat	= tflearn_merge_2d([layer_3a_add,layer_4b_up],"concat")
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_concat,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_3b_add	= tflearn_merge_2d([layer_3b_conv,layer_3b_concat],"elemwise_sum")
    layer_3b_up		= tflearn_deconv_2d(net=layer_3b_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

    # level 2 up
    layer_2b_concat	= tflearn_merge_2d([layer_2a_add,layer_3b_up],"concat")
    layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_concat,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

    layer_2b_add	= tflearn_merge_2d([layer_2b_conv,layer_2b_concat],"elemwise_sum")
    layer_2b_up		= tflearn_deconv_2d(net=layer_2b_add,nb_filter=fm,kernel=2,stride=2,is_train=True)

    # level 1 up
    layer_1b_concat	= tflearn_merge_2d([layer_1a_add,layer_2b_up],"concat")
    layer_1b_conv 	= tflearn_conv_2d(net=layer_1b_concat,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

    layer_1b_add	= tflearn_merge_2d([layer_1b_conv,layer_1b_concat],"elemwise_sum")

    # level 0 classifier
    layer_0b_conv	= tflearn.layers.conv.conv_2d(layer_1b_add,2,1,1,trainable=True)
    layer_0b_clf	= tflearn.activations.softmax(layer_0b_conv)

    # loss function
    def dice_loss_2d(y_pred,y_true):
        
        with tf.name_scope("dice_loss_2D_function"):
            
            # compute dice scores for each individually
            y_pred1,y_true1 = y_pred[:,:,:,1],y_true[:,:,:,1]
            intersection1   = tf.reduce_sum(y_pred1*y_true1)
            union1          = tf.reduce_sum(y_pred1*y_pred1) + tf.reduce_sum(y_true1*y_true1)
            dice1           = (2.0 * intersection1 + 1.0) / (union1 + 1.0)

        return(1.0 - dice1)

    # Optimizer
    regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',loss=dice_loss_2d,learning_rate=0.0001)
    built_model = tflearn.models.dnn.DNN(regress,tensorboard_dir='log')

    return(built_model)

# segmentation CNN
def AtriaNet_Seg(x, y):

    fm 			= 16	# feature map scale
    kkk 		= 5		# kernel size
    keep_rate 	= 0.8   # dropout

    # level 0 input
    layer_0a_input	= tflearn.layers.core.input_data(shape=[None,x,y,1])

    # level 1 down
    layer_1a_conv 	= tflearn_conv_2d(net=layer_0a_input,nb_filter=fm,kernel=kkk,stride=1,is_train=True)
    layer_1a_stack	= tflearn_merge_2d([layer_0a_input]*fm,"concat")

    layer_1a_add	= tflearn_merge_2d([layer_1a_conv,layer_1a_stack],"elemwise_sum")
    layer_1a_down	= tflearn_conv_2d(net=layer_1a_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

    # level 2 down
    layer_2a_conv 	= tflearn_conv_2d(net=layer_1a_down,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)
    layer_2a_conv 	= tflearn_conv_2d(net=layer_2a_conv,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

    layer_2a_add	= tflearn_merge_2d([layer_1a_down,layer_2a_conv],"elemwise_sum")
    layer_2a_down	= tflearn_conv_2d(net=layer_2a_add,nb_filter=fm*4,kernel=2,stride=2,is_train=True)

    # level 3 down
    layer_3a_conv 	= tflearn_conv_2d(net=layer_2a_down,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

    layer_3a_add	= tflearn_merge_2d([layer_2a_down,layer_3a_conv],"elemwise_sum")
    layer_3a_down	= tflearn_conv_2d(net=layer_3a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 4 down
    layer_4a_conv 	= tflearn_conv_2d(net=layer_3a_down,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_4a_add	= tflearn_merge_2d([layer_3a_down,layer_4a_conv],"elemwise_sum")
    layer_4a_down	= tflearn_conv_2d(net=layer_4a_add,nb_filter=fm*16,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 5
    layer_5a_conv 	= tflearn_conv_2d(net=layer_4a_down,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_5a_add	= tflearn_merge_2d([layer_4a_down,layer_5a_conv],"elemwise_sum")
    layer_5a_up		= tflearn_deconv_2d(net=layer_5a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 4 up
    layer_4b_concat	= tflearn_merge_2d([layer_4a_add,layer_5a_up],"concat")
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_concat,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_4b_add	= tflearn_merge_2d([layer_4b_conv,layer_4b_concat],"elemwise_sum")
    layer_4b_up		= tflearn_deconv_2d(net=layer_4b_add,nb_filter=fm*4,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 3 up
    layer_3b_concat	= tflearn_merge_2d([layer_3a_add,layer_4b_up],"concat")
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_concat,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_3b_add	= tflearn_merge_2d([layer_3b_conv,layer_3b_concat],"elemwise_sum")
    layer_3b_up		= tflearn_deconv_2d(net=layer_3b_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

    # level 2 up
    layer_2b_concat	= tflearn_merge_2d([layer_2a_add,layer_3b_up],"concat")
    layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_concat,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

    layer_2b_add	= tflearn_merge_2d([layer_2b_conv,layer_2b_concat],"elemwise_sum")
    layer_2b_up		= tflearn_deconv_2d(net=layer_2b_add,nb_filter=fm,kernel=2,stride=2,is_train=True)

    # level 1 up
    layer_1b_concat	= tflearn_merge_2d([layer_1a_add,layer_2b_up],"concat")
    layer_1b_conv 	= tflearn_conv_2d(net=layer_1b_concat,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

    layer_1b_add	= tflearn_merge_2d([layer_1b_conv,layer_1b_concat],"elemwise_sum")

    # level 0 classifier
    layer_0b_conv	= tflearn.layers.conv.conv_2d(layer_1b_add,4,1,1,trainable=True)
    layer_0b_clf	= tflearn.activations.softmax(layer_0b_conv)

    # loss function
    def dice_loss_2d(y_pred,y_true):
        
        with tf.name_scope("dice_loss_2D_function"):
            
            # compute dice scores for each individually
            y_pred1,y_true1 = y_pred[:,:,:,1],y_true[:,:,:,1]
            intersection1   = tf.reduce_sum(y_pred1*y_true1)
            union1          = tf.reduce_sum(y_pred1*y_pred1) + tf.reduce_sum(y_true1*y_true1)
            dice1           = (2.0 * intersection1 + 1.0) / (union1 + 1.0)
            
            y_pred2,y_true2 = y_pred[:,:,:,2],y_true[:,:,:,2]
            intersection2   = tf.reduce_sum(y_pred2*y_true2)
            union2          = tf.reduce_sum(y_pred2*y_pred2) + tf.reduce_sum(y_true2*y_true2)
            dice2           = (2.0 * intersection2 + 1.0) / (union2 + 1.0)
            
            y_pred3,y_true3 = y_pred[:,:,:,3],y_true[:,:,:,3]
            intersection3   = tf.reduce_sum(y_pred3*y_true3)
            union3          = tf.reduce_sum(y_pred3*y_pred3) + tf.reduce_sum(y_true3*y_true3)
            dice3           = (2.0 * intersection3 + 1.0) / (union3 + 1.0)
            
        return(1.0 - (dice1*0.333 + dice2*0.333 + dice3*0.333))

    # Optimizer
    regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',loss=dice_loss_2d,learning_rate=0.0001)
    built_model = tflearn.models.dnn.DNN(regress,tensorboard_dir='log')

    return(built_model)

def AtriaNet_AWT(x,y):

    fm 			= 16	# feature map scale
    kkk 		= 5		# kernel size
    keep_rate 	= 0.8   # dropout

    # level 0 input
    layer_0a_input	= tflearn.layers.core.input_data(shape=[None,x,y,1])

    # level 1 down
    layer_1a_conv 	= tflearn_conv_2d(net=layer_0a_input,nb_filter=fm,kernel=kkk,stride=1,is_train=True)
    layer_1a_stack	= tflearn_merge_2d([layer_0a_input]*fm,"concat")

    layer_1a_add	= tflearn_merge_2d([layer_1a_conv,layer_1a_stack],"elemwise_sum")
    layer_1a_down	= tflearn_conv_2d(net=layer_1a_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

    # level 2 down
    layer_2a_conv 	= tflearn_conv_2d(net=layer_1a_down,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)
    layer_2a_conv 	= tflearn_conv_2d(net=layer_2a_conv,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

    layer_2a_add	= tflearn_merge_2d([layer_1a_down,layer_2a_conv],"elemwise_sum")
    layer_2a_down	= tflearn_conv_2d(net=layer_2a_add,nb_filter=fm*4,kernel=2,stride=2,is_train=True)

    # level 3 down
    layer_3a_conv 	= tflearn_conv_2d(net=layer_2a_down,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

    layer_3a_add	= tflearn_merge_2d([layer_2a_down,layer_3a_conv],"elemwise_sum")
    layer_3a_down	= tflearn_conv_2d(net=layer_3a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 4 down
    layer_4a_conv 	= tflearn_conv_2d(net=layer_3a_down,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_4a_add	= tflearn_merge_2d([layer_3a_down,layer_4a_conv],"elemwise_sum")
    layer_4a_down	= tflearn_conv_2d(net=layer_4a_add,nb_filter=fm*16,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 5
    layer_5a_conv 	= tflearn_conv_2d(net=layer_4a_down,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_5a_add	= tflearn_merge_2d([layer_4a_down,layer_5a_conv],"elemwise_sum")
    layer_5a_up		= tflearn_deconv_2d(net=layer_5a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 4 up
    layer_4b_concat	= tflearn_merge_2d([layer_4a_add,layer_5a_up],"concat")
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_concat,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_4b_add	= tflearn_merge_2d([layer_4b_conv,layer_4b_concat],"elemwise_sum")
    layer_4b_up		= tflearn_deconv_2d(net=layer_4b_add,nb_filter=fm*4,kernel=2,stride=2,dropout=keep_rate,is_train=True)

    # level 3 up
    layer_3b_concat	= tflearn_merge_2d([layer_3a_add,layer_4b_up],"concat")
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_concat,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
    layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

    layer_3b_add	= tflearn_merge_2d([layer_3b_conv,layer_3b_concat],"elemwise_sum")
    layer_3b_up		= tflearn_deconv_2d(net=layer_3b_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

    # level 2 up
    layer_2b_concat	= tflearn_merge_2d([layer_2a_add,layer_3b_up],"concat")
    layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_concat,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
    layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

    layer_2b_add	= tflearn_merge_2d([layer_2b_conv,layer_2b_concat],"elemwise_sum")
    layer_2b_up		= tflearn_deconv_2d(net=layer_2b_add,nb_filter=fm,kernel=2,stride=2,is_train=True)

    # level 1 up
    layer_1b_concat	= tflearn_merge_2d([layer_1a_add,layer_2b_up],"concat")
    layer_1b_conv 	= tflearn_conv_2d(net=layer_1b_concat,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

    layer_1b_add	= tflearn_merge_2d([layer_1b_conv,layer_1b_concat],"elemwise_sum")

    # level 0 classifier
    layer_0b_conv	= tflearn.layers.conv.conv_2d(layer_1b_add,1,1,1,trainable=True)
    layer_0b_clf	= tflearn.activations.relu(layer_0b_conv)

    # loss function
    def mse_loss(y_pred,y_true):

        with tf.name_scope("mse_function"):
            
            # compute distance loss
            masked_true = tf.boolean_mask(y_true, tf.greater(y_true, tf.zeros_like(y_true)))
            masked_pred = tf.boolean_mask(y_pred, tf.greater(y_true, tf.zeros_like(y_true)))
            
            mse = tf.reduce_mean(tf.pow(masked_true - masked_pred,2))

        return(mse)

    # Optimizer
    regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',loss=mse_loss,learning_rate=0.0001)
    built_model   = tflearn.models.dnn.DNN(regress,tensorboard_dir='log')

    return(built_model)

###############################################################################################################################################################
### Fibrosis
###############################################################################################################################################################
def utah_threshold(img, lab, std_factor):
	
	# Inputs: img = 3D LGE-MRI, lab = 3D LA wall label, std_factor = tune threshold (2-4)
	
	# leave the LA wall pixels only
	img[lab!=1] = 0
	
	# limit intensity ranges to only 2% - 40%
	thresh_low  = np.quantile(img[img>0], 0.02)
	thresh_high = np.quantile(img[img>0], 0.40)
	
	img_norm = np.copy(img)
	img_norm[img_norm < thresh_low]  = 0
	img_norm[img_norm > thresh_high] = 0
	
	# define fibrosis output map
	fibrosis = np.zeros_like(img)
	
	# loop through each slice
	for i in range(img.shape[2]):		
		if np.sum(img[:,:,i]) > 0:

			# define individual slice images
			tmp,fibrosis_slice = img_norm[:,:,i],fibrosis[:,:,i]
			
			# define threshold
			thres_slice = np.mean(tmp[tmp>0]) + np.std(tmp[tmp>0]) * std_factor

			# add to fibrosis slice
			fibrosis_slice[img[:,:,i] > thres_slice] = 1
			fibrosis[:,:,i] = fibrosis_slice
		
	return(fibrosis)