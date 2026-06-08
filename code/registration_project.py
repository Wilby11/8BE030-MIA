"""
All the functions needed for the registration project.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy import ndimage
from IPython import get_ipython


def rotate(phi):
    # 2D rotation matrix.
    # Input:
    # phi - rotation angle
    # Output:
    # T - transformation matrix

    T = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]) 
    return T

def c2h(X):
    # Convert cartesian to homogeneous coordinates.
    # Input:
    # X - cartesian coordinates
    # Output:
    # Xh - homogeneous coordinates

    n = np.ones([1, X.shape[1]])
    Xh = np.concatenate((X, n))

    return Xh

def t2h(T, t):
    # Convert a 2D transformation matrix to homogeneous form.
    # Input:
    # T - 2D transformation matrix
    # t - 2D translation vector
    # Output:
    # Th - homogeneous transformation matrix

    Th = np.eye(3)
    Th[0:2, 0:2] = T
    Th[0:2, 2] = t
    return Th

def rigid2h(rot, tx, ty):
    # Input:
    # x - vector of 3 parameters, where the first parameter is the rotation angle in radians and the remaining two parameters are the translation in x and y direction.
    # Output:
    # Th - homogeneous transformation matrix

    T = rotate(rot)
    Th = t2h(T, [tx, ty])
    return Th

def affine2h(rot, sx, sy, shx, shy, tx, ty):
	# Input:
	# x - vector of 7 parameters, where the first 5 are the parameters of the affine
	# transformation, and the last two are the translation parameters. The first parameter 
	# is the rotation angle in radians, the second and third parameters are the scaling 
	# factors in x and y direction, the fourth and fifth parameters are the shear factors 
	# in x and y direction, and the last two parameters are the translation parameters in
	# x and y direction.
	# Output:
	# Ah - homogeneous affine transformation matrix

	Ah = np.eye(3)
	Ah[0:2, :] = np.array([[sx*np.cos(rot), -shy+sx*np.sin(rot), tx],
							[shx+sy*np.sin(rot), sy*np.cos(rot), ty]])
	return Ah


def cpselect(imagePath1, imagePath2):
	# Pops up a matplotlib window in which to select control points on the two images given as input.
	#
	# Input:
    # imagePath1 - fixed image path
    # imagePath2 - moving image path
    # Output:
    # X - control points in the fixed image
    # Xm - control points in the moving image
	
	#load the images
	image1 = plt.imread(imagePath1)
	image2 = plt.imread(imagePath2)
	
	#ensure that the plot opens in its own window
	get_ipython().run_line_magic('matplotlib', 'qt')
	
	#set up the overarching window
	fig, axes = plt.subplots(1,2)
	fig.figsize = [16,9]
	fig.suptitle("Left Mouse Button to create a point.\n Right Mouse Button/Delete/Backspace to remove the newest point.\n Middle Mouse Button/Enter to finish placing points.\n First select a point in Image 1 and then its corresponding point in Image 2.")
	
	#plot the images
	axes[0].imshow(image1)
	axes[0].set_title("Image 1")
	
	axes[1].imshow(image2)
	axes[1].set_title("Image 2")
	
	#accumulate points
	points = plt.ginput(n=-1, timeout=500)
	plt.close(fig)
	
	#restore to inline figure placement
	get_ipython().run_line_magic('matplotlib', 'inline')
	
	#if there is an uneven amount of points, raise an exception
	if not (len(points)%2 == 0):
		raise Exception("Uneven amount of control points: {0}. Even amount of control points required.".format(len(points)))
		
	#if there are no points, raise an exception
	if not (len(points)> 0):
		raise Exception("No control points selected.")
	
	#subdivide the points into two different arrays. If the current number is even belongs to the first first image, and uneven to the second image. (Assuming the points were entered in the correct order.)
	#X and Y values are on rows, with each column being a pair of values.
	k = len(points)//2
	X = np.empty((2,k))
	X[:] = np.nan
	Xm = np.empty((2,k))
	Xm[:] = np.nan

	for i in np.arange(len(points)):
		if i%2 == 0 :
			X[0,i//2] = points[i][0]
			X[1,i//2] = points[i][1]
		else:
			Xm[0,i//2] = points[i][0]
			Xm[1,i//2] = points[i][1]
	
	return X, Xm


def image_transform(I, Th, output_shape=None):
    # Image transformation by inverse mapping.
    # Input:
    # I - image to be transformed
    # Th - homogeneous transformation matrix
    # output_shape - size of the output image (default is same size as input)
    # Output:
    # It - transformed image
	# Xt - remapped coordinates
    # we want double precision for the interpolation, but we want the
    # output to have the same data type as the input - so, we will
    # convert to double and remember the original input type

    input_type = type(I)

    # default output size is same as input
    if output_shape is None:
        output_shape = I.shape

    # spatial coordinates of the transformed image
    x = np.arange(0, output_shape[1])
    y = np.arange(0, output_shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
    # convert to homogeneous coordinates
    Xh = c2h(X)

    # Xt is a 3-by-p matrix of the remapped coordinates in homogeneous form
    # Xt = (Th)^(-1) * Xh
    Xt = np.linalg.inv(Th).dot(Xh)
    It = ndimage.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(output_shape)
    Xt = Xt[:2,:]
    return It, Xt


def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations
    # E - squared error for the optimal solution

    # compute the error
    m, n = A.shape
    assert m >= n, "The system of equations is underdetermined. Least-squares solution is not applicable."
    # w = (A^T A)^(-1) A^T b
    w = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    E = np.transpose(A.dot(w) - b).dot(A.dot(w) - b)

    return w, E


def ls_affine(X, Xm):
    # Least-squares fitting of an affine transformation.
    # Input:
    # X - Points in the fixed image
    # Xm - Corresponding points in the moving image
    # Output:
    # T - affine transformation in homogeneous form.

    A = np.transpose(Xm)
    # Add a col of ones to the transposed Xm matrix
    A = np.concatenate((A, np.ones((A.shape[0], 1))), axis=1)
    w123, E123 = ls_solve(A, X[0,:])
    w456, E456 = ls_solve(A, X[1,:])
    
    T = np.array([[w123[0], w123[1], w123[2]],[w456[0], w456[1], w456[2]],[0, 0, 1]])
    
    return T


def correlation(I, J):
    # Compute the normalized cross-correlation between two images.
    # Input:
    # I, J - input images
    # Output:
    # CC - normalized cross-correlation
    # it's always good to do some parameter checks

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    u = I.reshape((I.shape[0]*I.shape[1],1))
    v = J.reshape((J.shape[0]*J.shape[1],1))

    # subtract the mean
    u = u - u.mean(keepdims=True)
    v = v - v.mean(keepdims=True)
    CC = ((u.T @ v) / (np.sqrt(u.T @ u) * np.sqrt(v.T @ v))).item()
    return CC

def joint_histogram(I, J, num_bins=16, minmax_range=None):
    # Compute the joint histogram of two signals.
    # Input:
    # I, J - input images
    # num_bins: number of bins of the joint histogram (default: 16)
    # range - range of the values of the signals (default: min and max
    # of the inputs)
    # Output:
    # p - joint histogram

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    # make sure the inputs are column-vectors of type double (highest
    # precision)
    I = I.reshape((I.shape[0]*I.shape[1],1)).astype(float)
    J = J.reshape((J.shape[0]*J.shape[1],1)).astype(float)

    # if the range is not specified use the min and max values of the
    # inputs
    if minmax_range is None:
        minmax_range = np.array([min(min(I),min(J)), max(max(I),max(J))])

    # this will normalize the inputs to the [0 1] range
    I = (I-minmax_range[0]) / (minmax_range[1]-minmax_range[0])
    J = (J-minmax_range[0]) / (minmax_range[1]-minmax_range[0])

    # and this will make them integers in the [0 (num_bins-1)] range
    I = np.round(I*(num_bins-1)).astype(int)
    J = np.round(J*(num_bins-1)).astype(int)

    n = I.shape[0]
    hist_size = np.array([num_bins,num_bins])

    # initialize the joint histogram to all zeros
    p = np.zeros(hist_size)

    for k in range(n):
        p[I[k], J[k]] = p[I[k], J[k]] + 1

    # Normalize the joint histogram to get the joint probability mass function
    p = p / np.sum(p)
    return p


def mutual_information(p):
    # Compute the mutual information from a joint histogram.
    # Input:
    # p - joint histogram: np.array of size (num_bins, num_bins)
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)  # sum over columns to get the marginal histogram for I
    p_I = p_I.reshape(-1, 1) # reshape to make it a column vector
    p_J = np.sum(p, axis=0)  # sum over rows to get the marginal histogram for J
    p_J = p_J.reshape(1, -1) # reshape to make it a row vector

    # Compute the mutual information using the formula: MI = sum(p * log(p / (p_I * p_J)))
    MI = np.sum(p * np.log(p / (p_I * p_J)))
    return MI

def dice_score(A, B):
    A = A.astype(bool)
    B = B.astype(bool)

    intersection = np.logical_and(A, B).sum()

    return 2 * intersection / (A.sum() + B.sum())

def jaccard_index(A, B):
    A = A.astype(bool)
    B = B.astype(bool)

    intersection = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()

    return intersection / union

def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    g = np.zeros_like(x)

    # Compute the partial derivatives of the function at x using
    # central finite differences.
    # g[k] stores the partial derivative w.r.t. the k-th parameter.
    for k in range(x.size):
        dx = np.zeros_like(x)
        dx[k] = h
        g[k] = (fun(x + 0.5*dx) - fun(x - 0.5*dx)) / h

    return g


def rigid_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with a rigid transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    SCALING = 100

    # the first element is the rotation angle
    T = rotate(x[0])

    # the remaining two element are the translation
    #
    # the gradient ascent/descent method work best when all parameters
    # of the function have approximately the same range of values
    # this is  not the case for the parameters of rigid registration
    # where the transformation matrix usually takes  much smaller
    # values compared to the translation vector this is why we pass a
    # scaled down version of the translation vector to this function
    # and then scale it up when computing the transformation matrix
    Th = t2h(T, x[1:]*SCALING)

    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed
    # moving image
    C = correlation(I, Im_t)

    if return_transform:
        return C, Im_t, Th
    else:
        return C

def affine_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the roation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)
    
    NUM_BINS = 64
    SCALING = 100
    
    # transform the moving image
    Th = affine2h(x[0], x[1], x[2], x[3], x[4], x[5]*SCALING, x[6]*SCALING)
    Im_t, _ = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed moving image
    C = correlation(I, Im_t)
    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_mi(I, Im, x, return_transform=True):
    # Computes mutual information between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    NUM_BINS = 64
    SCALING = 100

    # transform the moving image
    Th = affine2h(x[0], x[1], x[2], x[3], x[4], x[5]*SCALING, x[6]*SCALING)
    Im_t,_ = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed moving image
    p = joint_histogram(I, Im_t, num_bins=NUM_BINS)
    C = mutual_information(p)
    if return_transform:
        return C, Im_t, Th
    else:
        return C

def intensity_based_registration_demo(I_path, Im_path, similarity_func, mu, num_iter, init_val, title):
    # I_path: path to the fixed image
    # Im_path: path to the moving image
    # similarity_func: similarity function 
    # mu: learning rate
    # num_iter: number of iterations 
    # init_val: initial parameter values for the transformation (default = [0, 0, 0] for rigid registration)
    # title: title of the plot

    # Read the fixed and moving image:
    I = plt.imread(I_path)
    Im = plt.imread(Im_path)
    # Initial parameter values:
    x = init_val
    # NOTE: for affine registration you have to initialize more parameters and 
    # the scaling parameters should be initialized to 1 instead of 0.
    # Similarity function:
    fun = lambda x: similarity_func(I, Im, x, return_transform=False)
    # Learning rate:
    mu = mu
    # Number of iterations
    num_iter = num_iter
    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)
    fig = plt.figure(figsize=(14,6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)
    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.05, 0.95,
        np.array2string(x, precision=3, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)
    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))
    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()
    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
        # gradient ascent
        g = ngradient(fun, x)
        x += g*mu
        # for visualization of the result
        S, Im_t, Th = similarity_func(I, Im, x, return_transform=True)
        clear_output(wait = True)
        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))
        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)
        display(fig)
    plt.close(fig)
    
    return S, Im_t, Th

def load_manually_selected_control_points():
     """ In order to verify the acquired results, this function could be used
         to load the manually selected control points across all nine pairs
         of (T1,T2)-weighted images (three slices for each of the three subjects).
         The control points form a list with 9 tuples, where each tuple consists
         of two lists: the x-coordinates, and the y-coordinates. """
     return [(np.array([[125.96451613, 161.73225806, 144.7, 130.22258065, # pair 1
                83.38387097,  79.97741935,  93.60322581, 113.61612903,
                183.87419355, 195.79677419, 216.23548387, 144.27419355,
                142.57096774, 211.5516129 , 195.37096774, 206.86774194,
                172.80322581, 209.42258065,  76.14516129, 202.18387097],
                [105.49464516, 104.21722581, 134.87529032,  49.714     ,
                83.35270968, 133.59787097, 223.86883871, 241.75270968,
                244.73335484, 227.70109677, 206.41077419, 165.10754839,
                177.88174194, 149.35270968, 112.30754839,  90.59141935,
                61.63658065, 167.23658065, 173.62367742, 193.21077419]]),
            np.array([[123.13870968, 157.62903226, 139.31935484, 130.80322581,
                81.40967742,  75.4483871 ,  83.53870968, 100.99677419,
                175.93870968, 187.43548387, 206.59677419, 136.33870968,
                136.33870968, 207.87419355, 193.82258065, 205.31935484,
                172.10645161, 203.19032258,  70.33870968, 195.1],
                [120.82367742, 121.24948387, 154.4623871 ,  64.19141935,
                96.12690323, 145.52045161, 237.92045161, 255.37851613,
                260.0623871 , 246.8623871 , 225.14625806, 180.8623871 ,
                193.63658065, 164.68174194, 130.19141935, 110.60432258,
                79.09464516, 183.84303226, 185.12045161, 212.37206452]])),
            (np.array([[126.39032258, 161.73225806, 142.14516129, 114.04193548, # pair 2
                160.45483871, 151.51290323, 184.72580645, 122.13225806,
                139.16451613,  99.56451613, 131.07419355, 125.53870968,
                163.43548387, 220.49354839, 223.9       , 209.42258065,
                69.33225806,  66.77741935, 106.80322581, 147.83548387],
                [114.01077419, 112.30754839, 139.13335484, 178.30754839,
                216.20432258, 165.53335484, 171.06883871, 209.39141935,
                235.79141935, 125.50754839,  96.12690323,  53.97206452,
                54.82367742, 147.64948387, 159.99787097, 195.7656129 ,
                136.15270968, 148.07529032, 236.21722581, 148.07529032]]),
            np.array([[131.65483871, 164.86774194, 144.85483871, 115.47419355,
                159.75806452, 150.39032258, 184.02903226, 122.28709677,
                136.76451613, 100.99677419, 134.63548387, 132.93225806,
                169.97741935, 222.3516129 , 224.90645161, 209.57741935,
                70.76451613,  69.06129032, 103.97741935, 147.83548387],
                [121.67529032, 120.39787097, 146.79787097, 182.99141935,
                225.57206452, 171.92045161, 180.43658065, 216.63012903,
                243.03012903, 129.33980645, 102.08819355,  59.50754839,
                62.914     , 157.86883871, 172.34625806, 204.70754839,
                139.98496774, 152.33335484, 241.75270968, 148.07529032]])),
            (np.array([[132.77741935, 156.62258065, 115.74516129, 177.06129032, # pair 3
                144.7, 120.85483871, 162.58387097, 110.63548387,
                197.07419355,  70.60967742, 128.94516129, 216.66129032,
                185.1516129 , 148.53225806,  97.86129032,  87.64193548,
                97.00967742, 171.52580645, 187.28064516,  77.42258065],
                [113.15916129, 116.13980645, 113.15916129, 123.37851613,
                162.55270968, 204.70754839, 209.39141935, 232.38496774,
                219.61077419, 147.64948387,  57.37851613, 151.48174194,
                95.27529032,  80.37206452, 183.41722581,  90.1656129 ,
                160.84948387, 184.26883871, 142.9656129 , 196.19141935]]),
            np.array([[141.4483871 , 165.71935484, 122.71290323, 186.58387097,
                156.77741935, 134.20967742, 177.21612903, 124.84193548,
                210.42903226,  79.70645161, 132.50645161, 227.88709677,
                192.54516129, 155.92580645, 109.93870968,  95.88709677,
                108.66129032, 183.60322581, 199.78387097,  89.07419355],
                [114.8623871 , 115.28819355, 114.43658065, 122.52690323,
                162.97851613, 204.28174194, 208.114     , 233.23658065,
                219.18496774, 151.48174194,  59.50754839, 147.64948387,
                92.72045161,  78.66883871, 185.54625806,  93.99787097,
                163.40432258, 184.26883871, 141.2623871 , 202.57851613]])), 
            (np.array([[133.62903226, 161.73225806, 129.37096774, 159.60322581, # pair 4
                153.64193548, 114.89354839, 142.57096774,  69.33225806,
                99.99032258, 188.13225806, 217.93870968, 213.25483871,
                152.79032258, 194.51935484, 102.97096774, 139.16451613,
                180.89354839,  76.57096774,  73.16451613, 104.67419355],
                [ 56.95270968,  56.52690323, 104.64303226, 103.79141935,
                85.90754839, 105.92045161, 161.27529032, 148.92690323,
                82.07529032,  99.95916129, 146.79787097, 119.97206452,
                231.10754839, 213.64948387, 169.79141935, 174.47529032,
                149.35270968, 195.7656129 , 114.43658065, 141.2623871]]),
            np.array([[135.48709677, 161.03548387, 128.67419355, 159.75806452,
                154.6483871 , 114.62258065, 141.4483871 ,  67.78387097,
                100.14516129, 188.71290323, 216.81612903, 212.98387097,
                146.55806452, 190.84193548, 100.99677419, 138.89354839,
                181.9,  73.74516129,  71.61612903, 104.82903226],
                [ 62.0623871 ,  62.0623871 , 108.90109677, 108.90109677,
                92.72045161, 110.60432258, 165.53335484, 153.18496774,
                87.61077419, 106.34625806, 155.73980645, 126.78496774,
                236.64303226, 220.03658065, 175.32690323, 178.73335484,
                154.88819355, 200.02367742, 119.54625806, 147.64948387]])),
            (np.array([[128.09354839, 159.60322581, 135.75806452, 150.23548387, # pair 5
                116.17096774, 167.69354839, 216.23548387, 211.5516129 ,
                190.26129032, 133.62903226, 111.06129032, 150.66129032,
                74.86774194, 112.33870968, 104.2483871 ,  79.5516129 ,
                155.77096774, 106.80322581, 188.98387097, 207.29354839],
                [109.32690323, 109.75270968, 132.74625806, 134.02367742,
                183.84303226, 187.67529032, 148.07529032, 114.01077419,
                78.24303226,  56.95270968, 100.38496774,  68.44948387,
                186.39787097, 135.30109677, 154.88819355, 102.514     ,
                226.42367742, 185.54625806, 205.55916129, 192.78496774]]),
            np.array([[129.9516129 , 159.75806452, 136.33870968, 152.51935484,
                115.9       , 168.27419355, 217.24193548, 212.55806452,
                192.11935484, 134.20967742, 109.51290323, 150.81612903,
                75.87419355, 112.06774194, 103.5516129 ,  80.13225806,
                156.3516129 , 106.95806452, 191.26774194, 208.72580645],
                [115.28819355, 115.28819355, 140.41077419, 141.68819355,
                187.67529032, 197.04303226, 154.4623871 , 119.12045161,
                84.20432258,  63.7656129 , 105.92045161,  73.13335484,
                193.63658065, 141.2623871 , 158.72045161, 107.19787097,
                232.81077419, 191.08174194, 209.39141935, 197.46883871]])),
            (np.array([[128.94516129, 125.53870968, 158.32580645, 158.7516129 , # pair 6
                130.6483871 , 165.99032258, 183.02258065, 213.68064516,
                200.48064516, 178.33870968, 190.68709677, 200.05483871,
                157.47419355,  75.29354839,  69.75806452,  82.95806452,
                122.13225806, 145.5516129 ,  99.99032258, 154.91935484],
                [134.87529032, 148.07529032, 153.18496774, 134.87529032,
                61.63658065,  64.19141935,  80.37206452, 148.92690323,
                125.08174194, 176.60432258, 213.64948387, 153.18496774,
                192.78496774, 182.99141935, 150.63012903, 104.21722581,
                78.24303226,  75.68819355, 154.03658065, 105.49464516]]),
            np.array([[141.02258065, 139.31935484, 171.25483871, 171.25483871,
                140.59677419, 174.23548387, 192.54516129, 227.03548387,
                212.98387097, 192.97096774, 207.02258065, 213.40967742,
                174.23548387,  90.3516129 ,  83.53870968,  94.18387097,
                132.50645161, 155.07419355, 112.49354839, 165.71935484],
                [136.15270968, 147.22367742, 152.75916129, 134.02367742,
                63.7656129 ,  64.61722581,  79.09464516, 145.09464516,
                122.52690323, 175.32690323, 211.09464516, 151.48174194,
                192.78496774, 186.39787097, 154.03658065, 108.90109677,
                78.24303226,  75.68819355, 155.73980645, 105.49464516]])),
            (np.array([[ 76.99677419,  86.36451613, 138.73870968, 214.95806452, # pair 7    
                188.13225806, 197.92580645, 112.33870968, 141.71935484,
                160.88064516, 163.86129032, 104.2483871 ,  65.92580645,
                198.3516129 , 141.71935484,  99.56451613, 200.48064516,
                77.42258065, 203.46129032, 132.77741935,  84.23548387],
                [196.19141935, 209.81722581,  50.5656129 , 177.45593548,
                224.29464516, 213.22367742, 233.6623871 , 138.70754839,
                89.73980645,  65.89464516,  72.28174194, 140.83658065,
                138.28174194, 155.73980645, 170.64303226, 122.95270968,
                86.75916129,  95.27529032,  66.32045161, 123.80432258]]),
            np.array([[ 79.28064516,  89.92580645, 144.00322581, 220.22258065,
                191.69354839, 201.91290323, 116.32580645, 146.55806452,
                167.8483871 , 170.82903226, 110.79032258,  71.19032258,
                206.17096774, 146.55806452, 103.97741935, 208.3,
                83.11290323, 210.00322581, 140.17096774,  88.6483871],
                [201.30109677, 214.92690323,  56.95270968, 184.26883871,
                231.10754839, 219.61077419, 239.19787097, 143.81722581,
                95.27529032,  72.28174194,  77.81722581, 146.37206452,
                144.24303226, 159.14625806, 175.32690323, 129.33980645,
                91.01722581, 102.08819355,  71.00432258, 128.914]])),
            (np.array([[118.72580645, 138.31290323, 155.77096774, 139.59032258, # pair 8
                109.35806452, 217.93870968, 217.08709677, 197.07419355,
                136.18387097,  65.92580645, 120.00322581, 176.20967742,
                196.22258065, 105.9516129 , 105.52580645,  90.19677419,
                97.86129032, 200.05483871, 116.17096774, 187.70645161],
                [105.49464516, 109.32690323, 102.08819355, 125.50754839,
                230.25593548, 155.73980645, 137.00432258,  83.77851613,
                80.37206452, 163.40432258, 172.34625806, 177.03012903,
                162.12690323, 152.75916129,  95.27529032, 211.52045161,
                133.59787097, 207.68819355,  73.55916129, 127.63658065]]),
            np.array([[130.37741935, 150.39032258, 166.99677419, 152.09354839,
                124.84193548, 231.29354839, 230.01612903, 209.1516129 ,
                149.11290323,  78.85483871, 133.78387097, 189.99032258,
                210.00322581, 118.45483871, 117.60322581, 103.97741935,
                112.06774194, 215.53870968, 126.54516129, 200.20967742],
                [106.34625806, 109.32690323, 102.08819355, 124.23012903,
                231.10754839, 154.03658065, 135.30109677,  82.07529032,
                78.66883871, 165.53335484, 173.62367742, 178.30754839,
                160.84948387, 153.61077419,  94.84948387, 212.37206452,
                134.87529032, 206.41077419,  73.55916129, 125.08174194]])),
            (np.array([[114.04193548, 123.83548387, 156.62258065, 171.52580645, # pair 9
                141.29354839, 146.82903226, 215.38387097, 216.23548387,
                194.51935484,  69.75806452, 112.33870968, 157.47419355,
                138.73870968, 105.1       ,  95.73225806, 173.22903226,
                122.55806452,  92.32580645, 180.46774194, 136.18387097],
                [174.47529032, 122.52690323, 116.5656129 , 175.32690323,
                134.44948387,  56.10109677, 139.13335484, 165.95916129,
                211.94625806, 168.93980645, 227.27529032, 207.2623871 ,
                117.41722581, 117.41722581,  73.55916129,  97.83012903,
                75.68819355, 130.19141935, 187.67529032, 208.53980645]]),
            np.array([[109.51290323, 124.84193548, 154.22258065, 168.7       ,
                139.31935484, 149.96451613, 213.83548387, 213.83548387,
                188.28709677,  66.08064516, 106.95806452, 151.24193548,
                138.46774194, 103.5516129 ,  96.31290323, 176.79032258,
                123.56451613,  89.5       , 177.64193548, 129.9516129 ],
                [185.97206452, 134.44948387, 132.32045161, 189.37851613,
                145.52045161,  70.57851613, 157.01722581, 185.12045161,
                228.97851613, 178.30754839, 240.90109677, 223.01722581,
                131.89464516, 128.0623871 ,  85.90754839, 112.30754839,
                87.61077419, 140.41077419, 204.28174194, 221.73980645]]))]