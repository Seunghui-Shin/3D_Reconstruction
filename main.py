import numpy as np
import cv2
import glob
import open3d as o3d

supervise_mode = True

# Your information here
name = '신승희'
student_id = '2018102114'

if supervise_mode:
    print('name:%s id:%s'%(name, student_id))

# ====================================================================================
# Camera calibration
# ====================================================================================
# Set directory path (images capturing check pattern)
# Example) calibration_dir_path = 'calibration/*.png'
calibration_dir_path = 'calibration/*.jpeg'
calibration_images = glob.glob(calibration_dir_path)

# intrinsic parameters and distortion coefficient
# With these parameter, you can get undistorted image and new intrinsic parameter of them (K_undist)
K = np.array([], dtype=np.float32)
dist = np.array([], dtype=np.float32)
# new matrix for undistorted intrinsic parameter
K_undist = np.array([], dtype=np.float32)

# Your code here
# Goals
# 1. Get camera intrinsic parameters from your captured images K, dist, K_undist
# 2. Try to get undistorted images by warping captured images using K_undist
# reference: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

# ****************************** Your code here (M-1) ******************************
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = []
imgpoints = []

for fname in calibration_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
cv2.destroyAllWindows()

img = cv2.imread('calibration/near_front.jpeg')
h,  w = img.shape[:2]
K_undist, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
# **********************************************************************************

if supervise_mode:
    print('1-1. Calibration: K matrix')
    print(K)
    print('1-2. Calibration: distortion coefficients')
    print(dist)
    print('1-3. Calibration: Undistorted K matrix')
    print(K_undist)
    for fname in calibration_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        img_undist = cv2.undistort(gray, K, dist, None, K_undist)
        cv2.imshow('undistorted image', img_undist)
        cv2.waitKey(0)

# ====================================================================================
# load stereo images (Left and Right)
# ====================================================================================
#  set your left and right images
# Example
# imgL = cv2.imread('stereo/left.png')
# imgR = cv2.imread('stereo/right.png')

imgL = cv2.imread('stereo/left.jpeg')
imgR = cv2.imread('stereo/right.jpeg')

# convert to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Convert to undistorted images
# imgLU: undistorted image of imgL
# imgRU: undistorted image of imgR
# grayLU: undistorted image of grayL
# grayRU: undistorted image of grayR
imgLU = np.array([])
imgRU = np.array([])
grayLU = np.array([])
grayRU = np.array([])

# undistorted images
# ****************************** Your code here (M-2) ******************************

imgLU = cv2.undistort(imgL, K, dist, None, K_undist)
imgRU = cv2.undistort(imgR, K, dist, None, K_undist)
grayLU = cv2.undistort(grayL, K, dist, None, K_undist)
grayRU = cv2.undistort(grayR, K, dist, None, K_undist)

# **********************************************************************************

if supervise_mode:
    cv2.imshow('rgb undistorted', cv2.hconcat([imgLU, imgRU]))
    cv2.imshow('gray undistorted', cv2.hconcat([grayLU, grayRU]))
# ====================================================================================
# stereo matching (Dense matching)
# ====================================================================================
# Goals
#  1. Get disparity map (8 bit unsigned)
#  Note. The output of disparity function (StereoBM, etc.) is 16-bit
#
# reference: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_depthmap.html
#
disp8 = np.array([], np.uint8)

# ****************************** Your code here (M-3) ******************************

import numpy as np
import cv2
from matplotlib import pyplot as plt

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(grayLU, grayRU)
disp8 = disparity.astype('uint8')
plt.imshow(disparity,'gray')
plt.show()

# **********************************************************************************

if supervise_mode:
    imgLU[disp8 < 1, :] = 0  # min value -> 0
    imgLU[disp8 > 239, :] = 0  # max value -> 0
    cv2.imshow('disparity', disp8)
    cv2.imshow('Left Post-processing', imgLU)
    cv2.waitKey(0)

# ====================================================================================
# Visualization
# ====================================================================================
# In advance, you should install open3D (open3d.org)
# pip install open3d
#

pcd = o3d.geometry.PointCloud()

#  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
#  pc_color: array(Nx3), each row composed with R G,B in the rage of 0~1
pc_points = np.array([], np.float32)
pc_color = np.array([], np.float32)

# 3D reconstruction
# Concatenate pc_points and pc_color
# ****************************** Your code here (M-4) ******************************
h, w = imgLU.shape[:2]
focal_length = 0.08 * w

Q = np.float32([[1,0,0,-0.5*w],
    [0,-1,0,0.5*h],
    [0,0,focal_length*0.05,0],
    [0,0,0,1]])

points_3D = cv2.reprojectImageTo3D(disp8, Q)
colors = cv2.cvtColor(imgLU, cv2.COLOR_BGR2RGB)/255

mask_map = disp8 > disp8.min()
mask_map = disp8 < disp8.max()
pc_points = points_3D[mask_map]
pc_color = colors[mask_map]

# **********************************************************************************

#  add position and color to point cloud
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_color)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
cv2.destroyAllWindows()

#  end of code