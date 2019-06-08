import cv2
import numpy as np
import math
#import picamera
import io
from IPython.display import Image
import time
import glob
from termcolor import colored
import math
import re
import cv2
import numpy as np
from matplotlib import pyplot as plt
def findCenter(x1,x2,y1,y2,boundaries,center):
    threshImg = cv2.imread('threshold_img.png') 

    start = time.time()
    threshImg = cv2.resize(threshImg, (0,0), fx=0.5, fy=0.5)
    threshImg  = cv2.flip( threshImg, 0 )
    thresh_img = threshImg.copy()
    # define the list of boundaries
    
    masks = []
    images = []
    titles = []
    angles = []
    # loop over the boundaries
    for x in range(x1,x2):
        for y in range(y1,y2):
            angles = []
            for (lower, upper, title, color) in boundaries:
                # create NumPy arrays from the boundaries
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")

                # find the colors within the specified boundaries and apply
                # the mask
                mask = cv2.inRange(threshImg, lower, upper)
                kernel = np.ones((3,3),np.uint8)

                mask = cv2.dilate(mask,kernel,iterations = 6)

                output = cv2.bitwise_and(threshImg, threshImg, mask = mask)
                gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.GaussianBlur(gray_image,(5,5),1)



                # find contours in the binary image
                contours, _ = cv2.findContours(gray_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                c = max(contours, key = cv2.contourArea)
                temp = c.reshape(c.shape[0],2)
                closest = closest_node(center, temp)
                cX, cY = temp[closest]
                # calculate moments for each contour
                '''M = cv2.moments(c)

                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])'''

                vect1 = findVec((x,y),(x-200,y))
                vect2 = findVec((x,y), (cX ,cY))
                angles.append(math.degrees(calcul_angle(vect1,vect2)))
            if angles[0] > 44.9 and angles[0] < 45.1 and angles[1] > 171.8 and angles[1] < 172:
                print("angles: ", angles)
                print("found: ",x,y)
                break

    print("done")
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def farthest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmax(dist_2)
	
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
	
def findVec(point1,point2,unitSphere = False):
    #setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
    finalVector = [0 for coOrd in point1]
    for dimension, coOrd in enumerate(point1):
        #finding total differnce for that co-ordinate(x,y,z...)
        deltaCoOrd = point2[dimension]-coOrd
        #adding total difference
        finalVector[dimension] = deltaCoOrd
    if unitSphere:
        totalDist = multiDimenDist(point1,point2)
        unitVector =[]
        for dimen in finalVector:
            unitVector.append( dimen/totalDist)
            return unitVector
    else:
        return finalVector
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))
   
def calcul_angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
	
def nothing(x):
    pass
def findThreshold():
    cv2.namedWindow('Colorbars')
    hmax='hmax'
    hmin='hMin'
    smax='sMax'
    smin='sMin'
    vmax='vMax'
    vmin='vMin'

    wnd = 'Colorbars'
    cv2.createTrackbar("hMax", "Colorbars",0,255,nothing)
    cv2.createTrackbar("hMin", "Colorbars",0,255,nothing)
    cv2.createTrackbar("sMax", "Colorbars",0,255,nothing)
    cv2.createTrackbar("sMin", "Colorbars",0,255,nothing)
    cv2.createTrackbar("vMax", "Colorbars",0,255,nothing)
    cv2.createTrackbar("vMin", "Colorbars",0,255,nothing)

    img = cv2.imread('threshold_img.png') 
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    while(1):
        hmax=cv2.getTrackbarPos("hmax", "Colorbars")
        hmin=cv2.getTrackbarPos("hMin", "Colorbars")
        smax=cv2.getTrackbarPos("sMax", "Colorbars")
        smin=cv2.getTrackbarPos("sMin", "Colorbars")
        vmax=cv2.getTrackbarPos("vMax", "Colorbars")
        vmin=cv2.getTrackbarPos("vMin", "Colorbars")

        lower = np.array((hmin,smin,vmin), dtype = "uint8")
        upper = np.array((hmax,smax,vmax), dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(img, lower, upper)
        kernel_dil = np.ones((3,3),np.uint8)

        mask = cv2.dilate(mask,kernel_dil,iterations = 4)
        '''mask = cv2.erode(mask,kernel_erod,iterations = 3)    
        mask = cv2.dilate(mask,kernel_dil,iterations = 1)
    '''
        output = cv2.bitwise_and(img, img, mask = mask)

        # cv2.imshow(wnd)
        cv2.imshow("img",img)
        cv2.imshow("output",output)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    cv2.destroyAllWindows()
	
def find_angles(threshImg,center):
    start = time.time()
    threshImg = cv2.resize(threshImg, (0,0), fx=0.5, fy=0.5)
    threshImg  = cv2.flip( threshImg, 0 )
    thresh_img = threshImg.copy()
    # define the list of boundaries
    '''boundaries = [
         ([0, 0, 162], [255, 0, 255], 'r', (0,0,255)),
         ([0, 37, 51], [255, 255, 60], 'y', (0,255,255)),
         ([255, 0, 0], [255, 0, 255], 'b', (255,0,0)),
         ([0, 178, 0], [255, 181, 232], 'g', (0,255,0))
    ]'''

    boundaries = [
         ([0, 0, 162], [255, 0, 255], 'r', (0,0,255)),
         ([0, 37, 51], [255, 255, 60], 'y', (0,255,255)),
         ([255, 0, 0], [255, 0, 255], 'b', (255,0,0)),
         ([0, 181, 0], [255, 183, 42], 'g', (0,255,0))
    ]
    masks = []
    images = []
    titles = []
    angles = []
    # loop over the boundaries
    for (lower, upper, title, color) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(threshImg, lower, upper)
        kernel = np.ones((3,3),np.uint8)

        mask = cv2.dilate(mask,kernel,iterations = 6)

        output = cv2.bitwise_and(threshImg, threshImg, mask = mask)
        gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image,(5,5),1)



        # find contours in the binary image
        contours, _ = cv2.findContours(gray_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)
        temp = c.reshape(c.shape[0],2)
        closest = closest_node(center, temp)
        cX, cY = temp[closest]
        # calculate moments for each contour
        '''M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])'''


        vect1 = findVec(center,(center[0]-200,center[1]))
        vect2 = findVec(center, (cX ,cY))
        angle = math.degrees(calcul_angle(vect1,vect2))
        angles.append(angle)
        cv2.line(thresh_img,(cX ,cY),center,color, 2)    

        cv2.circle(thresh_img, center, 5, (255, 255, 255), -1)
        # show the images
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        images.append(output)
        titles.append(title + ", angle: " + str(angle))
        masks.append(mask)

    cv2.line(thresh_img,(center[0]-200,center[1]),center,(255,255,255), 2)    

    thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2RGB)
    images.append(thresh_img)
    titles.append('Original')
    i = 0
    print("time: ",time.time() - start)
    print(angles)
    show_images(images, cols = 3, titles = titles)
    return angles