import cv
from cv import *
import numpy
import ctypes
import traceback, os
import time
import logging
import StringIO
from PIL import Image
from collections import deque
import numpy as np
import ext

logging.basicConfig(
	level=logging.DEBUG,
	datefmt="%H:%M",
	)
logger = logging.getLogger("log")


def contour_to_rect(contours):
    rects = []
    if contours:
        seq = contours
        while seq:
            c = ext.as_contour(ext.wrapped(seq))
            r = (c.rect.x, c.rect.y, c.rect.width, c.rect.height)
            rects.append((r, seq))
            seq = seq.h_next()
    return rects
    
    
def merge(rect1, rect2):
    x = min(rect1[0], rect2[0])
    y = min(rect1[1], rect2[1])
    x2 = max(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = max(rect1[1] + rect1[3], rect2[1] + rect2[3])
    return (x, y, x2 - x, y2 - y)
    
    
def overlaps(rect1, rect2, delta=(0, 0)):
    left1 = rect1[0] - delta[0]
    right1 = rect1[0] + rect1[2] + delta[0]
    top1 = rect1[1] - delta[1]
    bottom1 = rect1[1] + rect1[3] + delta[1]
    left2 = rect2[0] - delta[0]
    right2 = rect2[0] + rect2[2] + delta[0]
    top2 = rect2[1] - delta[1]
    bottom2 = rect2[1] + rect2[3] + delta[1]
    return not (left2 > right1 or right2 < left1 or top2 > bottom1 or bottom2 < top1)


def intersect(r1, r2):
    intersection = [0, 0, 0, 0]
    # find overlapping region 
    intersection[0] = r2[0] if r1[0] < r2[0] else r1[0] 
    intersection[1] = r2[1] if r1[1] < r2[1] else r1[1]  
    intersection[2] = r1[0] + r1[2] if r1[0] + r1[2] < r2[0] + r2[2] else r2[0] + r2[2]
    intersection[2] -= intersection[0]; 
    intersection[3] = r1[1] + r1[3] if r1[1] + r1[3] < r2[1] + r2[3] else r2[1] + r2[3]
    intersection[3] -= intersection[1]; 
    # check for non-overlapping regions 
    if (intersection[2] <= 0) or (intersection[3] <= 0):
        intersection = [0, 0, 0, 0] 
    return intersection
    
    
def get_patch(image, rect, pad=0):
    min_x, min_y = max(rect[0] - pad, 0), max(rect[1] - pad, 0)
    max_x, max_y = min(rect[0] + rect[2] + pad, image.width), min(rect[1] + rect[3] + pad, image.height)
    rect = (min_x, min_y, max_x - min_x, max_y - min_y)
    cv.SetImageROI(image, tuple(int(k) for k in rect))
    patch = cv.CreateImage((int(rect[2]), int(rect[3])), 8, 1)
    cv.Copy(image, patch)
    cv.ResetImageROI(image)    
    return patch

class OpencvRecorder:
    """ 
    Default OpenCV recording device.
    """
    def __init__(self, destination, framerate=25, size=None, codec=0):
        """
        Args:
            size (tuple): Image size.
            framerate (int): Video frame rate.
            codec (int): Opencv codec ID.
        """        
        self.size = size
        self.destination = destination
        self.framerate = framerate
        self.codec = codec
        self.frame_num = 0
        self.writer = None
    
    def write(self, image):
        """
        Write frame.
        
        cv_image (IplImage): Destination image.
        """
        if not self.writer:
            if not self.size:
                self.size = cv.GetSize(image)
            self.writer = cv.CreateVideoWriter(self.destination, self.codec, self.framerate, self.size, 1)
        cv.WriteFrame(self.writer, image);
        self.frame_num += 1

def num(mat):
    return np.asarray(cv.GetMat(mat))


def num2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im


def cv2num(im):
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }
  arrdtype=im.depth
  a = np.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
  a.shape = (im.height,im.width,im.nChannels)
  return a

       
def regions_to_mask(regions, mask, inverted=False, value=255):
    if inverted:
        cv.Set(mask, cv.ScalarAll(value))
        colour = cv.ScalarAll(0)
    else:
        cv.Zero(mask)
        colour = cv.ScalarAll(value)
    for rect in regions:
        if not hasattr(rect, "x"):
            rect = rect.rect
        cv.Rectangle(mask, (rect.x, rect.y), (rect.x + rect.width, rect.y + rect.height), colour, -1)
     
     

class OpencvRecorder:
    """ 
    Default OpenCV recording device.
    """
    def __init__(self, destination, framerate=25, size=None, codec=0):
        """
        Args:
            size (tuple): Image size.
            framerate (int): Video frame rate.
            codec (int): Opencv codec ID.
        """        
        self.size = size
        self.destination = destination
        self.framerate = framerate
        self.codec = codec
        self.frame_num = 0
        self.writer = None
    
    def write(self, image):
        """
        Write frame.
        
        cv_image (IplImage): Destination image.
        """
        if not self.writer:
            if not self.size:
                self.size = cv.GetSize(image)
            self.writer = cv.CreateVideoWriter(self.destination, self.codec, self.framerate, self.size, 1)
        cv.WriteFrame(self.writer, image);
        self.frame_num += 1
    
        
if __name__ == "__main__":
    cvNamedWindow('inspect', CV_WINDOW_AUTOSIZE)
    a = numpy.eye(3)
    b = cvNumpyAsMat(a)
    print cvGetReal2D(b, 0, 0)
