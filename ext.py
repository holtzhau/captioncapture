import numpy as N
import cv
#from _CVtypes import IplImage, ListPOINTER, CvMat, IplImage, CvRect, CvContour, CvSeq
from ctypes import *
import ctypes

LESSTHAN = 0
GREATERTHAN = 1

_ext = CDLL('./libext.so')
# offset and size of a rectangle
class CvRect(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("height", c_int)]
# Sequence
class CvSeq(Structure):
    _fields_ = [("flags", c_int),
                ("header_size", c_int),
                ("h_prev", c_void_p),
                ("h_next", c_void_p),
                ("v_prev", c_void_p),
                ("v_next", c_void_p),
                ("total", c_int),
                ("elem_size", c_int),
                ("block_max", c_void_p),
                ("ptr", c_void_p),
                ("delta_elems", c_int),
                ("storage", c_void_p),
                ("free_blocks", c_void_p),
                ("first", c_void_p)]                
    
    def as_contour(self):
        return ctypes.cast(ctypes.pointer(self), POINTER(CvContour))[0]
    
    def hrange(self):
    	"""
    	generator function iterating along h_next
    	"""
    	s = ctypes.pointer(self)
    	t = type(self)
    	while s:
    	    yield s[0]
    	    s = ctypes.cast(s[0].h_next , POINTER(CvSeq))
    	    
#Added the fields for contour - Start
class CvContour(Structure):
    _fields_ = [("flags", c_int),
                ("header_size", c_int),
                ("h_prev", c_void_p),
                ("h_next", POINTER(CvSeq)),
                ("v_prev", c_void_p),
                ("v_next", c_void_p),
                ("total", c_int),
                ("elem_size", c_int),
                ("block_max", c_void_p),
                ("ptr", c_void_p),
                ("delta_elems", c_int),
                ("storage", c_void_p),
                ("free_blocks", c_void_p),
                ("first", c_void_p),
                ('rect', CvRect),
                ("color", c_int),
                ("reserved", c_int * 3)]


class WrappedObject(Structure):
    _fields_ = [
                ("PyObject_HEAD", c_byte * (object.__basicsize__)),
                ("ctx", c_void_p),
                ("base", c_void_p),
                ("size", c_int)
            ]

def wrapped(x):
    return WrappedObject.from_address(id(x)).ctx

def as_contour(seq):
    return CvSeq.from_address(wrapped(seq)).as_contour()


_ext.filter_contours.restype = c_void_p
_ext.filter_contours.argtypes = [c_void_p, c_int, c_int]
def filter_contours(seq, number, operation):
    w = WrappedObject.from_address(id(seq))
#    print w, w.ctx, w.base, id(seq), len(seq)
    if len(seq) > 0 and w.ctx:
        c = _ext.filter_contours(c_void_p(w.ctx), number, operation)
        w.ctx = c
        return w
    else:
        return None

	
