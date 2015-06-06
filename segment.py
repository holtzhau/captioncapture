import cv, cv2
import numpy as np
import os
import ext
from random import randint
import subprocess
from tools import get_patch, overlaps, intersect, contour_to_rect, merge, cv2num 
from tools import OpencvRecorder
import json
import string
import time
import sys

def in_vertical(seed, start, rect):
    if abs(start[3]-rect[3]) / float(start[3]) < 0.4:
        seed_mid = seed[1] + seed[3]/2
        if seed_mid - start[3]/2 < rect[1] + rect[3]/2 < seed_mid + start[3]/2:
            return True
    return False

def detect(image, debug=False, display=None):
    work_image = cv.CreateImage((image.width, image.height), 8, 1)
    cv.CvtColor(image, work_image, cv.CV_BGR2GRAY)
    image = work_image
    edge = cv.CloneImage(image)    
    thresholded = cv.CloneImage(image)    
    v_edges = cv.CloneImage(image)    
    h_edges = cv.CloneImage(image)    
    vertical = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_16S, 1)
    cv.Sobel(image, vertical, 1, 0, 1)
    cv.Abs(vertical, vertical) 
    cv.Convert(vertical, v_edges)
    storage = cv.CreateMemStorage(0)
    result = np.asarray(cv.GetMat(v_edges), dtype=np.float)        
    threshold = 6
    rects = []

    while len(rects) < 1 and threshold > 0:
        rects = []
        cv.Convert(vertical, v_edges)
#        cv.Threshold(v_edges, v_edges, threshold, 255, cv.CV_THRESH_BINARY)
        cv.AdaptiveThreshold(v_edges, v_edges, 255, cv.CV_ADAPTIVE_THRESH_MEAN_C, cv.CV_THRESH_BINARY_INV, 17, threshold)
        # # if not enough edge response, repeat at lower threshold
        # if cv.Sum(v_edges)[0]/255/(v_edges.width*v_edges.height) < 0.03:
        #     cv.Convert(vertical, v_edges)
        #     cv.AdaptiveThreshold(v_edges, v_edges, 255, cv.CV_ADAPTIVE_THRESH_MEAN_C, cv.CV_THRESH_BINARY_INV, 17, 4)           
#        threshold -= 1        
        storage = cv.CreateMemStorage(0)
        contour_image = cv.CloneImage(v_edges)
        contours = cv.FindContours(contour_image, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_NONE, (0,0))
        ext.filter_contours(contours, 30, ext.LESSTHAN)
        max_size = int(image.width * image.height * 0.1)
        # ext.filter_contours(contours, 200**2, ext.GREATERTHAN)
        ext.filter_contours(contours, max_size, ext.GREATERTHAN)

        if display:
            cv.Merge(v_edges, v_edges, v_edges, None, display)
        seeds = []
        if contours:
            seq = contours
            rects = []
            while seq:
                c = ext.as_contour(ext.wrapped(seq))                
                r = (c.rect.x, c.rect.y, c.rect.width, c.rect.height)
                rects.append(r)
                if display:            
                    cv.Rectangle(display, (c.rect.x, c.rect.y), (c.rect.x + c.rect.width, c.rect.y + c.rect.height),
                        (0,0,255), 1)
                seq = seq.h_next()
    rects.sort(lambda x, y: cmp(x[0] + x[2]/2, y[0] + y[2]/2))
    seeds = rects[:]
    seeds.sort(lambda x, y: cmp(y[2]*y[3], x[2]*x[3]))
    groups = []
    skip = False
    for seed in seeds:        
        if seed not in rects:
            break
        found = False
        for group in groups:
            if seed in group:
                found = True
        if found:
            continue
        r = seed

        start = seed
        start_index = rects.index(seed)
        groups.append([seed])
        i = start_index - 1     
        # delta = max(150, seed[2]/2)
        delta = seed[2] * 0.66
        if debug:
            print "left", seed, delta
            col = (randint(0,255), randint(0,255), randint(0,255))
            cv.Rectangle(display, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255,255,255), 3)
            cv.Rectangle(display, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), col, -1)
            cv.ShowImage("main", display)
            if not skip:
                c = cv.WaitKey(0)
            if c == ord("a"):
                skip = True             
        # scan left
        while 1:
            if i < 0:
                break
            rect = rects[i]
            if rect[0]+rect[2] < seed[0]-delta:
                if debug:
                    print "esc1", rect
                break
            if in_vertical(seed, start, rect):
                seed = rect
                groups[-1].append(rect)
                r = rect                     
                if debug:
                    print rect
                    cv.Rectangle(display, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), col, -1)
                    cv.ShowImage("main", display)
                    if not skip:
                        c = cv.WaitKey(0)
                    if c == ord("a"):
                        skip = True  
            else:
                if debug:
                    print "rej1", rect
            i -= 1
        # scan right
        seed = start
        start_index = rects.index(seed)
        i = start_index + 1
        if debug:
            print
            print "right", seed
        while 1:
            if i >= len(rects):
                break
            rect = rects[i]
            if rect[0] > seed[0]+seed[2]+delta:
                if debug:
                    print "esc2", rect,  rect[0]+rect[2]/2 , seed[0]+seed[2]/2+delta
                break    
            if in_vertical(seed, start, rect):
                seed = rect
                groups[-1].append(rect)
                r = rect                        
                if debug:
                    print rect
                    cv.Rectangle(display, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), col, -1)
                    cv.ShowImage("main", display)
                    if not skip:
                        c = cv.WaitKey(0)
                    if c == ord("a"):
                        skip = True  
            else:
                if debug:
                    print "rej2", rect
            i += 1
        if debug:
            print       
            
    # find min and max extent of group    
    group_rects = []
    for group in groups:     
        min_x, min_y = 1E6, 1E6
        max_x, max_y = -1, -1
        dev = []
        col = (randint(0,255), randint(0,255), randint(0,255))
        for rect in group:
            r = rect
            if display:
                if r == group[0]:
                    cv.Rectangle(display, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255,255,255), 3)
                cv.Rectangle(display, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), col, -1) 
            min_x = min(min_x, r[0])
            min_y = min(min_y, r[1])
            max_x = max(max_x, r[0] + r[2])
            max_y = max(max_y, r[1] + r[3])
        if display:
            cv.Rectangle(display, (min_x, min_y), (max_x, max_y), (0,255,0), 1)            
        width = max_x - min_x
        height = max_y - min_y
        rect = (min_x, min_y, width, height)
        group_rects.append(rect)
    return group_rects
    


def bounding_rect(rects, source=None, pad=0):
    min_x, min_y, max_x, max_y = 1E6, 1E6, 0, 0
    for r in rects:     
        min_x = min(r[0] - pad, min_x)
        min_y = min(r[1] - pad, min_y)
        max_x = max(r[0] + r[2] + 2*pad, max_x)
        max_y = max(r[1] + r[3] + 2*pad, max_y)
    if max_x == 0 and max_y == 0:
        return None
    min_x, min_y = max(min_x, 0), max(min_y, 0)
    if source:
        w, h = cv.GetSize(source)
        max_x, max_y = min(max_x, w-1), min(max_y, h-1)
    return min_x, min_y, max_x - min_x, max_y - min_y


next = False
def segment_rect(image, rect, debug=False, display=None, target_size=None, group_range=(3, 25)):
    global next
    skip = False
    best_chars = []
    best_threshold = None
    thresholded = cv.CloneImage(image)
    contour_image = cv.CloneImage(image)
    edges = cv.CloneImage(image)

    min_x, min_y, width, height = rect
    cv.SetImageROI(thresholded, rect)
    cv.SetImageROI(contour_image, rect)
    cv.SetImageROI(image, rect)
    cv.SetImageROI(edges, rect)
    
    horizontal = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_16S, 1)
    magnitude32f = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 1)
    vertical = cv.CloneImage(horizontal)
    magnitude = cv.CloneImage(horizontal)
    cv.Sobel(image, horizontal, 0, 1, 3)
    cv.Sobel(image, vertical, 1, 0, 3)
    cv.Pow(horizontal, horizontal, 2)
    cv.Pow(vertical, vertical, 2)
    cv.Add(vertical, horizontal, magnitude)
    cv.Convert(magnitude, magnitude32f)
    cv.Pow(magnitude32f, magnitude32f, 0.5)
    cv.Convert(magnitude32f, edges)
    
    original_rect = rect
    if display:
        cv.SetImageROI(display, rect)    
    for threshold in range(1, 20, 1):
        cv.SetImageROI(thresholded, original_rect)
    #for i in range(30, 60, 1):
        if display:
            cv.Merge(image, image, image, None, display)
        cv.Copy(image, thresholded)
        #cv.Threshold(thresholded, thresholded, i, 255, cv.CV_THRESH_BINARY_INV)
        cv.AdaptiveThreshold(thresholded, thresholded, 255, cv.CV_ADAPTIVE_THRESH_MEAN_C, cv.CV_THRESH_BINARY_INV, 17, threshold)
        #cv.AdaptiveThreshold(thresholded, thresholded, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, i)
        # skip rects greater than 50% thresholded
        summed = cv.Norm(thresholded, None, cv.CV_L1, None) / 255 / thresholded.width / thresholded.height
        if summed > 0.5:
            continue
        if debug:
            cv.ShowImage("edge", thresholded)
        storage = cv.CreateMemStorage(0)
        cv.Copy(thresholded, contour_image)
        contours = cv.FindContours(contour_image, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE, (0,0))
        ext.filter_contours(contours, 20, ext.LESSTHAN)
        groups = []
        rects = []
        edge_counts = []
        overlappings = {}
        if contours:
            seq = contours
            while seq:        
                c = ext.as_contour(ext.wrapped(seq))
                r = (c.rect.x, c.rect.y, c.rect.width, c.rect.height)
                rects.append(r)
                seq = seq.h_next()
            similarity = 0.45 #0.3
            rects.sort(lambda x, y: cmp(y[2]*y[3], x[2]*x[3]))
            for rect in rects:
                if debug:
                    print
                    print "R", rect, len(groups)
                cv.SetImageROI(edges, (original_rect[0] + rect[0], original_rect[1] + rect[1], rect[2], rect[3]))
                edge_count = cv.Sum(edges)[0] / 255 / (rect[2]*rect[3])
                edge_counts.append(edge_count)
#                cv.ShowImage("edges", edges)
#                cv.WaitKey(0)
                if debug and target_size:
                    print "X", target_size, rect
                    print (target_size[0] - rect[2]) / target_size[0]
                    print (target_size[1] - rect[3]) / target_size[1]
                if rect[2] > rect[3] or float(rect[3])/rect[2] < 3./3 or edge_count < 0.1\
                or (rect[2] == image.width and rect[3] == image.height) \
                or (target_size and not 0 < (target_size[0] - rect[2]) / target_size[0] < 0.3 \
                and not 0 < (target_size[1] - rect[3]) / target_size[1] < 0.05):
                    if debug:
                        print "rej", rect[2], ">", rect[3], "edge=", edge_count
                        cv.Rectangle(display, (rect[0], rect[1]),
                        (rect[0] + rect[2], rect[1] + rect[3]),
                        (0,0,255), 1)
                        cv.ShowImage("main", display)
                        if not skip and not next:
                            c = cv.WaitKey(0)
                            if c == ord("a"):
                                skip = True
                            if c == ord("z"):
                                next = True
                    continue
                added = False
                for group_id, group in enumerate(groups):
                    avg_width, avg_height, avg_y = 0, 0, 0
                    overlap = None
                    c = 0
                    for r in group:
                        avg_y += r[1] + r[3]/2.0
                        avg_width += r[2]    
                        avg_height += r[3]
                        irect = intersect(r, rect)
                        if irect[2]*irect[3] > 0.2*r[2]*r[3]:
                            overlappings.setdefault(group_id, []).append([r, rect])
                    avg_y /= float(len(group))
                    avg_width /= float(len(group))
                    avg_height /= float(len(group))
                    if debug:
                        print group
                    if (abs(avg_width - rect[2]) / avg_width < similarity or \
                     (rect[2] < avg_width)) and \
                    abs(avg_height - rect[3])/ avg_height < similarity and \
                    abs(avg_y - (rect[1] + rect[3]/2.0)) / avg_y < similarity:
                        group.append(rect)
                        added = True
                    else:
                        pass
                if not added:
                    # first char in group
                    groups.append([rect])
                if debug:
                    print "now:"
                    for g in groups:
                        print g
                    cv.Rectangle(display, (rect[0], rect[1]),
                        (rect[0] + rect[2], rect[1] + rect[3]),
                        (255,0,0), 1)
                    cv.ShowImage("main", display)
                    if not skip and not next:
                        c = cv.WaitKey(0)
                        if c == ord("a"):
                            skip = True
                        if c == ord("z"):
                            next = True
        if groups:
            #handle overlapping regions, default to average width match
            for group_id, over in overlappings.items():
                group = groups[group_id]
                avg_width = 0
                avg_height = 0
                for r in group:
                    avg_width += r[2]
                    avg_height += r[3]   
                avg_width /= float(len(group))
                avg_height /= float(len(group))
                for r1, r2 in over:
                    if r2 not in group or r1 not in group:
                        continue
                    if debug:
                        print "over", r1, r2, r1[2]*r1[3], r2[2]*r2[3], avg_width
                    d1 = abs(r1[2] - avg_width) + abs(r1[3] - avg_height)
                    d2 = abs(r2[2] - avg_width) + abs(r2[3] - avg_height)
                    if d1 < d2:
                        group.remove(r2)
                    else:
                        group.remove(r1)
                        
            #group = max(groups, key=len)
            # from longest groups, find largest area
            groups.sort(key=len)
            groups.reverse()
            max_area = 0
            mad_index = -1
            for i, g in enumerate(groups[:5]):
                area = 0
                for r in g:
                    area += r[2]*r[3]
                if area > max_area:
                    max_area = area
                    max_index = i
            group = groups[max_index]
            # vertical splitting
            avg_width, avg_height, avg_y = 0, 0, 0
            if debug:
                print "G", group
            for r in group:
                avg_y += r[1] + r[3]/2.0
                avg_width += r[2]    
                avg_height += r[3]
            avg_y /= float(len(group))
            avg_width /= float(len(group))
            avg_height /= float(len(group))        
            band_rects = []
            bound = bounding_rect(group)
            for i, rect in enumerate(rects):
                if edge_counts[i] < 0.1:
                    continue
                if (abs(avg_width - rect[2]) / avg_width < similarity or \
                 (rect[2] < avg_width)) and \
                 (abs(avg_height - rect[3]) / avg_height < similarity or  \
                 (rect[3] < avg_height)) and \
                abs(avg_y - (rect[1] + rect[3]/2.0)) < avg_height/2: 
                    band_rects.append(rect)

            band_rects.sort(lambda x, y: cmp(y[2]*y[3], x[2]*x[3]))

            for i, rect_a in enumerate(band_rects[:-1]):
                if rect_a[2]*rect_a[3] < 0.2*avg_width*avg_height:
                    continue
                merge_rects = []
                for rect_b in band_rects[i+1:]:
                    w = avg_width
                    m1 = rect_a[0] + rect_a[2]/2
                    m2 = rect_b[0] + rect_b[2]/2
                    if abs(m1 - m2) < w:
                        merge_rects.append(rect_b)
                if debug:
                    print "M", merge_rects
                if merge_rects:
                    merge_rects.append(rect_a)
                    rect = bounding_rect(merge_rects)
                    area = 0
                    for r in merge_rects:
                        area += r[2]*r[3]
                    if (abs(avg_width - rect[2]) / avg_width < similarity or \
                    (rect[2] < avg_width)) and \
                    abs(avg_height - rect[3])/ avg_height < similarity and \
                    area > 0.5*(avg_width*avg_height) and \
                    abs(avg_y - (rect[1] + rect[3]/2.0)) / avg_y < similarity:
                        for r in merge_rects:
                            if r in group:
                                group.remove(r)
                        # merge into group
                        new_group = []
                        merged = False
                        for gr in group:
                            area2 = max(gr[2]*gr[3], rect[2]*rect[3])
                            isect = intersect(gr, rect)
                            if isect[2]*isect[3] > 0.4 * area2:
                                x = min(gr[0], rect[0])
                                y = min(gr[1], rect[1])
                                x2 = max(gr[0] + gr[2], rect[0] + rect[2])
                                y2 = max(gr[1] + gr[3], rect[1] + rect[3])
                                new_rect = (x, y, x2 - x, y2 - y)
                                new_group.append(new_rect)
                                merged = True  
                            else:
                                new_group.append(gr)
                        if not merged:
                            new_group.append(rect)
                        group = new_group
                        cv.Rectangle(display, (rect[0], rect[1]),
                            (rect[0] + rect[2], rect[1] + rect[3]),
                            (255,0,255), 2)
            # avoid splitting
            split = False
            # select higher threshold if innovates significantly
            best_width = 0.0
            if best_chars:
                best_area = 0.0
                for rect in best_chars:
                    best_area += rect[2] * rect[3]
                    best_width += rect[2]
                best_width /= len(best_chars)
                area = 0.0
                overlapped = 0.0                        
                avg_width = 0.0
                avg_height = 0.0                    
                for rect in group:
                    area += rect[2] * rect[3]
                    avg_width += rect[2]
                    avg_height += rect[3]
                    for char in best_chars:
                        section = intersect(rect, char)
                        if section[2] * section[3] > 0:
                            overlapped += section[2] * section[3] 
                avg_width /= len(group)
                avg_height /= len(group)
                quotient = overlapped / area
                quotient2 = (area - overlapped) / best_area
                if debug:
                    print area, overlapped, best_area
                    print group
                    print "QUO", quotient
                    print "QUO2", quotient2
            else:
                quotient = 0
                quotient2 = 1
                best_area = 0
            
            group.sort(lambda x, y: cmp(x[0] + x[2]/2, y[0] + y[2]/2))
            best_chars.sort(lambda x, y: cmp(x[0] + x[2]/2, y[0] + y[2]/2))
            if group_range[0] <= len(group) <= group_range[1] and avg_width > 5 and avg_height > 10 and \
            ((quotient2 > 0.05 and (best_area == 0 or abs(area - best_area)/best_area < 0.4))            
            or (quotient2 > 0.3 and area > best_area)):
                if debug:
                    print "ASSIGNED", group
                best_chars = group
                best_threshold = threshold #get_patch(thresholded, original_rect)
            else:
                if debug:
                    print "not", quotient2, len(group), avg_width, avg_height, area, best_area
        
        # best_chars = groups
        if debug:
            for rect in best_chars:
                cv.Rectangle(display, (rect[0], rect[1]),
                                        (rect[0] + rect[2], rect[1] + rect[3]),
                                        (0,255,0), 1)
            cv.ShowImage("main", display)
            if not skip and not next:
                c = cv.WaitKey(0)
                if c == ord("a"):
                    skip = True
                if c == ord("z"):
                    next = True
    best_chars.sort(lambda x, y: cmp(x[0], y[0]))
    return best_chars, best_threshold

def edge_magnitude(image):
    magnitude32f = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 1)
    horizontal = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_16S, 1)
    vertical = cv.CloneImage(horizontal)
    magnitude = cv.CloneImage(horizontal)
    
    mag = cv.CloneImage(image)
    cv.Sobel(image, horizontal, 0, 1, 1)
    cv.Sobel(image, vertical, 1, 0, 1)
    cv.Pow(horizontal, horizontal, 2)
    cv.Pow(vertical, vertical, 2)
    
    cv.Add(vertical, horizontal, magnitude)
    cv.Convert(magnitude, magnitude32f)
    cv.Pow(magnitude32f, magnitude32f, 0.5)
    cv.Convert(magnitude32f, mag)
    return mag
 
def segment(image, rects, debug=False, display=None):
    global next
    work_image = cv.CreateImage((image.width, image.height), 8, 1)
    cv.CvtColor(image, work_image, cv.CV_BGR2GRAY)
    image = work_image
    thresholded = cv.CloneImage(image)    
    rectified = cv.CloneImage(image)
    inverted = cv.CloneImage(image)
    candidates = []
    next = False
    for rect in rects:
        min_x, min_y, width, height = rect
        col = (randint(0,255), randint(0,255), randint(0,255))
        # only focus on horizontally sized rectangles 
        if 10./5 < float(width)/(height):
            delta_x = 10            
            delta_y = 5
            min_x = max(0, min_x - delta_x)
            min_y = max(0, min_y - delta_y)
            if min_x + width + 2*delta_x >= image.width:
                width = image.width - min_x
            else:
                width += 2*delta_x
            if min_y + height + 2*delta_y >= image.height:
                height = image.height - min_y
            else:
                height += 2*delta_y
            rect = min_x, min_y, width, height
            
            # expand around region
            cv.Copy(image, rectified)
            cv.Not(image, inverted)
            min_x, min_y, width, height = rect
            new_rect = rect
            if display:
                cv.ResetImageROI(display)
            #cv.ShowImage("main", rectified)
            #cv.WaitKey(0)            
            for im in [rectified, inverted]:
                best_chars, best_threshold = segment_rect(im, rect, debug=debug, display=display)
                bound = bounding_rect(best_chars, im, pad=5)
                best_bound = bound

                if best_chars:
                    min_x, min_y, width, height = rect
                    path = "/tmp/candidate%s.png" % len(candidates)
                    region = get_patch(im, new_rect)
    #                bound = bounding_rect(best_chars)
                    bound = best_bound
    #                bound = (bound[0] + offset[0], bound[1] + offset[1], bound[2], bound[3])
    #                region = get_patch(rectified, bound)
                    region = get_patch(region, bound)

                    cv.SaveImage(path, region)
                    # rect = (min_x, min_y, width, height)
                    candidates.append({"location":path, "chars":best_chars, 
                                       "rect": new_rect, "threshold": best_threshold,
                                       "image":region, "invert": im == inverted})

            cv.ResetImageROI(image)
            cv.ResetImageROI(rectified)
            cv.ResetImageROI(inverted)
            if display:
                cv.ResetImageROI(display)
    candidates.sort(lambda x, y: cmp(len(y["chars"]), len(x["chars"])))
#    print candidates
    return candidates

def rects_to_mask(regions, mask, inverted=False, value=255):
    if inverted:
        cv.Set(mask, cv.ScalarAll(value))
        colour = cv.ScalarAll(0)
    else:
        cv.Zero(mask)
        colour = cv.ScalarAll(value)
    for rect in regions:
        cv.Rectangle(mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), colour, -1)

def edge_threshold(image, roi=None, debug=0):
    thresholded = cv.CloneImage(image)
    horizontal = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_16S, 1)
    magnitude32f = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 1)
    vertical = cv.CloneImage(horizontal)
    v_edge = cv.CloneImage(image)
    magnitude = cv.CloneImage(horizontal)

    mag = cv.CloneImage(image)
    cv.Sobel(image, horizontal, 0, 1, 1)
    cv.Sobel(image, vertical, 1, 0, 1)
    cv.Pow(horizontal, horizontal, 2)
    cv.Pow(vertical, vertical, 2)
    
    cv.Add(vertical, horizontal, magnitude)
    cv.Convert(magnitude, magnitude32f)
    cv.Pow(magnitude32f, magnitude32f, 0.5)
    cv.Convert(magnitude32f, mag)
    if roi:
        cv.And(mag, roi, mag)
    cv.Normalize(mag, mag, 0, 255, cv.CV_MINMAX , None)
    cv.Threshold(mag, mag, 122, 255, cv.CV_THRESH_BINARY)
    draw_image = cv.CloneImage(image)
    and_image = cv.CloneImage(image)
    results = []

    threshold_start = 17
    for window_size in range(threshold_start, threshold_start+1, 1):
        r = 20
        for threshold in range(0, r):
            cv.AdaptiveThreshold(image, thresholded, 255, \
                cv.CV_ADAPTIVE_THRESH_MEAN_C, cv.CV_THRESH_BINARY_INV, window_size, threshold)
            storage = cv.CreateMemStorage(0)
            contour_image = cv.CloneImage(thresholded)
            contours = cv.FindContours(contour_image, storage, cv.CV_RETR_LIST)
            cv.Zero(draw_image)
            cv.DrawContours(draw_image, contours, (255, 255, 255), (255, 255, 255), 1, 1)
            if roi:
                cv.And(draw_image, roi, draw_image)
            cv.And(draw_image, mag, and_image)
            m1 = np.asarray(cv.GetMat(draw_image))
            m2 = np.asarray(cv.GetMat(mag))
            total = mag.width*mag.height #cv.Sum(draw_image)[0]

            coverage = cv.Sum(and_image)[0]/(mag.width*mag.height)
            if debug:
                print threshold, coverage
                cv.ShowImage("main", draw_image)
                cv.ShowImage("main2", thresholded)
                cv.WaitKey(0) 
            results.append((coverage, threshold, window_size))

    results.sort(lambda x, y: cmp(y, x))
    _, threshold, window_size = results[0]
    cv.AdaptiveThreshold(image, thresholded, 255, cv.CV_ADAPTIVE_THRESH_MEAN_C, \
        cv.CV_THRESH_BINARY, window_size, threshold)

    return thresholded

def recognize(image, display=None, visualize=None, clean=False):
    rects = detect(image, debug=0, display=display)
    candidates = segment(image, rects, debug=0, display=display)
    font = cv.InitFont(cv.CV_FONT_VECTOR0, 0.4, 0.4, 0.0, 0, 0)

    source = cv.CreateImage((image.width, image.height), 8, 1)
    source2 = cv.CreateImage((image.width, image.height), 8, 1)
    mask = cv.CloneImage(source)
    cv.CvtColor(image, source, cv.CV_BGR2GRAY)
    for i, c in enumerate(candidates):
        window_name = "candidate%s" % i
        candidate = cv.CloneImage(c["image"])


        rect = c["rect"]
        invert  = c["invert"]
        if visualize:
            # cv.SetImageROI(visualize, rect)
            cv.SetImageROI(source, rect)
            cv.SetImageROI(source2, rect)
            cv.SetImageROI(image, rect)
            cv.SetImageROI(mask, rect)
            bound_rect = bounding_rect(c["chars"])
            rects_to_mask([bound_rect], mask, value=255)
            cv.Zero(source2)
            edge = edge_threshold(source)
            cv.Copy(edge, source2, mask)

            text1, conf1 = tesseract(source)
            text2, conf2 = tesseract(source2)
            print text1, text2
            print conf1, conf2
            cv.ShowImage("source", source)
            cv.ShowImage("thresholded", source2)
            cv.ShowImage("edge", edge)
            cv.ShowImage("mask", mask)
            cv.WaitKey(5)
            gray = (150, 150, 150)
            col1, col2, col3 = gray, gray, gray 
            k = 70
            if np.mean(conf1) >= np.mean(conf2):
                if any(conf1 > k):
                    if any(conf2 > k) and len(text2) > len(text1) * 2:
                        col2 = (0, 255, 0)
                    else:
                        col1 = (0, 255, 0)
                    col3 = (0, 255, 0)
            else:
                if any(conf2 > k):
                    if any(conf1 > k) and len(text1) > len(text2) * 2:
                        col1 = (0, 255, 0)
                    else:
                        col2 = (0, 255, 0)
                    col3 = (0, 255, 0)

            if clean:
                if col1 != gray and text1 is not None:
                    cv.PutText(visualize, text1, (rect[0], rect[1]-5), font, col1)
                if col2 != gray and text2 is not None:
                    cv.PutText(visualize, text2, (rect[0], rect[1]-5), font, col2)
            else:        
                if text1 is not None:
                    cv.PutText(visualize, text1, (rect[0], rect[1]-5), font, col1)
                if text2 is not None:
                    cv.PutText(visualize, text2, (rect[0], rect[1]-15), font, col2)
            cv.Rectangle(visualize, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), col3, 1)     
            cv.ResetImageROI(source)
            cv.ResetImageROI(source2)
            cv.ResetImageROI(image)
            cv.ResetImageROI(mask)

def tesseract(image):
    import tesseract_sip as tesseract
    tessdata_prefix = os.environ.get('TESSDATA_PREFIX')
    # cv.SaveImage('test.png', image)
    # image = cv.LoadImage('test.png')
    if not tessdata_prefix:
        tessdata_prefix = '/usr/share/tesseract-ocr/tessdata/'

    if not os.path.exists(tessdata_prefix):
        # if you get this error, you need to download tesseract-ocr-3.02.eng.tar.gz 
        # and unpack it in this directory. 
        print >> sys.stderr, 'WARNING: tesseract OCR data directory was not found'
    api = tesseract.TessBaseAPI()
    if not api.Init(tessdata_prefix, 'eng', tesseract.OEM_DEFAULT):
        print >> sys.stderr, "Error initializing tesseract"
        exit(1)
    api.SetPageSegMode(tesseract.PSM_SINGLE_LINE)
    # api.SetPageSegMode(tesseract.PSM_AUTO)
    # cvimg = cv2.imread('test.png')
    # api.SetImage(cvimg)
    source = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U, 3)
    cv.Merge(image, image, image, None, source)
    api.SetImage(cv2num(source))
    text = api.GetUTF8Text()
    text = text.encode('ascii','ignore').strip()
    return text, np.array(api.AllWordConfidences())

if __name__ == "__main__":
    location = "data"
    cv.NamedWindow("original")
    cv.NamedWindow("main")
    cv.MoveWindow("original", 100, 100)
    cv.MoveWindow("main", 500, 100)
    font = cv.InitFont(cv.CV_FONT_VECTOR0, 0.8, 0.8, 0.0, 0, 0)
    small_font = cv.InitFont(cv.CV_FONT_VECTOR0, 0.6, 0.6, 0.0, 0, 0)

    display = None
    if len(sys.argv) < 2:
        print "Usage segment.py <input video file>"
        sys.exit(1)
    capture = cv.CreateFileCapture(sys.argv[1])

    pretty_print = 1 
    record = False
    if record:
        writer = OpencvRecorder('save.avi', framerate=10)

    i = 0
    # XXX skip first few frames of video
    for i in range(2300):
        image = cv.QueryFrame(capture)

    while 1:
        i += 1
        print i
        image = cv.QueryFrame(capture)
        if not image:
            print 'Video not found'
            break
        if not display:
            display = cv.CloneImage(image)
            original = cv.CloneImage(image)
        else:
            cv.Copy(image, display)
            cv.Copy(image, original)
        cv.ShowImage("original", image)
        char = cv.WaitKey(10)
        if char != -1:
            for i in range(1000):
                print i
                image = cv.QueryFrame(capture)
            cv.ShowImage("original", image)
            cv.WaitKey(10)


        result = recognize(image, display=display, visualize=original, 
                           clean=pretty_print)
        cv.ShowImage("main", original)
        if record:
            writer.write(original)
        cv.WaitKey(10)

