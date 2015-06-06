#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"

#include "Python.h"
extern "C" {



extern int test_numpy(double* data, int len) {
        int i;
        printf("data = %p\n", (void*) data);
        for (i = 0; i < len; i++) {
                printf("data[%d] = %f\n", i, data[i]);
        }
        printf("len = %d\n", len);
        return len + 1;
}


extern int test_opencv(IplImage* image) {
        cvNot(image, image);
}

// extern int enable_ipp() {
//     const char* version;
//     const char* loaded_plugs;
//     cvUseOptimized(1);
//     cvGetModuleInfo( NULL, &version, &loaded_plugs);                       
//     printf("%s\n", version);
//     printf("%s\n", loaded_plugs);
// }

const int LESSTHAN = 0;
const int GREATERTHAN = 1;


extern void test_contours(CvSeq* contours) {
    CvSeq *prev_seq = NULL, *seq = NULL;
    int i = 0;
    printf("X %d\n", contours->total);
}


extern int contour_count(CvSeq* contours) {
    int count= 0;
    CvSeq *seq = NULL;   
    if (contours) {
        for( seq = contours; seq; seq = seq->h_next )        
        {
            count++;
        }
    }
    return count;
}
        

extern void cvClearMemStorage(CvMemStorage* storage) {
    cvClearMemStorage(storage);
}

// filter contours according to size
extern CvSeq* filter_contours(CvSeq* contours, int min_size, int comparator) {
    CvSeq *prev_seq = NULL, *seq = NULL; 
    int i = 0;
    //printf("X %d %d\n", contours->total, contours->v_next);
    for(seq = contours; seq; seq = seq->h_next)
    {
        i++;
        CvContour* cnt = (CvContour*)seq;
        //printf("%d %d %d\n", i, cnt->rect.height, cnt->rect.width);
        if (((cnt->rect.width * cnt->rect.height < min_size ) && (comparator == LESSTHAN)) ||
        ((cnt->rect.width * cnt->rect.height > min_size ) && (comparator == GREATERTHAN)))            
        {
            // delete a contour
            prev_seq = seq->h_prev;
            if( prev_seq )
            {
                prev_seq->h_next = seq->h_next;
                if( seq->h_next ) seq->h_next->h_prev = prev_seq;
            }
            else
            {
                contours = seq->h_next;
                if( seq->h_next ) seq->h_next->h_prev = NULL;
            }
            //~ total -= 1;
        }
    }
    return contours;
    //~ if (contours)
        //~ contours->total = total;
}



}

