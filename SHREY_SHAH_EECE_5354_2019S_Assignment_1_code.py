
# Name : Shrey Shah
# Course: EECE 5354 Computer Vision
# Assignment No. : 01
# Date : 01/27/2019

#!/usr/bin/env python

'''
Example code for live video processing
Also multithreaded video processing sample using opencv 3.4

Usage:
   python testcv_mt.py {<video device number>|<video file name>}

   Use this code as a template for live video processing

   Also shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.

Keyboard shortcuts: (video display window must be selected

   ESC - exit
   space - switch between multi and single threaded processing
   a - adjust contrast, brightness, and gamma
   d - running difference of current and previous image
   e - displays canny edges
   f - displays raw frames
   h - display image hue band
   o - apply a 5x5 "fat plus" opening to the thresholded image
   q - histogram equalize value image
   t - do thresholding
   g - 2D seperable filtering with gaussian kernel
   c - Apply various colormaps from the slider
   m - To apply Morphological Erosion on the Image
   s - To Downsample the image
   v - write video output frames to file "vid_out.avi"

Click and drag the left mouse button to select the region of interest
Options in ROI:
    e - display canny edges
    f - display raw frames
    c - Apply various colormaps form the slider
    g - 2D seperable filtering with gaussian kernel
    ESC - exit ROI


'''

# import the necessary packages
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
from multiprocessing.pool import ThreadPool
from collections import deque
from common import clock, draw_str, StatValue
import video
import argparse
# import math
# from matplotlib import pyplot as plt


# used to execute process_frame when in non threaded mode
class DummyTask:
    def __init__(self, data):
        self.data = data

    def ready(self):
        return True

    def get(self):
        return self.data


# initialize global variables
frame_counter = 0
show_frames = True
diff_frames = False
show_edges = False
show_hue = False
do_threshold = False
adj_img = False
adj_gam = False
m_open = False
hist_eq = False
vid_frames = False
gauss_filter = False
cmap = False
morph = False
Down_sample = False
struct_el = 0
struct_slider_max = 2
contrast = 128
contrast_slider_max = 255
brightness = 128
brightness_slider_max = 255
gamma = 128
gamma_slider_max = 255
threshold = 128
threshold_slider_max = 255
sigma = 1
sigma_slider_max = 64
colormap = 0
cmap_slider_max = 12
draw = True
point = []
drawing = False


# this routine is run each time a new video frame is captured
def process_frame(frame, prevFrame, t0):

    if adj_img:
        global contrast, brightness, gamma
        # shift value to get actual brightness offset
        brite = brightness - 128
        # compute the contrast value from the trackbar setting
        if contrast > 127:
            contr = 1+(5*(contrast - 128)/128)
        else:
            contr = 1/(1+(5*(128 - contrast)/128))
        # adjust brightness and contrast
        frame = ((np.float_(frame)-128) * contr) + 128 + brite
        # compute the gamma value from the trackbar setting
        if gamma > 127:
            gam = 1+(2*(gamma - 128)/128)
        else:
            gam = 1/(1+(2*(128 - gamma)/128))
        # apply the gamma function
        frame = 255 * ((frame / 255) ** (1 / gam))
        # then convert the result back to uint8 after clipping at 0 and 255
        frame = np.uint8(np.clip(frame, 0, 255))

    if hist_eq:
        # convert image to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]  # separate the channels
        val = cv.equalizeHist(val)
        hsv = cv.merge((hue, sat, val))
        frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    if not show_frames and show_edges:  # edges alone
        edges = cv.Canny(frame, 100, 200)
        thisFrame = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
    elif show_frames and show_edges:  # edges and frames
        edges = cv.Canny(frame, 100, 200)
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
        thisFrame = cv.add(frame, edges)
    else:  # current frame
        thisFrame = frame.copy()

    if do_threshold:
        # create threshold mask
        threshMask = get_threshold_mask(frame)
        # apply the mask
        thisFrame = threshMask * thisFrame

    if show_hue:
        # convert image to HSV
        hsv = cv.cvtColor(thisFrame, cv.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]  # separate the channels
        # the maximum hue is 170, so scale it into [0,255]
        h32 = np.float32(hue) * 255 / 170
        sch = np.uint8(np.clip(h32, 0, 255))  # clip at 255 and convert back to uint8
        # apply the opencv builtin hue colormap
        thisFrame = cv.applyColorMap(sch, cv.COLORMAP_HSV)

    if diff_frames:
        # compute absolute difference between the current and previous frame
        difframe = cv.absdiff(thisFrame, prevFrame)
        # save current frame as previous
        prevFrame = thisFrame.copy()
        # set the current frame to the difference image
        thisFrame = difframe.copy()
    else:
        # save current frame as previous
        prevFrame = thisFrame.copy()

    if gauss_filter and show_frames and not show_edges:
        global sigma
        # Generate the Gaussian Kernel
        kernel = cv.getGaussianKernel(11,sigma)
        # Apply the sepFilter2D with Gaussian Kernel on the frame
        thisFrame = cv.sepFilter2D(frame, -1, kernel, kernel)

    if cmap and show_frames :
        global colormap
        # Apply colormap on the frame
        thisFrame = cv.applyColorMap(frame,colormap)

    if morph and show_frames:
        global struct_el
        # Get the structuring element for the morphological function
        z = cv.getStructuringElement(struct_el, (7, 7))
        # perfor erosion operation on the frame
        thisFrame = cv.erode(frame, z)

    if Down_sample and show_frames:
        # Perform downsampling on the frame
        thisFrame = cv.pyrDown(frame)
    return thisFrame, prevFrame, t0


def get_threshold_mask(frame):
    global threshold
    # mB, mG, mR, _ = np.uint8(cv.mean(frame))
    B, G, R = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    # _, tB = cv.threshold(B, mB, 1, cv.THRESH_BINARY)
    # _, tG = cv.threshold(G, mG, 1, cv.THRESH_BINARY)
    # _, tR = cv.threshold(R, mR, 1, cv.THRESH_BINARY)
    _, tB = cv.threshold(B, threshold, 1, cv.THRESH_BINARY)
    _, tG = cv.threshold(G, threshold, 1, cv.THRESH_BINARY)
    _, tR = cv.threshold(R, threshold, 1, cv.THRESH_BINARY)
    if m_open:
        # create structuring element for morph ops
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        tB = cv.morphologyEx(tB, cv.MORPH_OPEN, se, 1)
        tG = cv.morphologyEx(tG, cv.MORPH_OPEN, se, 1)
        tR = cv.morphologyEx(tR, cv.MORPH_OPEN, se, 1)
    threshMask = cv.merge((tB, tG, tR))

    return threshMask


def on_brightness_trackbar(val):
    global brightness
    brightness = val


def on_contrast_trackbar(val):
    global contrast
    contrast = val


def on_gamma_trackbar(val):
    global gamma
    gamma = val


def on_threshold_trackbar(val):
    global threshold
    threshold = val

def on_sigma_trackbar(val):
    global sigma
    sigma = val

def on_cmap_trackbar(val):
    global colormap
    colormap = val

def roi_process(img_frame):
    # For performinh operations on the region of interest
    if not show_frames and show_edges:  # edges alone
        edges = cv.Canny(img_frame, 100, 200)
        thisFrame = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
    elif show_frames and show_edges:  # edges and frames
        edges = cv.Canny(img_frame, 100, 200)
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
        thisFrame = cv.add(img_frame, edges)
    else:  # current frame
        thisFrame = img_frame.copy()
    if cmap and show_frames :
        global colormap
        # To apply colormap on the ROI
        thisFrame = cv.applyColorMap(img_frame,colormap)
    if gauss_filter and show_frames and not show_edges:
        global sigma
        # To perform Gaussian Filtering on the ROI
        kernel = cv.getGaussianKernel(11,sigma)
        thisFrame = cv.sepFilter2D(img_frame, -1, kernel, kernel)
    return thisFrame

def draw_ROI(event, x, y, flags, params):
    # To create a region of interest
    global point,drawing
    if event == cv.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            point = [(x, y)]
        else:
            drawing = False
    elif event == cv.EVENT_LBUTTONUP:
        if drawing is True:
            point.append((x, y))
            drawing = False


def on_struct_trackbar(val):
    global struct_el
    struct_el = val

# create a video capture object
def create_capture(source=0):

    # parse source name (defaults to 0 which is the first USB camera attached)

    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isthreshold():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try:
        source = int(source)
    except ValueError:
        pass

    params = dict(s.split('=') for s in chunks[1:])

    # video capture object defined on source

    timeout = 100
    iter = 0
    cap = cv.VideoCapture(source)
    while (cap is None or not cap.isOpened()) & (iter < timeout):
        time.sleep(0.1)
        iter = iter + 1
        cap = cv.VideoCapture(source)

    if iter == timeout:
        print('camera timed out')
        return None
    else:
        print(iter)

    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        return None

    return cap

# main program
if __name__ == '__main__':
    import sys

    # print in the program shell window the text at the beginning of the file
    print(__doc__)

    # if there is no argument in the program invocation default to camera 0
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    # grab initial frame, create window
    cv.waitKey(1) & 0xFF
    cap = video.create_capture(fn)
    ret, frame = cap.read()
    frame_counter += 1
    height, width, channels = frame.shape
    prevFrame = frame.copy()
    cv.namedWindow("video")
    #cv.setMouseCallback('video', draw_ROI)
    # Create video of Frame sequence -- define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    cols = np.int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    rows = np.int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    vid_out = cv.VideoWriter('vid_out.avi', fourcc, 20.0, (cols, rows))

    # Set up multiprocessing
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    threaded_mode = True
    onOff = False

    # initialize time variables
    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()

    # main program loop
    while True:
        while len(pending) > 0 and pending[0].ready():  # there are frames in the queue
            res, prevFrame, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            # plot info on threading and timing on the current image
            # comment out the next 3 lines to skip the plotting
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))
            # write output video frame
            if vid_frames:
                vid_out.write(res)
            # show the current image
            cv.imshow('video', res)

        if len(pending) < threadn:  # fewer frames than threads ==> get another frame
            # get frame
            ret, frame = cap.read()
            frame_counter += 1
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), prevFrame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, prevFrame, t))
            pending.append(task)

        # check for a keypress
        key = cv.waitKey(1) & 0xFF
        cv.setMouseCallback('video', draw_ROI)
        # threaded or non threaded mode
        if key == ord(' '):
            threaded_mode = not threaded_mode
        # toggle point processes -- adjust image
        if key == ord('a'):
            adj_img = not adj_img
            if adj_img:
                cv.createTrackbar("brightness", 'video', brightness, brightness_slider_max, on_brightness_trackbar)
                cv.createTrackbar("contrast", 'video', contrast, contrast_slider_max, on_contrast_trackbar)
                cv.createTrackbar("gamma", 'video', gamma, gamma_slider_max, on_gamma_trackbar)
            else:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', res)
        # toggle edges
        if key == ord('e'):
            show_edges = not show_edges
            if not show_edges and not show_frames:
                show_frames = True
        # toggle frames
        if key == ord('f'):
            show_frames = not show_frames
            if not show_frames and not show_edges:
                show_frames = True
        # image difference mode
        if key == ord('d'):
            diff_frames = not diff_frames
        # display image hue band
        if key == ord('h'):
            show_hue = not show_hue
        # equalize image value band
        if key == ord('q'):
            hist_eq = not hist_eq
        # threshold the image
        if key == ord('t'):
            do_threshold = not do_threshold
            if do_threshold:
                cv.createTrackbar("threshold", 'video', threshold, threshold_slider_max, on_threshold_trackbar)
            else:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', res)
        # do morphological opening on thresholded image (only applied to thresholded image)
        if key == ord('o'):
            m_open = not m_open
        # write video frames
        if key == ord('v'):
            vid_frames = not vid_frames
            if vid_frames:
                print("Frames are being output to video")
            else:
                print("Frames are not being output to video")

        # Do seperable filtering of the image with gaussian kernel
        if key == ord('g') :
            gauss_filter = not gauss_filter
            if gauss_filter:
                cv.createTrackbar("Gaussian Filtering", 'video', sigma, sigma_slider_max, on_sigma_trackbar)

            else:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', res)
        # Apply colormap to the frame
        if key == ord('c') :
            cmap = not cmap
            if cmap:
                cv.createTrackbar('Colormap', 'video', colormap, cmap_slider_max, on_cmap_trackbar)

            else :
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video',res)
        # To downsample the frame
        if key ==ord('s'):
            Down_sample = not Down_sample
        # To draw a rectangular ROI and performing operations in it
        draw = True
        if len(point) == 2:
            cv.destroyWindow('video')
            while draw:
                _, frame = cap.read()
                cv.rectangle(frame, point[0], point[1], (0, 255, 0), 1)
                ROI = frame[point[0][1]:point[1][1], point[0][0]:point[1][0]]
                k = cv.waitKey(1) & 0xFF
                if k == ord('e'):
                    show_edges = not show_edges
                    if not show_edges and not show_frames:
                        show_frames = True

                    # toggle frames
                if k == ord('f'):
                    show_frames = not show_frames
                    if not show_frames and not show_edges:
                        show_frames = True
                if k == ord('c'):
                    cmap = not cmap
                    if cmap:
                        cv.createTrackbar('Colormap', 'video', colormap, cmap_slider_max, on_cmap_trackbar)

                    else:
                        cv.destroyWindow('video')
                        cv.namedWindow('video')
                        cv.imshow('video', res)
                if k == ord('g'):
                    gauss_filter = not gauss_filter
                    if gauss_filter:
                        cv.createTrackbar("Gaussian Filtering", 'video', sigma, sigma_slider_max, on_sigma_trackbar)

                    else:
                        cv.destroyWindow('video')
                        cv.namedWindow('video')
                        cv.imshow('video', res)
                if k == ord('v'):
                    vid_frames = not vid_frames
                    if vid_frames:
                        print("Frames are being output to video")
                    else:
                        print("Frames are not being output to video")

                if k==27:
                    draw = False
                    point = []
                    break
                ROI = roi_process(ROI)
                frame[point[0][1]:point[1][1], point[0][0]:point[1][0]] = ROI
                if vid_frames:
                    vid_out.write(frame)
                cv.imshow('video', frame)


        if key == ord('m'):
            # To perform morphological erosion on the frame
            morph = not morph
            if morph:
                cv.createTrackbar('Structuring Element', 'video', struct_el, struct_slider_max, on_struct_trackbar)
            else :
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', res)

        # ESC terminates the program
        if key == 27:
            break

# release video capture object
cap.release()
# release video output object
vid_out.release()
cv.destroyAllWindows()
