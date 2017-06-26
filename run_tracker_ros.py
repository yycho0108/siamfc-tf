from __future__ import division
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import cv2
import rospy

from timer import Timer

import src.siamese as siam
from src.parse_arguments import parse_arguments

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y

def lerp(a,b,w):
    return w*a + (1.0-w)*b

class SiamParams(object):
    def __init__(self,
            response_up = 8,
            window_influence = 0.25,
            z_lr = 0.01,
            scale_num = 3,
            scale_step = 1.04,
            scale_penalty = 0.97,
            scale_lr = 0.59,
            scale_min = 0.2,
            scale_max = 5,
            thresh_fail = 0.24
            ):
        self.response_up = response_up
        self.window_influence = window_influence
        self.z_lr = z_lr
        self.scale_num = scale_num
        self.scale_step = scale_step
        self.scale_penalty = scale_penalty
        self.scale_lr = scale_lr
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.thresh_fail = thresh_fail

class SiamFCTracker(object):
    def __init__(self, env, design):

        self.hp = SiamParams(
                window_influence=0.5,
                z_lr=0.01,
                scale_num=16,
                scale_step=1.002,
                scale_penalty=0.97,
                scale_lr=0.2,
                scale_min=0.2,
                scale_max=5.0
                )
        hp = self.hp
        self.design = design

        self.final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        self.image, self.templates_z, self.scores = siam.build_tracking_graph_2(self.final_score_sz, design, env)

        self.scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
        # cosine window to penalize large displacements    
        self.hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        self.penalty = np.transpose(self.hann_1d) * self.hann_1d
        self.penalty /= np.max(self.penalty)

    def initialize(self, sess, image_, bbox_):
        hp = self.hp
        design = self.design

        image_ = image_[...,::-1].astype(np.float32) #anticipate bgr-uint8
        cx,cy,w,h = bbox_

        self.pos_x = cx
        self.pos_y = cy
        self.target_w = w
        self.target_h = h

        context = design.context*(self.target_w+self.target_h)
        self.z_sz = np.sqrt(np.prod((self.target_w+context)*(self.target_h+context)))
        self.x_sz = float(design.search_sz) / design.exemplar_sz * self.z_sz
        
        self.templates_z_ = sess.run(self.templates_z, feed_dict={
            siam.pos_x_ph: self.pos_x,
            siam.pos_y_ph: self.pos_y,
            siam.z_sz_ph: self.z_sz,
            self.image : image_})

    def update(self, sess, image_):
        hp = self.hp
        design = self.design

        image_ = image_[...,::-1].astype(np.float32)

        scaled_exemplar    = self.z_sz * self.scale_factors
        scaled_search_area = self.x_sz * self.scale_factors
        scaled_target_w    = self.target_w * self.scale_factors
        scaled_target_h    = self.target_h * self.scale_factors
        
        scores_ = sess.run(
                self.scores,
                feed_dict={
                    siam.pos_x_ph: self.pos_x,
                    siam.pos_y_ph: self.pos_y,
                    siam.x_sz0_ph: scaled_search_area[0],
                    siam.x_sz1_ph: scaled_search_area[1],
                    siam.x_sz2_ph: scaled_search_area[2],
                    self.templates_z: np.squeeze(self.templates_z_),
                    self.image: image_,
                })
        scores_ = np.squeeze(scores_)
        # penalize change of scale
        scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
        scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
        # find scale with highest peak (after penalty)
        new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))

        # update scaled sizes
        self.x_sz = lerp(scaled_search_area[new_scale_id], self.x_sz, hp.scale_lr)
        self.target_w = lerp(scaled_target_w[new_scale_id], self.target_w, hp.scale_lr)
        self.target_h = lerp(scaled_target_h[new_scale_id], self.target_h, hp.scale_lr)

        # select response with new_scale_id
        score_ = scores_[new_scale_id,:,:]
        score_ = score_ - np.min(score_)
        score_ = score_/np.sum(score_)

        # apply displacement penalty
        disp_penalty = ((1.0 - hp.window_influence)+self.penalty)
        disp_penalty /= np.max(disp_penalty)
        score_ = score_ * disp_penalty

        ###
        s = np.max(score_) * 10000
        if s < hp.thresh_fail:
            return None
        ###

        self.pos_x, self.pos_y = _update_target_position(self.pos_x, self.pos_y, score_, self.final_score_sz, design.tot_stride, design.search_sz, hp.response_up, self.x_sz)
        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        bbox = self.pos_x-self.target_w/2, self.pos_y-self.target_h/2, self.target_w, self.target_h

        # update the target representation with a rolling average
        if hp.z_lr>0:
            new_templates_z_ = sess.run(self.templates_z, feed_dict={
                                                            siam.pos_x_ph: self.pos_x,
                                                            siam.pos_y_ph: self.pos_y,
                                                            siam.z_sz_ph: self.z_sz,
                                                            self.image: image_
                                                            })
            self.templates_z_ = lerp(new_templates_z_, self.templates_z_, hp.z_lr)#(1-hp.z_lr)*np.asarray(self.templates_z_) + hp.z_lr*np.asarray(new_templates_z_)
        
        # update template patch size
        self.z_sz = lerp(scaled_exemplar[new_scale_id], self.z_sz, hp.scale_lr)

        return bbox

def mouse_cb(event, x, y, f, p):
    global drag, pos_x, pos_y, target_w, target_h, initialized
    if event == cv2.EVENT_LBUTTONDOWN:
        drag = True
        pos_x = x
        pos_y = y
    if event == cv2.EVENT_LBUTTONUP:
        if drag:
            drag = False
            if x < pos_x:
                x, pos_x = pos_x, x
            if y < pos_y:
                y, pos_y = pos_y, y
            target_w = x - pos_x
            target_h = y - pos_y
            pos_x += target_w/2
            pos_y += target_h/2
            initialized = True

def init_bbox(image_):
    global drag, pos_x, pos_y, target_w, target_h, initialized
    
    initialized = False
    drag = False
    cv2.destroyAllWindows()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_cb)

    while not initialized:
        cv2.imshow('image', image_)
        cv2.waitKey(10)
        continue
    bbox_ = (pos_x, pos_y, target_w, target_h) #cx,cy,w,h
    return bbox_

def main():

    _, _, _, env, design = parse_arguments()

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

    cam = cv2.VideoCapture('/home/yoonyoungcho/ext/frame%04d.jpg')
    for i in range(10):
        ret, image_ = cam.read()

    bbox_ = init_bbox(image_)

    tracker = SiamFCTracker(env, design)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tracker.initialize(sess, image_, bbox_)
        while True:
            ret, image_ = cam.read()
            if not ret:
                break
            with Timer('siam'):
                bbox_ = tracker.update(sess, image_) 

            if bbox_ is None:
                bbox_ = init_bbox(image_)
                tracker.initialize(sess, image_, bbox_)

            x, y, w, h = map(int, bbox_)
            cv2.rectangle(image_, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.imshow('track', image_)
            if cv2.waitKey(10) == 27:
                break

if __name__ == '__main__':
    sys.exit(main())
