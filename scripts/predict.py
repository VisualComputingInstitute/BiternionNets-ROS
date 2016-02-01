#!/usr/bin/env python
# encoding: utf-8

from os.path import abspath, expanduser, join as pjoin
from sys import stderr

import cv2
import rospy
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image as ROSImage
from biternion.msg import HeadOrientations

import numpy as np
import DeepFried2 as df
from common import mknet, bit2deg

# Distinguish between STRANDS and SPENCER.
try:
    from upper_body_detector.msg import UpperBodyDetector
except ImportError:
    from rwth_perception_people_msgs.msg import UpperBodyDetector
    from spencer_tracking_msgs.msg import TrackedPersons2d


def cutout(img, detrect):
    x, y, w, h = detrect

    # Need to be careful for negative indices in conjunction with
    # numpy's (and thus OpenCV's) wrap-around.
    y2, x2 = y+h, x+w
    y1, x1 = max(y, 0), max(x, 0)
    return img[y1:y2, x1:x2]


def get_rects(msg):
    if isinstance(msg, TrackedPersons2d):
        return [(p2d.x, p2d.y, p2d.w, p2d.h) for p2d in msg.boxes]
    elif isinstance(msg, UpperBodyDetector):
        return list(zip(ubd.pos_x, ubd.pos_y, ubd.width, ubd.height))
    else:
        raise TypeError("Unknown source type: {}".format(type(msg)))


class Predictor(object):
    def __init__(self):
        rospy.loginfo("Initializing biternion predictor")
        self.counter = 0

        self.hfact = float(rospy.get_param("~hfactor", "-1.0"))
        self.wfact = float(rospy.get_param("~wfactor", "1.0"))
        if self.wfact <= 0:
            self.wfact = 1

        modelname = abspath(expanduser(rospy.get_param("~model", ".")))

        rospy.loginfo("Predicting {}-bodies using {}".format(self.hfact if self.hfact > 0 else "full", modelname))

        topic = rospy.get_param("~topic", "/tmpluc")
        self.pub = rospy.Publisher(topic, HeadOrientations, queue_size=3)
        self.pub_vis = rospy.Publisher(topic + '/image', ROSImage, queue_size=3)

        # Create and load the network.
        self.net = mknet()
        self.net.__setstate__(np.load(modelname)['arr_0'])
        self.net.evaluate()

        # Do a fake forward-pass for precompilation.
        self.net.forward(np.zeros((1,3,46,46), df.floatX))

        src = rospy.get_param("~src", "tra")
        subs = []
        if src == "tra":
            subs.append(message_filters.Subscriber(rospy.get_param("~tra", "/TODO"), TrackedPersons2d))
        elif src == "ubd":
            subs.append(message_filters.Subscriber(rospy.get_param("~ubd", "/upper_body_detector/detections"), UpperBodyDetector))
        else:
            raise ValueError("Unknown source type: " + src)

        subs.append(message_filters.Subscriber(rospy.get_param("~rgb", "/head_xtion/rgb/image_rect_color"), ROSImage))
        #message_filters.Subscriber(rospy.get_param("~d", "/head_xtion/depth/image_rect_meters"), ROSImage),

        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.1)
        ts.registerCallback(self.cb)

    def cb(self, src, rgb):  #, d):
        header = rgb.header
        b = CvBridge()
        rgb = b.imgmsg_to_cv2(rgb)[:,:,::-1]  # Need to do BGR-RGB conversion manually.
        #d = b.imgmsg_to_cv2(d)

        imgs = []

        for detrect in get_rects(src):
            detrect = self.factrect(detrect)
            det_rgb = cutout(rgb, detrect)
            #det_d = cutout(d, detrect)

            # Resize and stick into the minibatch.
            im = cv2.resize(det_rgb, (50, 50))
            im = np.rollaxis(im, 2, 0)
            im = im[:,2:-2,2:-2]  # TODO: Augmentation?
            print(im.shape)
            imgs.append(im.astype(df.floatX)/255)

            stderr.write("\r{}".format(self.counter)) ; stderr.flush()
            self.counter += 1

        if 0 < len(imgs):
            preds = bit2deg(self.net.forward(np.array(imgs)))
            print(preds)

            self.pub.publish(HeadOrientations(
                header=header,
                angles=list(preds),
                confidences=[0.83] * len(imgs)
            ))

            # Visualization
            if 0 < self.pub_vis.get_num_connections():
                rgb_vis = rgb[:,:,::-1].copy()
                for detrect, alpha in zip(get_rects(src), preds):
                    l, t, w, h = self.factrect(detrect)
                    px =  int(round(np.cos(np.deg2rad(alpha-90))*w/2))
                    py = -int(round(np.sin(np.deg2rad(alpha-90))*h/2))
                    cv2.rectangle(rgb_vis, (detrect[0], detrect[1]), (detrect[0]+detrect[2],detrect[1]+detrect[3]), (0,255,255), 1)
                    cv2.rectangle(rgb_vis, (l,t), (l+w,t+h), (0,255,0), 2)
                    cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px,t+h//2+py), (0,255,0), 2)
                    cv2.putText(rgb_vis, "{:.1f}".format(alpha), (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                vismsg = b.cv2_to_imgmsg(rgb_vis, encoding='rgb8')
                vismsg.header = header  # TODO: Seems not to work!
                self.pub_vis.publish(vismsg)

    def factrect(self, rect):
        x, y, w, h = rect
        # NOTE: Order is important here.
        h = int(round(min(self.hfact*w, h) if self.hfact > 0 else h))
        x = x + int(round((1 - self.wfact)/2*w))
        w = int(round(self.wfact*w))

        return x, y, w, h


if __name__ == "__main__":
    rospy.init_node("biternion_predict")
    p = Predictor()
    rospy.spin()
    rospy.loginfo("Predicted a total of {} UBDs.".format(p.counter))
