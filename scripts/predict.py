#!/usr/bin/env python
# encoding: utf-8

from os.path import abspath, expanduser, join as pjoin
import sys
from importlib import import_module

import numpy as np
import cv2

import rospy
from tf.transformations import quaternion_about_axis
from rospkg import RosPack
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from biternion.msg import HeadOrientations
from visualization_msgs.msg import Marker

import DeepFried2 as df

from common import bit2deg, ensemble_biternions, subtractbg, cutout

# Distinguish between STRANDS and SPENCER.
try:
    from upper_body_detector.msg import UpperBodyDetector
except ImportError:
    from rwth_perception_people_msgs.msg import UpperBodyDetector
    from spencer_tracking_msgs.msg import TrackedPersons2d


def get_rects(msg, with_depth=False):
    if isinstance(msg, TrackedPersons2d):
        return [(p2d.x, p2d.y, p2d.w, p2d.h) + ((p2d.depth,) if with_depth else tuple()) for p2d in msg.boxes]
    elif isinstance(msg, UpperBodyDetector):
        return list(zip(*([msg.pos_x, msg.pos_y, msg.width, msg.height] + ([msg.median_depth] if with_depth else []))))
    else:
        raise TypeError("Unknown source type: {}".format(type(msg)))


class Predictor(object):
    def __init__(self):
        rospy.loginfo("Initializing biternion predictor")
        self.counter = 0

        modelname = rospy.get_param("~model", "head_50_50")
        weightsname = abspath(expanduser(rospy.get_param("~weights", ".")))
        rospy.loginfo("Predicting using {} & {}".format(modelname, weightsname))

        topic = rospy.get_param("~topic", "/biternion")
        self.pub = rospy.Publisher(topic, HeadOrientations, queue_size=3)
        self.pub_vis = rospy.Publisher(topic + '/image', ROSImage, queue_size=3)
        self.pub_pa = rospy.Publisher(topic + "/pose", PoseArray, queue_size=3)

        # Create and load the network.
        netlib = import_module(modelname)
        self.net = netlib.mknet()
        self.net.__setstate__(np.load(weightsname))
        self.net.evaluate()

        self.aug = netlib.mkaug(None, None)
        self.preproc = netlib.preproc
        self.getrect = netlib.getrect

        # Do a fake forward-pass for precompilation.
        im = cutout(np.zeros((480,640,3), np.uint8), 0, 0, 150, 450)
        im = next(self.aug.augimg_pred(self.preproc(im), fast=True))
        self.net.forward(np.array([im]))

        src = rospy.get_param("~src", "tra")
        subs = []
        if src == "tra":
            subs.append(message_filters.Subscriber(rospy.get_param("~tra", "/TODO"), TrackedPersons2d))
        elif src == "ubd":
            subs.append(message_filters.Subscriber(rospy.get_param("~ubd", "/upper_body_detector/detections"), UpperBodyDetector))
        else:
            raise ValueError("Unknown source type: " + src)

        rgb = rospy.get_param("~rgb", "/head_xtion/rgb/image_rect_color")
        subs.append(message_filters.Subscriber(rgb, ROSImage))
        subs.append(message_filters.Subscriber(rospy.get_param("~d", "/head_xtion/depth/image_rect_meters"), ROSImage))
        subs.append(message_filters.Subscriber('/'.join(rgb.split('/')[:-1] + ['camera_info']), CameraInfo))

        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.5)
        ts.registerCallback(self.cb)

    def cb(self, src, rgb, d, caminfo):
        header = rgb.header
        bridge = CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb)[:,:,::-1]  # Need to do BGR-RGB conversion manually.
        d = bridge.imgmsg_to_cv2(d)
        imgs = []
        for detrect in get_rects(src):
            detrect = self.getrect(*detrect)
            det_rgb = cutout(rgb, *detrect)
            det_d = cutout(d, *detrect)

            # Preprocess and stick into the minibatch.
            im = subtractbg(det_rgb, det_d, 1.0, 0.5)
            im = self.preproc(im)
            imgs.append(im)
            sys.stderr.write("\r{}".format(self.counter)) ; sys.stderr.flush()
            self.counter += 1

        if 0 < len(imgs):
            bits = [self.net.forward(batch) for batch in self.aug.augbatch_pred(np.array(imgs), fast=True)]
            preds = bit2deg(ensemble_biternions(bits))
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
                    l, t, w, h = self.getrect(*detrect)
                    px =  int(round(np.cos(np.deg2rad(alpha-90))*w/2))
                    py = -int(round(np.sin(np.deg2rad(alpha-90))*h/2))
                    cv2.rectangle(rgb_vis, (detrect[0], detrect[1]), (detrect[0]+detrect[2],detrect[1]+detrect[3]), (0,255,255), 1)
                    cv2.rectangle(rgb_vis, (l,t), (l+w,t+h), (0,255,0), 2)
                    cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px,t+h//2+py), (0,255,0), 2)
                    cv2.putText(rgb_vis, "{:.1f}".format(alpha), (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                vismsg = bridge.cv2_to_imgmsg(rgb_vis, encoding='rgb8')
                vismsg.header = header  # TODO: Seems not to work!
                self.pub_vis.publish(vismsg)

            if 0 < self.pub_pa.get_num_connections():
                fx, cx = caminfo.K[0], caminfo.K[2]
                fy, cy = caminfo.K[4], caminfo.K[5]

                poseArray = PoseArray()
                poseArray.header.stamp = header.stamp
                poseArray.header.frame_id = header.frame_id

                for (dx, dy, dw, dh, dd), alpha in zip(get_rects(src, with_depth=True), preds):
                    dx, dy, dw, dh = self.getrect(dx, dy, dw, dh)

                    # PoseArray message for boundingbox centres
                    pose = Pose()
                    pose.position.x = dd*((dx+dw/2.0-cx)/fx)
                    pose.position.y = dd*((dy+dh/2.0-cy)/fy)
                    pose.position.z = dd
                    # TODO: Use global UP vector (0,0,1) and transform into frame used by this message.
                    q = quaternion_about_axis(np.deg2rad(alpha - 90), [0, -1, 0])
                    pose.orientation.w = q[3] # No rotation atm.
                    pose.orientation.x = q[0]
                    pose.orientation.y = q[1]
                    pose.orientation.z = q[2]
                    poseArray.poses.append(pose)

                self.pub_pa.publish(poseArray)


if __name__ == "__main__":
    rospy.init_node("biternion_predict")

    # Add the "models" directory to the path!
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'scripts'))
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'models'))

    p = Predictor()
    rospy.spin()
    rospy.loginfo("Predicted a total of {} UBDs.".format(p.counter))
