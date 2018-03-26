#!/usr/bin/env python
# encoding: utf-8

from os.path import abspath, expanduser, join as pjoin
import os
from sys import stderr

import cv2
import rospy
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image as ROSImage

# Distinguish between STRANDS and SPENCER.
try:
    from rwth_perception_people_msgs.msg import UpperBodyDetector
except ImportError:
    from upper_body_detector.msg import UpperBodyDetector



def cutout(img, detrect, hfact=3):
    x, y, w, h = detrect

    # Need to be careful for negative indices in conjunction with
    # numpy's (and thus OpenCV's) wrap-around.
    y2, x2 = y+hfact*h, x+w
    y1, x1 = max(y, 0), max(x, 0)
    return img[y1:y2, x1:x2]


class Dumper(object):
    def __init__(self):
        rospy.loginfo("Initializing UBD dumper")
        self.counter = 0

        self.dirname = abspath(expanduser(rospy.get_param("~dir", ".")))
        self.full = rospy.get_param("~fullbody", "False") in ("True", "true", "yes", "1", 1, True)
        self.hfact = 3 if self.full else 1

        rospy.loginfo("Dumping {} into {}".format("full-bodies" if self.full else "upper-bodies", self.dirname))

        # Create target directory, or warn if non-empty.
        try:
            if len(os.listdir(self.dirname)) > 0:
                rospy.logwarn("CAREFUL, files may be overwritten since directory is not empty: {}".format(self.dirname))
        except OSError:
            os.makedirs(self.dirname)

        subs = [
            message_filters.Subscriber(rospy.get_param("~ubd", "/upper_body_detector/detections"), UpperBodyDetector),
            message_filters.Subscriber(rospy.get_param("~rgb", "/head_xtion/rgb/image_rect_color"), ROSImage),
            message_filters.Subscriber(rospy.get_param("~d", "/head_xtion/depth/image_rect_meters"), ROSImage),
        ]

        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=50, slop=0.1)
        ts.registerCallback(self.cb)

    def cb(self, ubd, rgb, d):
        b = CvBridge()
        rgb = b.imgmsg_to_cv2(rgb)[:,:,::-1]  # Need to do BGR-RGB conversion manually.
        d = b.imgmsg_to_cv2(d)

        for i, detrect in enumerate(zip(ubd.pos_x, ubd.pos_y, ubd.width, ubd.height)):
            cv2.imwrite(pjoin(self.dirname, "{:06d}.png".format(self.counter)), cutout(rgb, detrect, self.hfact))
            det_d = cutout(d, detrect, self.hfact)

            stderr.write("\r{}".format(self.counter)) ; stderr.flush()
            self.counter += 1


if __name__ == "__main__":
    rospy.init_node("dump_ubd")
    d = Dumper()
    rospy.spin()
    rospy.loginfo("Dumped a total of {} UBDs.".format(d.counter))
