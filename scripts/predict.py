#!/usr/bin/env python
# encoding: utf-8

from os.path import abspath, expanduser, join as pjoin
import sys
from importlib import import_module

import numpy as np
import cv2

import rospy
from tf import TransformListener, Exception as TFException
from tf.transformations import quaternion_about_axis
from rospkg import RosPack
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, QuaternionStamped
from biternion.msg import HeadOrientations
from visualization_msgs.msg import Marker

# Distinguish between STRANDS and SPENCER.
try:
    from rwth_perception_people_msgs.msg import UpperBodyDetector
    from spencer_tracking_msgs.msg import TrackedPersons2d, TrackedPersons
    HAS_TRACKED_PERSONS = True
except ImportError:
    from upper_body_detector.msg import UpperBodyDetector
    from mdl_people_tracker.msg import TrackedPersons2d
    HAS_TRACKED_PERSONS = False


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

        # Ugly workaround for "jumps back in time" that the synchronizer sometime does.
        self.last_stamp = rospy.Time()

        # Create and load the network.
        netlib = import_module(modelname)
        self.model = netlib.Model(weightsname, GPU=False)

        # Do a fake forward-pass for precompilation/GPU init/...
        self.model(np.zeros((480,640,3), np.uint8),
                   np.zeros((480,640), np.float32), [(0,0,150,450)])
        rospy.loginfo("BiternionNet initialized")

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

        tra3d = rospy.get_param("~tra3d", "")
        if src == "tra" and tra3d and HAS_TRACKED_PERSONS:
            self.pub_tracks = rospy.Publisher(topic + "/tracks", TrackedPersons, queue_size=3)
            subs.append(message_filters.Subscriber(tra3d, TrackedPersons))
            self.listener = TransformListener()
        else:
            self.pub_tracks = None

        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.5)
        ts.registerCallback(self.cb)

    def cb(self, src, rgb, d, caminfo, *more):
        # Ugly workaround because approximate sync sometimes jumps back in time.
        #if rgb.header.stamp <= self.last_stamp:
        #    rospy.logwarn("Jump back in time detected and dropped like it's hot")
        #    return

        self.last_stamp = rgb.header.stamp

        detrects = get_rects(src)

        # Early-exit to minimize CPU usage if possible.
        #if len(detrects) == 0:
        #    return

        # If nobody's listening, why should we be computing?
        listeners = sum(p.get_num_connections() for p in (self.pub, self.pub_vis, self.pub_pa))
        if self.pub_tracks is not None:
            listeners += self.pub_tracks.get_num_connections()
        if listeners == 0:
            return

        header = rgb.header
        bridge = CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
        d = bridge.imgmsg_to_cv2(d)

        # Do the extraction and prediction
        preds, confs = self.model(rgb, d, detrects)
        self.counter += len(preds)
        sys.stderr.write("\r{}".format(self.counter)) ; sys.stderr.flush()

        # Publish angle predictions
        if 0 < self.pub.get_num_connections():
            self.pub.publish(HeadOrientations(
                header=header,
                angles=list(preds),
                confidences=list(confs),
            ))

        # Visualization
        # TODO: Visualize confidence, too.
        if 0 < self.pub_vis.get_num_connections():
            rgb_vis = rgb.copy()
            for detrect, alpha in zip(detrects, preds):
                l, t, w, h = self.model.getrect(*detrect)
                px =  int(round(np.cos(np.deg2rad(alpha))*w/2))
                py = -int(round(np.sin(np.deg2rad(alpha))*h/2))
                cv2.rectangle(rgb_vis, (detrect[0], detrect[1]), (detrect[0]+detrect[2],detrect[1]+detrect[3]), (0,0,255), 1)
                cv2.rectangle(rgb_vis, (l,t), (l+w,t+h), (0,255,0), 2)
                cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px,t+h//2+py), (0,255,0), 2)
                # cv2.putText(rgb_vis, "{:.1f}".format(alpha), (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            vismsg = bridge.cv2_to_imgmsg(rgb_vis, encoding='rgb8')
            vismsg.header = header  # TODO: Seems not to work!
            self.pub_vis.publish(vismsg)

        if 0 < self.pub_pa.get_num_connections():
            fx, cx = caminfo.K[0], caminfo.K[2]
            fy, cy = caminfo.K[4], caminfo.K[5]

            poseArray = PoseArray(header=header)

            for (dx, dy, dw, dh, dd), alpha in zip(get_rects(src, with_depth=True), preds):
                dx, dy, dw, dh = self.model.getrect(dx, dy, dw, dh)

                # PoseArray message for boundingbox centres
                poseArray.poses.append(Pose(
                    position=Point(
                        x=dd*((dx+dw/2.0-cx)/fx),
                        y=dd*((dy+dh/2.0-cy)/fy),
                        z=dd
                    ),
                    # TODO: Use global UP vector (0,0,1) and transform into frame used by this message.
                    orientation=Quaternion(*quaternion_about_axis(np.deg2rad(alpha), [0, -1, 0]))
                ))

            self.pub_pa.publish(poseArray)

        if len(more) == 1 and self.pub_tracks is not None and 0 < self.pub_tracks.get_num_connections():
            t3d = more[0]
            try:
                self.listener.waitForTransform(header.frame_id, t3d.header.frame_id, rospy.Time(), rospy.Duration(1))
                for track, alpha in zip(t3d.tracks, preds):
                    track.pose.pose.orientation = self.listener.transformQuaternion(t3d.header.frame_id, QuaternionStamped(
                        header=header,
                        # TODO: Same as above!
                        quaternion=Quaternion(*quaternion_about_axis(np.deg2rad(alpha), [0, -1, 0]))
                    )).quaternion
                self.pub_tracks.publish(t3d)
            except TFException:
                pass


if __name__ == "__main__":
    rospy.init_node("biternion_predict")

    # Add the "models" directory to the path!
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'scripts'))
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'models'))

    p = Predictor()
    rospy.spin()
    rospy.loginfo("Predicted a total of {} UBDs.".format(p.counter))
