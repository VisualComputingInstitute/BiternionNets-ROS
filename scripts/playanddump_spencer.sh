#!/bin/bash

# TODO WRITE THIS ACTUALLY, ITS JUST COPY PASTA SO FAR
#super-script for playing and dumping data
#usage ./playanddump PATH_TO_BAG_FOLDER PATH_TO_DUMP_FOLDER DUMPER_FLAGS...

BAG_FOLDER=$1 ; shift
DUMP_FOLDER=$1 ; shift

rosparam set use_sim_time true

for i in $(ls $BAG_FOLDER | grep .bag); do
    p=$(echo `basename $i` | sed 's/.bag//')
    # No need to kill as in ROS starting the same node again kills the old one!
    # NOTE: This won't work if the UBD has respawn=true in the launchfile, which is the default.
    roslaunch spencer_rwth all.launch sensor_frame:=rear hog:=false &
    sleep 5
    roslaunch biternion dump_tracks.launch src:=rear dir:=$DUMP_FOLDER/$p "$@" &
    #hfactor:=0.6 subbg:=true
    sleep 5
    rosbag play --clock -r 0.1 $BAG_FOLDER/$i
    sleep 5
done
