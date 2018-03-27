#!/bin/bash

#super-script for playing and dumping data
#usage ./playanddump PATH_TO_BAG_FOLDER PATH_TO_DUMP_FOLDER DUMPER_FLAGS...

BAG_FOLDER=$1 ; shift
DUMP_FOLDER=$1 ; shift

rosparam set use_sim_time true

roslaunch upper_body_detector upper_body_detector.launch &

for i in $(ls $BAG_FOLDER | grep .bag); do
    p=$(echo `basename $i` | sed 's/.bag//')
    # No need to kill as in ROS starting the same node again kills the old one!
    roslaunch biternion dump_ubdcpp.launch dir:=$DUMP_FOLDER/$p "$@" &
    #PID=$!
    sleep 5
    rosbag play --clock -r 0.5 $BAG_FOLDER/$i
    #echo $foo
    #kill $foo
    sleep 5
done
