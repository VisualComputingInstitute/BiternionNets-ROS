#super-script for playing and dumping data
#usage ./playanddump PATH_TO_BAG_FOLDER

rosparam set use_sim_time true

for i in $( ls $1 | grep .bag); do
  roslaunch spencer_rwth all.launch sensor_frame:=rear &
  foo=$!
  p=$(echo $i | sed 's/.bag//' | sed 's/.*\-//')
  roslaunch biternion dump_tracks.launch dir:=/work/kurin/spencer_data/dump/$p hfactor:=0.5 wfactor:=0.8 src:=rear subbg:=true &
  sleep 5
  rosbag play --clock -r 0.3 $1$i
  echo $foo
  kill $foo
  sleep 5
done



