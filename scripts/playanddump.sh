#super-script for playing and dumping data
#usage ./playanddump PATH_TO_BAG_FOLDER

rosparam set use_sim_time true

for i in $( ls $1 | grep .bag); do
  p=$(echo $i | sed 's/.bag//' | sed 's/.*\-//')
  screen -m -d roslaunch biternion dump_tracks.launch dir:=/work/kurin/spencer_data/dump/$p hfactor:=0.5 wfactor:=0.8 src:=rear &
  rosbag play --clock -r 1 $1$i
done



