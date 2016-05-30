# BiternionNets-ROS
An implementation of BiternionNets for ROS, ready to run on a robot.

Install instructions
--------------------

Most ROS installations are using Ubuntu, this is the recommended way:

```
$ sudo apt-get install python-virtualenv libopenblas-dev liblapack-dev
$ virtualenv --system-site-packages pyenv
$ . pyenv/bin/activate
$ pip install --upgrade numpy
$ pip install git+https://github.com/Theano/Theano.git
$ pip install git+https://github.com/lucasb-eyer/DeepFried2.git
$ pip install git+https://github.com/lucasb-eyer/lbtoolbox.git
```

You can now do the following now and hope there's no error.
It may take a while since it is pre-compiling quite some things:

```
$ python -c 'import DeepFried2'
```

## How to prepare data

This tutorial explains how to collect, label, load and use data to reproduce our experiments. The original paper is [here](http://www.vision.rwth-aachen.de/publication/0021/).


### Collection

We used robot's video camera to capture the person rotating on the spot. Then we used our upper body detector, cut out the head pictures out of it (so the position was pretty the same in our case).

### Labeling

Now when we have all the pictures of the heads, we rescale them to the same size and split all the dataset two times:

/4p<br>
_______/backleft<br>
_______/backright<br>
_______/frontleft<br>
_______/frontright<br>
/4x<br>
_______/back<br>
_______/front<br>
_______/left<br>
_______/right<br>

4p and 4x folders contain the same pictures but splitted sorted differently. p here stands for plus(+), x for x. The quick idea is the following:

<img class='center' src="pic/labeling.png"/>

### Training

To train run scripts/train.py:

```bash
python train.py -d data_dir
```

data_dir should contain 4p and 4x folders that we prepared on the previous step. To get other parameters run:

```bash
python train.py -h
```

After training you will have .npz file. And now it's time to use it for prediction.

### Prediction

Prediction parameters are in /launch/predict.launch file. Specify your model ('model' argument) here and .npz file ('weights').

Prepare ROS for prediction:

```bash
source /opt/ros/indigo/setup.bash #source your ROS setup file
roscore #start roscore
rosparam set use_sim_time true #make ros use simulation time
roslaunch spencer_rwth all.launch sensor_frame:=front
rosbag play --clock -r 0.3 bagfile_here #play some data with rosbag
```


Run the prediction node finally:

```bash
roslaunch biternion predict.launch
```

After that you can open rviz and check your predictions.

