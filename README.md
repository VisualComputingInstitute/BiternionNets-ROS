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
