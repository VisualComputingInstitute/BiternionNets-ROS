#!/usr/bin/env python

from os import listdir, remove, makedirs
from os.path import join, exists
from shutil import copy2 as cp
from sys import argv, exit
from collections import defaultdict

if len(argv) != 4:
    print("Usage: {} FROM_DIR LABELS_DIR TO_DIR".format(argv[0]))
    print("")
    print("FROM_DIR is the directory containing all images.")
    print("LABELS_DIR needs to contain two sub-directories `4x` and `4p`.".format(argv[0]))
    print("TO_DIR is the target directory within which new `4x` and `4p` folders are created.".format(argv[0]))
    exit(1)

datadir = argv[1]
lbl_dir = argv[2]
new_dir = argv[3]
end = 'rgb.png'

print("Collection in progress...")

from_dict = defaultdict(lambda: defaultdict(list))
to_dict = {}

# form from_dict, which contains all unlabelled source files.
for f in listdir(datadir):
  if not f.startswith('.') and f.endswith('.png'):
    name,trid,pid,_ = f.rsplit('_')
    from_dict[name][trid].append(pid)

#form to_dict
d = {'4p': ['backleft','backright', 'frontleft', 'frontright'],
     '4x': ['back', 'front', 'left', 'right']}
to_dict = {'4p': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
           '4x': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))}

#create all dirs we need
for k in d:
  for lbl in d[k]:
    path = join(new_dir, k, lbl)
    if not exists(path):
      makedirs(path)

for k in d:
    for lbl in d[k]:
        for f in listdir(join(lbl_dir, k, lbl)):
            if not f.startswith('.'):
                name,trid,pid,_ = f.rsplit('_')
                to_dict[k][lbl][name][trid].append(pid)

#iterate in to_dict and copy pics from from_dict with the same name, tid and pid

def crem(t,lbl,name,tid,pid,end):
    f_name = join(datadir, "_".join([name, tid, pid, end]))
    t_name = join(new_dir, t, lbl, "_".join([name, tid, pid, end]))
    cp(f_name, t_name)

print("Generation in progress...")
n_im = 0
for t in to_dict:
    for lbl in to_dict[t]:
        for name in to_dict[t][lbl]:
            for tid in to_dict[t][lbl][name]:
                for p in to_dict[t][lbl][name][tid]:
                    if p in from_dict[name][tid]:
                        crem(t,lbl,name,tid,p,end)
                        n_im+=1

print "Finished. Copied %s images" % (n_im)
