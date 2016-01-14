from os import listdir, remove, makedirs
from os.path import join, exists
from shutil import copy2 as cp

datadir = './dump'
new_dir = './new_dump'
end = 'rgb.png'

from_dict = {}
to_dict = {}


#form from_dict
f_d = join(datadir,'dumps')
for f in listdir(f_d):
  if(not f.startswith('.')):
    p_d = {}
    for p in listdir(join(f_d,f)):
      if not p.endswith('.png'):
        continue
      trid,pid,_ = p.rsplit('_')
      curr = p_d.get(trid)
      if curr:
        curr.append(pid)
        p_d[trid] = curr
      else:
        p_d[trid] = [pid]
    from_dict[f] = p_d

#form to_dict
d = {'4p':['backleft','backright', 'frontleft', 'frontright'], '4x':['back', 'front', 'left', 'right']}
to_dict = {'4p':{},'4x':{}}

#create all dirs we need
for k in d:
  for el in d[k]:
    path = join(new_dir, k, el)
    if not exists(path):
      makedirs(path)

for k in d:
    for el in d[k]:
        to_dict[k][el] = {}
        p = join(datadir, k, el)
        for f in listdir(p):
            if(not f.startswith('.')):
                name,trid,pid,_ = f.rsplit('_')
                if to_dict.get(k).get(el).get(name,0) == 0:
                    to_dict[k][el][name] = {}
                if to_dict.get(k).get(el).get(name).get(trid, 0) == 0:
                    to_dict[k][el][name][trid] = [pid]
                else:
                    to_dict[k][el][name][trid].append(pid)

#iterate inside to_dict and copy all pics from from_dict that are inside
#min and max inside every folder with respect to track id and name

def crem(t,a,n,tid,pid,end):
    f_name = join(datadir, 'dumps', n, "_".join([tid, pid, end]))
    t_name = join(new_dir, t, a, "_".join([n, tid, pid, end]))
    csvf_name = join(datadir, 'dumps', n, "_".join([tid, pid, 'd.csv']))
    csvt_name = join(new_dir, t, a, "_".join([n, tid, pid, 'd.csv']))
    #print(f_name, t_name)
    cp(f_name, t_name)
    cp(csvf_name, csvt_name)
    from_dict[n][tid].remove(pid)
    #remove(f_name)

for t in to_dict:
    for a in to_dict[t]:
        for n in to_dict[t][a]:
            for tid in to_dict[t][a][n]:
                pids = to_dict[t][a][n][tid]
                min_pid = min(pids)
                max_pid = max(pids)
                f_pids = from_dict.get(n, None)
                if f_pids:
                    f_pids = from_dict[n].get(tid, None)
                    if f_pids:
                        for pid in f_pids:
                           if min_pid<=pid<=max_pid:
                               crem(t, a, n, tid, pid, end)

