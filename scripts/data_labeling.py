from os import listdir, remove, makedirs
from os.path import join, exists
from shutil import copy2 as cp

datadir = '/work/kurin/spencer_data/dump'
new_dir = '/work/kurin/spencer_data/new_dump'
end = 'rgb.png'

from_dict = {}
to_dict = {}


#form from_dict
f_d = datadir
f_d = join(datadir,'50')
for f in listdir(f_d):
  if(not f.startswith('.')):
    #p_d = {}
    #for p in listdir(join(f_d,f)):
    if not f.endswith('.png'):
      continue
    n,trid,pid,_ = f.rsplit('_')
    curr = from_dict.get(n)
    if curr is not None:
      if curr.get(trid) is not None:
        curr[trid].append(pid)
      else:
        from_dict[n][trid] = [pid]
    else:
      from_dict[n] = {trid:[pid]}

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

#iterate in to_dict and copy pics from from_dict with the same name, tid and pid

def crem(t,a,n,tid,pid,end):
    f_name = join(datadir, '50', "_".join([n, tid, pid, end]))
    t_name = join(new_dir, t, a, "_".join([n, tid, pid, end]))
    #csvf_name = join(datadir, 'dumps', n, "_".join([tid, pid, 'd.csv']))
    #csvt_name = join(new_dir, t, a, "_".join([n, tid, pid, 'd.csv']))
    cp(f_name, t_name)
    #cp(csvf_name, csvt_name)
    #remove(f_name)

print("Generating in progress...")
n_im = 0
for t in to_dict:
    for a in to_dict[t]:
        for n in to_dict[t][a]:
            for tid in to_dict[t][a][n]:
                pids = to_dict[t][a][n][tid]
                if pids is not None:
                  for p in pids:
                    if p in from_dict[n].get(tid, []):
                      crem(t,a,n,tid,p,end)
                      n_im+=1
print "Finished. Copied %s images" % (n_im)

