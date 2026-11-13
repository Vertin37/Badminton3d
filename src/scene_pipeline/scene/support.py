from pathlib import Path
import numpy as np,json,hashlib
BODY_IDS=np.array([16,17,18,19,20,21,1,2,4,5,7,8])
COCO_IDS=np.arange(5,17)
COURT=np.array([[x,y,0] for y in [-6.7,-5.94,-1.98,1.98,5.94,6.7] for x in [-3.05,-2.59,0,2.59,3.05]]+
    [[-3.05,0,0],[3.05,0,0],[-3.05,0,1.55],[3.05,0,1.55],[0,0,1.524]],float)


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def template(asset):
    z=np.load(asset,allow_pickle=True);v=z['v_template'];w=z['weights'];faces=z['f'];patches=[]
    for ankle,toe in [(7,10),(8,11)]:
        ids=np.flatnonzero(w[:,ankle]+w[:,toe]>=.5);ids=ids[v[ids,1]<=np.quantile(v[ids,1],.18)];split=np.median(v[ids,2])
        patches.extend([ids[v[ids,2]<=split],ids[v[ids,2]>split]])
    _,cluster=np.unique(np.floor(v/.035).astype(int),axis=0,return_inverse=True);counts=np.bincount(cluster);mf=cluster[faces]
    mf=mf[(mf[:,0]!=mf[:,1])&(mf[:,0]!=mf[:,2])&(mf[:,1]!=mf[:,2])];_,ix=np.unique(np.sort(mf,axis=1),axis=0,return_index=True);mf=mf[np.sort(ix)]
    return patches,cluster,counts,mf
