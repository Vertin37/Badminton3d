"""Vectorised metric-line assignments; alternative rectangles remain candidates."""
import itertools
import cv2,numpy as np
from .common import WIDTH,HEIGHT,XS,YS,CORNERS,SAMPLED,SYMMETRIES,project,valid
from .observations import family_maps,angle_delta

XPAIRS=np.array(list(itertools.combinations(XS,2)))
YPAIRS=np.array(list(itertools.combinations(YS,2)))
PRIORS=[]
for x in XPAIRS:
    for y in YPAIRS:
        PRIORS.append([[1/(x[1]-x[0]),0,-x[0]/(x[1]-x[0])],[0,1/(y[1]-y[0]),-y[0]/(y[1]-y[0])],[0,0,1.]])
PRIORS=np.asarray(PRIORS)

def equation(line):
    a,b=line['segment'].reshape(2,2);v=np.r_[a-b,0.]
    q=np.array([a[1]-b[1],b[0]-a[0],a[0]*b[1]-b[0]*a[1]])
    return q/max(np.linalg.norm(q[:2]),1e-9)

def select_pairs(lines,center):
    if len(lines)<2:return []
    nearest=sorted(lines,key=lambda l:np.linalg.norm(l['segment'].reshape(2,2).mean(0)-center))[:8]
    best=sorted(lines,key=lambda l:-l['score'])[:8]
    retained=[]
    for line in best+nearest:
        if not any(line is old for old in retained):retained.append(line)
    pairs=[]
    for a,b in itertools.combinations(retained,2):
        pa=a['segment'].reshape(2,2).mean(0);pb=b['segment'].reshape(2,2).mean(0)
        if np.linalg.norm(pa-pb)<14 or angle_delta(a['angle'],b['angle'])>80:continue
        qa,qb=equation(a),equation(b)
        # Parallel paint-edge duplicates must not become opposite court sides.
        if angle_delta(a['angle'],b['angle'])<3 and abs(np.dot(qa,np.r_[pb,1]))<8:continue
        score=np.sqrt(a['score']*b['score'])/(1+np.linalg.norm((pa+pb)/2-center)/(WIDTH*.5))
        pairs.append((score,qa,qb))
    return sorted(pairs,key=lambda p:-p[0])[:24]

def fast_score(hs,obs):
    points=SAMPLED[:,::3].reshape(-1,2);homogeneous=np.c_[points,np.ones(len(points))]
    p=np.einsum('nij,pj->npi',hs,homogeneous)
    p=p[:,:,:2]/p[:,:,2:3]
    inside=(p[:,:,0]>=0)&(p[:,:,0]<WIDTH)&(p[:,:,1]>=0)&(p[:,:,1]<HEIGHT)&np.isfinite(p).all(2)
    x=np.nan_to_num(p[:,:,0]).round().clip(0,WIDTH-1).astype(int);y=np.nan_to_num(p[:,:,1]).round().clip(0,HEIGHT-1).astype(int)
    d=obs['distance'][y,x]
    support=np.where(inside,np.exp(-d*d/18.),0).sum(1)/inside.sum(1).clip(1)
    support[inside.sum(1)<30]=-1
    return support

def generate(obs,thresholds=(10.,18.)):
    center=obs['median_feet'].mean(0)
    if not np.isfinite(center).all():return [],{}
    all_seeds=[];statistics={}
    corner_h=np.c_[CORNERS,np.ones(4)]
    feet_h=np.c_[obs['median_feet'],np.ones(2)]
    total=0
    for threshold in thresholds:
        families=family_maps(obs,threshold)
        pairs=[select_pairs(lines,center) for lines in families['groups']]
        matrices=[];maps=[];ids=[]
        for swap in (False,True):
            first=pairs[1] if swap else pairs[0];second=pairs[0] if swap else pairs[1]
            for a,b in itertools.product(first,second):
                intersections=[np.cross(a[1],b[1]),np.cross(a[2],b[1]),np.cross(a[2],b[2]),np.cross(a[1],b[2])]
                points=np.asarray(intersections)
                if (abs(points[:,2])<1e-7).any():continue
                image=points[:,:2]/points[:,2:3]
                if abs(image).max()>WIDTH*6 or not cv2.isContourConvex(image.astype(np.float32)) or abs(cv2.contourArea(image.astype(np.float32)))<40:continue
                unit=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),image.astype(np.float32))
                hs=np.einsum('ij,njk->nik',unit,PRIORS)
                matrices.append(hs);maps.extend([int(swap)]*len(hs));ids.extend(np.arange(len(hs))+int(swap)*len(PRIORS))
        if not matrices:
            statistics[str(threshold)]={'families':[len(g) for g in families['groups']],'generated':0};continue
        hs=np.concatenate(matrices);maps=np.array(maps);ids=np.array(ids)
        total+=len(hs)
        p=np.einsum('nij,pj->npi',hs,corner_h)
        den=p[:,:,2];same=(den>1e-7).all(1)|(den<-1e-7).all(1)
        q=p[:,:,:2]/np.where(abs(den[:,:,None])>1e-12,den[:,:,None],1e-12)
        edges=np.roll(q,-1,axis=1)-q
        turns=edges[:,:,0]*np.roll(edges,-1,axis=1)[:,:,1]-edges[:,:,1]*np.roll(edges,-1,axis=1)[:,:,0]
        convex=(turns>0).all(1)|(turns<0).all(1)
        area=abs(np.sum(q[:,:,0]*np.roll(q[:,:,1],-1,axis=1)-q[:,:,1]*np.roll(q[:,:,0],-1,axis=1),axis=1))*.5
        keep=same&convex&np.isfinite(q).all((1,2))&(abs(q).max((1,2))<WIDTH*5)&(area>WIDTH*HEIGHT*.012)&(area<WIDTH*HEIGHT*2)
        # Containment and the projected net line are evaluated before costly
        # line scoring. No missing-pair default positive score is introduced.
        orientation=np.sign(np.sum(q[:,:,0]*np.roll(q[:,:,1],-1,axis=1)-q[:,:,1]*np.roll(q[:,:,0],-1,axis=1),axis=1))
        v=obs['median_feet'][None,:,None]-q[:,None]
        cross=edges[:,None,:,0]*v[:,:,:,1]-edges[:,None,:,1]*v[:,:,:,0]
        margin=10*np.linalg.norm(edges,axis=2)
        keep&=(cross*orientation[:,None,None]>=-margin[:,None]).all((1,2))
        net=np.cross(hs[:,:,0],hs[:,:,2]) # image line through world x at y=0
        sides=np.einsum('ni,pi->np',net,feet_h)
        keep&=sides[:,0]*sides[:,1]<=0
        hs=hs[keep];maps=maps[keep];ids=ids[keep]
        scores=np.concatenate([fast_score(hs[i:i+1024],obs) for i in range(0,len(hs),1024)]) if len(hs) else np.empty(0)
        selected=[]
        for prior in np.unique(ids):
            group=np.flatnonzero(ids==prior);selected.extend(group[np.argsort(scores[group])[-3:]].tolist())
        selected=sorted(set(selected),key=lambda i:-scores[i])[:256]
        for i in selected:
            all_seeds.append({'origin':'metric_line_combinations','H':hs[i]/hs[i,2,2],'family_map':(maps[i],1-maps[i]),'partition':threshold,'quick_score':float(scores[i])})
        statistics[str(threshold)]={'families':[len(g) for g in families['groups']],'pairs':[len(p) for p in pairs],'after_pair_gate':len(hs),'retained':len(selected)}
    all_seeds.sort(key=lambda seed:-seed['quick_score'])
    diverse=[]
    for seed in all_seeds:
        q=project(seed['H'],CORNERS)
        if any(np.linalg.norm(q[None]-project(old['H'],CORNERS)[SYMMETRIES],axis=-1).mean(1).min()<5 for old in diverse):continue
        diverse.append(seed)
        if len(diverse)>=160:break
    statistics['total_enumerated']=total
    return diverse,statistics
