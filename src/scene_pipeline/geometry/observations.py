"""Floor-supported line segments, excluding people and isolated bright clutter."""
import cv2,numpy as np
from .common import WIDTH,HEIGHT

def angle_delta(a,b):
    d=np.abs(np.asarray(a)-b)%180
    return np.minimum(d,180-d)

def detect(image,hints):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hist=np.zeros(180,float)
    body_mask=np.zeros((HEIGHT,WIDTH),np.uint8)
    pairs=[]
    for index,hint in enumerate(hints):
        if hint['pair_valid']:pairs.append(hint['feet'])
        for x0,y0,x1,y1 in (hint['boxes'] if index==0 else []):
            lo=np.maximum([x0,y0],0).astype(int);hi=np.minimum([x1,y1],[WIDTH-1,HEIGHT-1]).astype(int)
            if (hi>lo).all():cv2.rectangle(body_mask,tuple(lo),tuple(hi),255,-1)
        for point,weight in zip(hint['feet'],hint['weights']):
            if weight<.25 or not np.isfinite(point).all():continue
            x,y=point.round().astype(int);patch=hsv[max(0,y+3):min(HEIGHT,y+18),max(0,x-18):min(WIDTH,x+19)]
            if not patch.size:continue
            valid=(patch[:,:,1]>35)&(patch[:,:,2]>35)
            hist+=np.bincount(patch[:,:,0][valid],minlength=180)*weight
    hue=int(np.argmax(hist))
    hue_distance=angle_delta(hsv[:,:,0].astype(float),hue)
    floor=((hue_distance<20)&(hsv[:,:,1]>30)&(hsv[:,:,2]>30)).astype(np.uint8)
    if hist.sum()<30:floor=np.ones_like(floor)
    floor=cv2.morphologyEx(floor,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    top=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
    white=((hsv[:,:,2]>105)&(hsv[:,:,1]<165)&(top>8)).astype(np.uint8)
    white[body_mask>0]=0
    # The observation is a set of long segments, not every white pixel. This
    # stops shuttlecocks, reflections and clothes from directly supporting paint.
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected=lsd.detect(gray)[0]
    lines=[]
    if detected is not None:
        for segment in detected[:,0]:
            a,b=segment.reshape(2,2);v=b-a;length=float(np.linalg.norm(v))
            if length<max(26,WIDTH*.025):continue
            normal=np.array([-v[1],v[0]])/length
            sample=np.linspace(a,b,max(12,int(length/4)))
            def values(mask,offset):
                p=np.round(sample+normal*offset).astype(int);p[:,0]=p[:,0].clip(0,WIDTH-1);p[:,1]=p[:,1].clip(0,HEIGHT-1)
                return mask[p[:,1],p[:,0]]
            support=float(np.maximum.reduce([values(white,o) for o in (-2,-1,0,1,2)]).mean())
            ground=float(np.maximum(values(floor,-5),values(floor,5)).mean())
            foreground=float(values(body_mask,0).mean()/255)
            if support<.42 or ground<.7 or foreground>.2:continue
            angle=float(np.degrees(np.arctan2(v[1],v[0]))%180)
            score=length*support*ground
            lines.append({'segment':segment.astype(float),'length':length,'angle':angle,'score':score,'white_support':support,'floor_support':ground})
    lines.sort(key=lambda line:-line['score'])
    unique=[]
    for line in lines:
        a,b=line['segment'].reshape(2,2);v=b-a;normal=np.array([-v[1],v[0]])/line['length'];mid=(a+b)/2
        duplicate=False
        for old in unique:
            if angle_delta(line['angle'],old['angle'])>2.:continue
            p,q=old['segment'].reshape(2,2)
            if max(abs(np.dot(p-mid,normal)),abs(np.dot(q-mid,normal)))>2.3:continue
            along=v/line['length'];interval=np.array([np.dot(p-a,along),np.dot(q-a,along)])
            if interval.max()>=-12 and interval.min()<=line['length']+12:
                ends=np.array([a,b,p,q]);projections=ends@along
                old['segment']=np.r_[ends[projections.argmin()],ends[projections.argmax()]]
                old['length']=float(np.ptp(projections));old['score']=max(old['score'],line['score']);duplicate=True;break
        if not duplicate:unique.append(line)
        if len(unique)>=100:break
    if pairs:
        median=np.median(np.stack(pairs),axis=0);axis_vector=median[1]-median[0]
        axis=float(np.degrees(np.arctan2(axis_vector[1],axis_vector[0]))%180)
    else:median=np.full((2,2),np.nan);axis=0.
    binary=np.zeros((HEIGHT,WIDTH),np.uint8)
    for line in unique:
        p=line['segment'].round().astype(int);cv2.line(binary,tuple(p[:2]),tuple(p[2:]),1,2,cv2.LINE_AA)
    distance=cv2.distanceTransform(1-binary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    floor_distance=cv2.distanceTransform(1-floor,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    floor_distance[body_mask>0]=0
    return {'image':image,'floor':floor,'floor_distance':floor_distance,'body_mask':body_mask,'binary':binary,'distance':distance,
            'lines':unique,'axis':axis,'median_feet':median,'floor_hue':hue,'hints':hints}

def family_maps(obs,threshold):
    groups=[[],[]];masks=[np.zeros((HEIGHT,WIDTH),np.uint8) for _ in range(2)]
    for line in obs['lines']:
        family=0 if angle_delta(line['angle'],obs['axis'])<threshold else 1
        groups[family].append(line)
        p=line['segment'].round().astype(int);cv2.line(masks[family],tuple(p[:2]),tuple(p[2:]),1,2,cv2.LINE_AA)
    distances=[cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE) for mask in masks]
    return {'groups':groups,'masks':masks,'distances':distances,'threshold':threshold}
