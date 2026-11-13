"""Floor feature homographies, independent of moving target-player positions."""
import cv2,numpy as np
from .common import WIDTH,HEIGHT,project

def features(image,hint):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);hist=np.zeros(180)
    for p,w in zip(hint['feet'],hint['weights']):
        if not np.isfinite(p).all() or w<.25:continue
        x,y=p.round().astype(int);patch=hsv[max(0,y+4):min(HEIGHT,y+20),max(0,x-16):min(WIDTH,x+17)]
        if patch.size:
            good=(patch[:,:,1]>35)&(patch[:,:,2]>35);hist+=np.bincount(patch[:,:,0][good],minlength=180)
    hue=int(hist.argmax());d=abs(hsv[:,:,0].astype(float)-hue);d=np.minimum(d,180-d)
    mask=((d<22)&(hsv[:,:,1]>30)&(hsv[:,:,2]>30)).astype(np.uint8)*255
    mask=cv2.dilate(mask,np.ones((15,15),np.uint8)) if hist.sum()>20 else np.full((HEIGHT,WIDTH),255,np.uint8)
    for x0,y0,x1,y1 in hint['boxes']:
        cv2.rectangle(mask,(max(0,int(x0)-4),max(0,int(y0)-4)),(min(WIDTH-1,int(x1)+4),min(HEIGHT-1,int(y1)+4)),0,-1)
    sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.015,edgeThreshold=12)
    keys,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
    return {'points':np.array([k.pt for k in keys]).reshape(-1,2),'descriptors':desc,'image':image,'mask':mask}

def match(source,target):
    a,b=source['descriptors'],target['descriptors']
    if a is None or b is None:return None,{'status':'missing_features'}
    raw=cv2.BFMatcher().knnMatch(a,b,k=2)
    good=[pair[0] for pair in raw if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
    if len(good)<12:return None,{'status':'few_matches','matches':len(good)}
    p=source['points'][[m.queryIdx for m in good]];q=target['points'][[m.trainIdx for m in good]]
    cv2.setRNGSeed(20261113);h,inliers=cv2.findHomography(p,q,cv2.RANSAC,2.5,maxIters=2500,confidence=.995)
    if h is None:return None,{'status':'ransac_failed','matches':len(good)}
    valid=inliers.ravel()>0;count=int(valid.sum());fraction=float(valid.mean())
    spread=abs(cv2.contourArea(cv2.convexHull(p[valid].astype(np.float32))))/(WIDTH*HEIGHT)
    grid=np.array([[40,40],[WIDTH-40,40],[WIDTH-40,HEIGHT-40],[40,HEIGHT-40]],float);warped=project(h,grid)
    good_shape=np.isfinite(warped).all() and abs(warped).max()<WIDTH*5 and cv2.isContourConvex(warped.astype(np.float32))
    ratio=abs(cv2.contourArea(warped.astype(np.float32)))/((WIDTH-80)*(HEIGHT-80)) if good_shape else 0
    accepted=count>=12 and fraction>.35 and spread>.012 and .12<ratio<8
    return (h if accepted else None),{'status':'matched' if accepted else 'unreliable','matches':len(good),'inliers':count,'inlier_fraction':fraction,'spatial_fraction':float(spread)}
