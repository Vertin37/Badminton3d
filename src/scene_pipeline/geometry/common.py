"""High-resolution court geometry; no manually annotated image points."""
from pathlib import Path
import json,hashlib,os
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=Path(__file__).resolve().parents[3]
WIDTH,HEIGHT=960,540
XS=np.array([-3.05,-2.59,0.,2.59,3.05])
YS=np.array([-6.7,-5.94,-1.98,1.98,5.94,6.7])
POINTS=np.array([[x,y] for y in YS for x in XS],float)
CORNERS=POINTS[[0,4,29,25]].astype(np.float32)
SYMMETRIES=np.array([[0,1,2,3],[1,0,3,2],[3,2,1,0],[2,3,0,1]])
SEGMENTS=[np.array([[x,-6.7],[x,6.7]]) for x in XS if x!=0]
SEGMENTS += [np.array([[-3.05,y],[3.05,y]]) for y in YS]
SEGMENTS += [np.array([[0,-6.7],[0,-1.98]]),np.array([[0,1.98],[0,6.7]])]
KINDS=np.array([0]*4+[1]*6+[0]*2)
SAMPLED=np.stack([np.linspace(s[0],s[1],24) for s in SEGMENTS])

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert),encoding='utf-8');os.replace(temp,path)
def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return {} if default is None else default
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def project(h,points):
    p=np.asarray(points).reshape(-1,2);q=np.c_[p,np.ones(len(p))]@np.asarray(h).T
    with np.errstate(divide='ignore',invalid='ignore'):return q[:,:2]/q[:,2:3]
def homography(q):return cv2.getPerspectiveTransform(CORNERS,np.asarray(q,np.float32))
def box(image,width=WIDTH,height=HEIGHT):
    h,w=image.shape[:2];s=min(width/w,height/h)
    t=np.array([[s,0,(width-w*s)/2],[0,s,(height-h*s)/2],[0,0,1.]])
    return cv2.warpPerspective(image,t,(width,height),borderValue=(24,24,24)),t
def imwrite(path,image):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ok,b=cv2.imencode(path.suffix,image)
    if not ok:raise RuntimeError('Image encode failed')
    b.tofile(path)
def valid(h):
    q=project(h,CORNERS)
    if not np.isfinite(q).all() or abs(q).max()>WIDTH*6 or not cv2.isContourConvex(q.astype(np.float32)):return False
    area=abs(cv2.contourArea(q.astype(np.float32)))
    depth=np.c_[CORNERS,np.ones(4)]@np.asarray(h)[2]
    return bool(WIDTH*HEIGHT*.005<area<WIDTH*HEIGHT*2.5 and ((depth>0).all() or (depth<0).all()))
def overlay(image,h,color=(30,220,255)):
    result=image.copy()
    if h is not None:
        for s in SEGMENTS:
            p=project(h,s)
            if np.isfinite(p).all() and abs(p).max()<50000:cv2.line(result,tuple(p[0].round().astype(int)),tuple(p[1].round().astype(int)),color,max(1,round(image.shape[1]/700)),cv2.LINE_AA)
    return result
