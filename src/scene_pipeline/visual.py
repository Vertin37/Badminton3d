import cv2
import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
FONT=next((p for p in [Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists()),Path(''))
def text(img,items):
    pil=Image.fromarray(img);draw=ImageDraw.Draw(pil)
    for xy,message,size,color in items:
        font=ImageFont.truetype(str(FONT),size) if FONT.is_file() else ImageFont.load_default()
        draw.text(xy,message,font=font,fill=color)
    return np.asarray(pil).copy()

def camera(center,target,width,height,focal=None):
    center=np.asarray(center,float);forward=np.asarray(target)-center;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
    r=np.stack([right,down,forward]);focal=focal or width*.9
    def project(points):
        p=(np.asarray(points)-center)@r.T
        return p[...,:2]/np.maximum(p[...,2:3],.1)*focal+[width/2,height/2]
    return project,r,center

def align(a,b):
    a=a/max(np.linalg.norm(a),1e-8);b=b/max(np.linalg.norm(b),1e-8)
    v=np.cross(a,b);c=np.dot(a,b);s=np.linalg.norm(v)
    if s<1e-7:
        if c>0:return np.eye(3)
        v=np.cross(a,[1,0,0] if abs(a[0])<.8 else [0,1,0]);v/=np.linalg.norm(v)
        return 2*np.outer(v,v)-np.eye(3)
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew*((1-c)/(s*s))

def line3(img,project,points,color,thickness=1):
    uv=project(points)
    if np.isfinite(uv).all() and np.abs(uv).max()<30000:cv2.polylines(img,[np.rint(uv).astype(np.int32)],False,color,thickness,cv2.LINE_AA)

def court(img,project):
    poly=np.rint(project([[-3.05,-6.7,0],[3.05,-6.7,0],[3.05,6.7,0],[-3.05,6.7,0]])).astype(np.int32)
    cv2.fillConvexPoly(img,poly,(27,86,88),cv2.LINE_AA)
    for x in [-3.05,-2.59,2.59,3.05]:line3(img,project,[[x,-6.7,.005],[x,6.7,.005]],(175,218,210),1)
    for y in [-6.7,-5.94,-1.98,1.98,5.94,6.7]:line3(img,project,[[-3.05,y,.005],[3.05,y,.005]],(175,218,210),1)
    for y0,y1 in [(-6.7,-1.98),(1.98,6.7)]:line3(img,project,[[0,y0,.005],[0,y1,.005]],(175,218,210),1)
    line3(img,project,[[-3.05,0,0],[-3.05,0,1.55],[3.05,0,1.55],[3.05,0,0]],(201,215,227),2)
    for x in np.linspace(-3,3,25):line3(img,project,[[x,0,.78],[x,0,1.54]],(85,120,131),1)
    for z in np.linspace(.8,1.5,8):line3(img,project,[[-3.05,0,z],[3.05,0,z]],(85,120,131),1)

def draw_mesh(img,project,r,center,vertices,faces,color):
    uv=project(vertices);xyz=(vertices-center)@r.T
    tris=uv[faces];depth=xyz[faces,2].mean(1)
    normal=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
    light=.50+.50*np.abs(normal@np.array([.4,-.5,.77]));shades=np.clip(np.asarray(color)[None]*light[:,None],0,255).astype(np.uint8)
    valid=(depth>.3)&(tris[:,:,0].max(1)>=0)&(tris[:,:,0].min(1)<img.shape[1])&(tris[:,:,1].max(1)>=0)&(tris[:,:,1].min(1)<img.shape[0])
    area=np.abs(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]));valid&=area>.12
    for i in np.argsort(depth)[::-1]:
        if valid[i]:cv2.fillConvexPoly(img,np.rint(tris[i]).astype(np.int32),tuple(map(int,shades[i])),cv2.LINE_AA)

def racket(img,project,wrist,head,color):
    vector=head-wrist;direction=vector/np.linalg.norm(vector);a=np.cross(direction,[0,0,1.])
    if np.linalg.norm(a)<.05:a=np.cross(direction,[0,1.,0])
    a/=np.linalg.norm(a);b=np.cross(direction,a)
    handle=head-direction*.18;line3(img,project,[wrist,handle],(219,224,230),2)
    theta=np.linspace(0,2*np.pi,30);rim=head+np.cos(theta)[:,None]*a*.105+np.sin(theta)[:,None]*b*.14
    line3(img,project,rim,color,2)
    for q in [-.05,0,.05]:line3(img,project,[head+a*q-b*.11,head+a*q+b*.11],(175,195,205),1)
