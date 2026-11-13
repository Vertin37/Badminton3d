"""Content-addressed stages, atomic state and input validation."""
from pathlib import Path
import hashlib,json,os,sys
import cv2,numpy as np

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8');os.replace(tmp,p)

def video_info(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise FileNotFoundError('Cannot open video: '+str(path))
    result=dict(frames=int(cap.get(7)),width=int(cap.get(3)),height=int(cap.get(4)),fps=float(cap.get(5)))
    cap.release()
    if result['frames']<12 or result['fps']<=0 or not np.isfinite(result['fps']):raise ValueError('Invalid/too short video')
    return result

def cached(record,signature):
    if not Path(record).exists():return False
    try:
        r=read(record)
        return r['signature']==signature and all(Path(p).is_file() and digest(p)==sha for p,sha in r['outputs'].items())
    except (OSError,ValueError,KeyError):return False

def finish(record,signature,outputs):
    for path in outputs:
        if not Path(path).is_file():raise RuntimeError('Stage did not create '+str(path))
    write(record,dict(signature=signature,outputs={str(p):digest(p) for p in outputs}))
