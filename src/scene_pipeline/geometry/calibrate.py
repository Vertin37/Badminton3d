"""Court calibration using painted lines and supplied 2D player tracks."""
import time
import cv2,numpy as np
from .common import WIDTH,HEIGHT,CORNERS,SYMMETRIES,project,valid,overlay,imwrite,atomic
from .observations import detect
from .seeds import generate
from .solver import score,refine,family_map

def distance(a,b):
    return float(np.linalg.norm(project(a,CORNERS)[None]-project(b,CORNERS)[SYMMETRIES],axis=-1).mean(1).min())

def calibrate(image,hints,output=None):
    started=time.perf_counter();obs=detect(image,hints)
    seeds,stats=generate(obs)
    print('generated',len(seeds),'seconds',round(time.perf_counter()-started,2),flush=True)
    for seed in seeds:seed['diagnostics']=score(seed,obs)
    seeds.sort(key=lambda s:-s['diagnostics']['score'])
    short=[]
    for seed in seeds:
        if not any(distance(seed['H'],other['H'])<10 for other in short):short.append(seed)
        if len(short)>=12:break
    optimized=[refine(seed,obs) for seed in short]
    optimized.sort(key=lambda s:(not s['diagnostics'].get('eligible',False),-s['diagnostics']['score']))
    chosen=optimized[0] if optimized else None
    def pack(seed):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in seed.items()}
    result={'homography':chosen['H'].tolist() if chosen else None,
            'status':'supported' if chosen and chosen['diagnostics']['eligible'] else 'ambiguous',
            'selected':pack(chosen) if chosen else None,'alternatives':[pack(s) for s in optimized[1:]],
            'candidate_count':len(seeds),'generation':stats,'observed_lines':len(obs['lines']),
            'floor_hue':obs['floor_hue'],'seconds':time.perf_counter()-started}
    if output is not None:
        output.mkdir(parents=True,exist_ok=True);atomic(output/'calibration.json',result);atomic(output/'candidate_pool.json',[pack(s) for s in seeds])
        visual=image.copy()
        for line in obs['lines']:
            p=line['segment'].round().astype(int);cv2.line(visual,tuple(p[:2]),tuple(p[2:]),(255,0,255),2)
        for hint in hints:
            for p in hint['feet']:
                if np.isfinite(p).all():cv2.circle(visual,tuple(p.round().astype(int)),4,(0,220,255),-1)
        imwrite(output/'observations.jpg',visual)
        imwrite(output/'floor.jpg',obs['floor']*255)
        for index,seed in enumerate(optimized[:8]):
            visual=overlay(image,seed['H']);d=seed['diagnostics']
            cv2.putText(visual,f"{index} {seed['origin']} score {d['score']:.3f} lines {d.get('matched_lines',0)}",(15,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            imwrite(output/f'candidate_{index}.jpg',visual)
    return result
