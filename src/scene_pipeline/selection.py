"""Existing person tracker, adaptive frame count and confidence-aware court gate."""
import csv
from pathlib import Path
import cv2,numpy as np
from .artifacts import read,write,video_info
from .fusion import CourtReference,near_target_court

def load_detections(path,n):
    by_frame=[{} for _ in range(n)]
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            j=int(row['keypoint_id']);p=int(row.get('person_id',row.get('player_id')));i=int(row['frame_id'])-1
            if not 0<=i<n:raise ValueError('Detection frame outside input video')
            if j>=17:continue
            try:xyz=np.array([float(row['x']),float(row['y']),float(row['confidence'])],np.float32)
            except (ValueError,TypeError):xyz=np.zeros(3,np.float32)
            if row.get('raw_valid')=='0':xyz[2]=0
            if not np.isfinite(xyz).all():xyz=np.zeros(3,np.float32)
            by_frame[i].setdefault(p,np.zeros((17,3),np.float32))[j]=xyz
    return [np.stack(list(frame.values())) if frame else np.zeros((0,17,3),np.float32) for frame in by_frame]

def select(video,detections,output,config,court=None):
    import run_pose_test as existing
    cv2.setNumThreads(1);info=video_info(video);n=info['frames'];raw=load_detections(detections,n)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ref=CourtReference(read(court)) if court is not None else None
    cap=cv2.VideoCapture(str(video));tracker=existing._make_tracker();cuts=[0];previous=None;stats=[]
    with (output/'pose_data_tracked.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(existing.TRACKED_CSV_FIELDS)
        for i in range(n):
            ok,image=cap.read()
            if not ok:raise RuntimeError('Video decode failed')
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            mask=((hsv[:,:,0]>config['floor_hue_min'])&(hsv[:,:,0]<config['floor_hue_max'])&(hsv[:,:,1]>45)&(hsv[:,:,2]>40)).astype(np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8));count,labels,st,_=cv2.connectedComponentsWithStats(mask)
            floor=(labels==(1+st[1:,4].argmax())).astype(np.uint8) if count>1 else mask
            floor=cv2.dilate(floor,np.ones((51,51),np.uint8));h=ref.at(i) if ref else None
            people=[];feedback_rejected=0
            for person in raw[i]:
                good=person[:,2]>.3;feet=person[[15,16]];visible=feet[:,2]>.25
                if good.sum()<7 or np.ptp(person[good,1])<image.shape[0]*config['min_body_height_fraction'] or not visible.any():continue
                xy=feet[visible,:2].mean(0);pixel=np.clip(xy.round().astype(int),[0,0],[info['width']-1,info['height']-1])
                if not floor[pixel[1],pixel[0]]:continue
                image_drop=np.ptp(person[good,1])*config['airborne_image_drop_fraction']
                if h is not None and not near_target_court(h,xy,config['court_feedback_margin_m'],image_drop):feedback_rejected+=1;continue
                people.append(person)
            people=np.asarray(people) if people else np.zeros((0,17,3),np.float32)
            if len(people)==2:
                centers=people[:,[5,6,11,12],:2].mean(1);heights=np.array([np.ptp(p[p[:,2]>.3,1]) for p in people])
                if previous is not None:
                    d=np.linalg.norm(centers[:,None]-previous[None],axis=2)/np.maximum(heights[:,None],80)
                    jump=min(max(d[0,0],d[1,1]),max(d[0,1],d[1,0]))
                    if jump>config['cut_jump_fraction'] and i-cuts[-1]>15:cuts.append(i);tracker=existing._make_tracker()
                previous=centers
            else:previous=None
            kps=people[:,:,:2];scores=people[:,:,2]
            results=tracker.update(kps,scores,appearance_features=existing._extract_appearance_features(image,kps,scores))
            existing._write_tracked_rows(writer,i+1,results)
            stats.append(dict(frame=i,candidates=len(people),matched=sum(r.matched for r in results),court_feedback_used=h is not None,court_rejected=feedback_rejected))
            if i%300==0:print('selection',i,'/',n,flush=True)
    cap.release()
    existing.run_temporal_filter(output/'pose_data_tracked.csv',output,fps=info['fps'])
    write(output/'selection.json',dict(automatic=True,manual_points=False,cut_frames=cuts,frames=stats,
        two_candidates_frames=sum(s['candidates']==2 for s in stats),court_feedback_frames=sum(s['court_feedback_used'] for s in stats),court_rejected_detections=sum(s['court_rejected'] for s in stats)))
