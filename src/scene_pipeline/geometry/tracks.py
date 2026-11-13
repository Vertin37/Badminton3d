"""Existing 2D target-player tracks, with confidence and short-gap handling."""
import csv
import numpy as np
from scipy.ndimage import median_filter
from .common import PROJECT,digest
PATHS={'test':PROJECT/'outputs/pose_test/pose_data_stable.csv','test2':PROJECT/'outputs/test2_analysis/pose_data_stable.csv','test3':PROJECT/'outputs/test3_automatic_3d/01_pose/pose_data_tracked.csv'}

class Tracks:
    def __init__(self,name,n,player_ids=(0,1),path=None):
        self.path=PATHS[name] if path is None else path
        self.positions=np.full((n,2,17,2),np.nan);self.confidence=np.zeros((n,2,17));self.player_ids=tuple(player_ids)
        with self.path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                frame=int(row['frame_id'])-1;person=int(row['player_id']);joint=int(row['keypoint_id'])
                if not 0<=frame<n or person not in player_ids or not 0<=joint<17:continue
                confidence=float(row['confidence'])
                if row.get('raw_valid','1')=='0' or confidence<.3:continue
                xy=[float(row['x']),float(row['y'])]
                if not np.isfinite(xy).all():continue
                p=player_ids.index(person);self.positions[frame,p,joint]=xy;self.confidence[frame,p,joint]=confidence
        self.feet=np.full((n,2,2),np.nan);self.weights=np.zeros((n,2));self.bounds=np.full((n,2,4),np.nan)
        for frame in range(n):
            for p in range(2):
                good=self.confidence[frame,p,[15,16]]>.3
                if good.any():
                    self.feet[frame,p]=self.positions[frame,p,[15,16]][good].mean(0)
                    self.weights[frame,p]=self.confidence[frame,p,[15,16]][good].mean()
                valid=self.confidence[frame,p]>.3
                if valid.sum()>=5:
                    points=self.positions[frame,p,valid];lo=points.min(0);hi=points.max(0)
                    self.bounds[frame,p]=[lo[0]-8,lo[1]-8,hi[0]+8,hi[1]+8]
        for p in range(2):
            known=np.flatnonzero(self.weights[:,p]>.0)
            for a,b in zip(known[:-1],known[1:]):
                if 1<b-a<=15:
                    for frame in range(a+1,b):
                        t=(frame-a)/(b-a);self.feet[frame,p]=(1-t)*self.feet[a,p]+t*self.feet[b,p];self.weights[frame,p]=min(self.weights[a,p],self.weights[b,p])*.6
        self.sha256=digest(self.path)
    def get(self,frame,transform):
        from .common import project
        feet=self.feet[frame]
        mapped=project(transform,feet)
        boxes=[]
        for bound in self.bounds[frame]:
            if np.isfinite(bound).all():
                q=project(transform,bound.reshape(2,2));boxes.append([*q[0],*q[1]])
        return {'feet':mapped,'weights':self.weights[frame].copy(),'boxes':boxes,'source_frame':int(frame),
                'pair_valid':bool(np.isfinite(mapped).all() and (self.weights[frame]>.25).all())}
