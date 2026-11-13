"""Bounded court-to-person feedback and independent geometry quality guards."""
import numpy as np

def project(h,points):
    p=np.c_[np.asarray(points).reshape(-1,2),np.ones(np.asarray(points).reshape(-1,2).shape[0])]@np.asarray(h).T
    with np.errstate(divide='ignore',invalid='ignore'):return p[:,:2]/p[:,2:]

def trusted(calibration):
    d=(calibration.get('selected') or {}).get('diagnostics',{})
    return calibration.get('status')=='supported' and d.get('matched_lines',0)>=5 and min(d.get('matched_by_direction',[0,0]))>=2 and d.get('camera_metric_error',1e9)<6 and d.get('player_containment',0)>.78

class CourtReference:
    def __init__(self,result):
        self.rows=result['frames'];self.indices=np.array([r['source_frame'] for r in self.rows]);self.groups={g['id']:g for g in result['groups']}
    def at(self,frame):
        i=int(np.searchsorted(self.indices,frame));i=min(i,len(self.indices)-1)
        if i>0 and abs(self.indices[i-1]-frame)<abs(self.indices[i]-frame):i-=1
        row=self.rows[i];group=self.groups[row['group']]
        if not trusted(group['calibration']) or row['homography_metric_to_original'] is None:return None
        return np.asarray(row['homography_metric_to_original'])

def near_target_court(h,feet,margin=2.,image_drop=0.):
    """Loose bounds allow saves and airborne ankle projection errors."""
    if h is None:return True
    # Before 3-D contact is available an ankle can be airborne. Permit a
    # bounded image-down interval; do not equate its pixel with floor contact.
    hypotheses=np.asarray(feet).reshape(1,2)+np.array([[0,0],[0,image_drop*.5],[0,image_drop]])
    try:world=project(np.linalg.inv(h),hypotheses)
    except np.linalg.LinAlgError:return True
    return bool((np.isfinite(world).all(1)&(abs(world)<[3.05+margin,6.7+margin]).all(1)).any())

def accept_feedback(baseline,candidate,max_corner_shift=45.):
    """Reject feedback drift; learned/self-consistency score alone cannot win."""
    if len(baseline['groups'])!=len(candidate['groups']):return False,'view_partition_changed'
    corners=np.array([[-3.05,-6.7],[3.05,-6.7],[3.05,6.7],[-3.05,6.7]])
    for old,new in zip(baseline['groups'],candidate['groups']):
        a,b=old['calibration'],new['calibration']
        if not trusted(b):return False,'feedback_court_unsupported'
        if not trusted(a):continue
        x=a['selected']['diagnostics'];y=b['selected']['diagnostics']
        if y['forward_line_support']+0.01<x['forward_line_support']:return False,'line_support_worse'
        if y['matched_lines']<x['matched_lines']:return False,'fewer_matched_lines'
        if y['floor_support']+.03<x['floor_support']:return False,'floor_support_worse'
        qa=project(np.asarray(a['homography']),corners);qb=project(np.asarray(b['homography']),corners)
        sym=np.array([[0,1,2,3],[1,0,3,2],[3,2,1,0],[2,3,0,1]])
        shift=np.linalg.norm(qa[None]-qb[sym],axis=-1).mean(1).min()
        if shift>max_corner_shift:return False,'excessive_feedback_drift'
    return True,'line_and_metric_guards_passed'
