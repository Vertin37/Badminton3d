"""Shared camera and rigid root-trajectory fitting, with frozen GVHMR pose.

All rotations are proper SO(3) transformations. No per-limb fitting, learned
weights, world-axis reflection, or post-hoc floor clipping is used.
"""
import argparse,json,time
from dataclasses import dataclass,asdict
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d,percentile_filter,binary_opening,label
from scipy.sparse import coo_matrix,vstack,eye
from scipy.sparse.linalg import spsolve
from pathlib import Path
from .support import COURT,BODY_IDS,COCO_IDS,dump

@dataclass(frozen=True)
class Config:
    body_sigma_px:float=10.
    court_sigma_px:float=18.
    support_sigma_m:float=.035
    slip_step_sigma_m:float=.018
    root_prior_sigma_m:float=.80
    acceleration_sigma_m_s2:float=10.
    air_clearance_sigma_m:float=.035
    gravity_sigma_rad:float=.075
    max_camera_evaluations:int=38
    root_irls:int=3

def camera(raw):
    r=cv2.Rodrigues(raw[:3])[0];center=raw[3:6];f=np.exp(raw[6]);return r,-r@center,f,center
def project(points,r,t,f,pp):
    v=points@r.T+t;return v[...,:2]/np.maximum(v[...,2:3],.2)*f+pp
def project_camera(points,f,pp):return points[...,:2]/np.maximum(points[...,2:3],.5)*f+pp

def initialize(court_uv,confidence,focal,up):
    valid=confidence>.05;obj=COURT[valid].astype(float);uv=court_uv[valid].astype(float);pp=initialize.pp
    k=np.array([[focal,0,pp[0]],[0,focal,pp[1]],[0,0,1.]])
    ok,rv,tv=cv2.solvePnP(obj,uv,k,None,flags=cv2.SOLVEPNP_SQPNP)
    if not ok:raise RuntimeError('Court initialization failed')
    rr=cv2.Rodrigues(rv)[0];center=(-rr.T@tv).reshape(3)
    if center[2]<.3:
        ok,rv,tv,_=cv2.solvePnPRansac(obj,uv,k,None,reprojectionError=25,iterationsCount=200,flags=cv2.SOLVEPNP_EPNP);rr=cv2.Rodrigues(rv)[0];center=(-rr.T@tv).reshape(3)
    if center[2]<=0:raise RuntimeError('Court initialization has camera below floor; refusing an axis reflection.')
    return np.r_[rv.ravel(),center,np.log(focal)]

class System:
    def __init__(self,n):self.n=n;self.m=[];self.b=[]
    def add(self,columns,values,target,weight):
        target=np.asarray(target).reshape(-1);columns=np.asarray(columns).reshape(len(target),-1);values=np.broadcast_to(values,columns.shape);weight=np.broadcast_to(weight,target.shape).astype(float)
        scale=np.sqrt(np.maximum(weight,0));row=np.broadcast_to(np.arange(len(target))[:,None],columns.shape)
        self.m.append(coo_matrix(((values*scale[:,None]).ravel(),(row.ravel(),columns.ravel())),shape=(len(target),self.n)).tocsr());self.b.append(target*scale)
    def solve(self):
        a=vstack(self.m,format='csr');b=np.concatenate(self.b);return spsolve(a.T@a+eye(self.n)*1e-8,a.T@b)

def patches(vertices,group,normal):
    result=[]
    for k in range(4):
        v=vertices[:,:,group==k];mean=v.mean(2);h=np.quantile(v@normal,.10,axis=2);mean=mean+(h-mean@normal)[...,None]*normal
        result.append(mean)
    return np.stack(result,2)

def contacts(data,start,end,up):
    vertices=data['sole_vertices_local_cam'][:,start:end];sole=patches(vertices,data['sole_patch_group'],up);stat=data['static_probabilities'][:,start:end];fps=float(data['fps']);n=end-start
    q=np.zeros_like(stat,dtype=float);air=np.zeros((2,n),bool);clear=np.zeros((2,n));native=data['native_sole_heights'][:,start:end].min(-1)
    for p in [0,1]:
        ground=percentile_filter(native[p],15,size=min(61,n//2*2-1),mode='nearest');clear[p]=np.maximum(native[p]-ground,0)
        candidates=binary_opening(clear[p]>.075,structure=np.ones(3))
        torso=np.linalg.norm(data['uv'][p,start:end,5:7].mean(1)-data['uv'][p,start:end,11:13].mean(1),axis=-1).clip(15)
        for k,ankle in enumerate([15,15,16,16]):
            observed=gaussian_filter1d(data['uv'][p,start:end,ankle],1,axis=0);speed=np.linalg.norm(np.gradient(observed,axis=0),axis=-1)*fps/torso
            still=1/(1+(speed/1.5)**4);gap=(sole[p]@up)[:,k]-(sole[p]@up).min(1)
            q[p,:,k]=np.clip((stat[p,:,k]-.25)/.75,0,1)*np.sqrt(still)*np.exp(-(gap/.09)**2)*data['confidence'][p,start:end,ankle]
            q[p,:,k]=gaussian_filter1d(q[p,:,k],.8);q[p,:,k]*=binary_opening(q[p,:,k]>.22,structure=np.ones(3))
        # Native global height also contains drift. A long height excursion or
        # an interval with strong stationary-foot evidence is not a flight.
        # Gate BEFORE optimization, and freeze these scores across comparisons.
        components,total=label(candidates)
        for component in range(1,total+1):
            ids=np.flatnonzero(components==component)
            if len(ids)<=round(.9*fps) and np.median(q[p,ids].max(-1))<.45:air[p,ids]=True
        q[p,air[p]]=0
    return q,air,clear

def solve_roots(case,raw,cfg,physical=True):
    r,t,f,center=camera(raw);normal=r[:,2];local=case['local'];n=local.shape[1];pp=case['pp'];fps=case['fps']
    soles=patches(case['sole_vertices'],case['sole_group'],normal)
    native=case['native'].copy();native[...,2]*=f/case['native_focal'];roots=[]
    for p in [0,1]:
        current=native[p].copy();uv=case['uv'][p];conf=case['confidence'][p];xy=(uv-pp)/f
        ax=np.stack([np.ones_like(xy[...,0]),xy[...,0]*0,-xy[...,0]],-1);ay=np.stack([xy[...,1]*0,np.ones_like(xy[...,1]),-xy[...,1]],-1)
        bx=xy[...,0]*local[p,:,:,2]-local[p,:,:,0];by=xy[...,1]*local[p,:,:,2]-local[p,:,:,1]
        a=np.stack([ax,ay],2).reshape(n,-1,3);b=np.stack([bx,by],2).reshape(n,-1);columns=np.arange(n)[:,None,None]*3+np.arange(3)
        columns=np.broadcast_to(columns,a.shape).reshape(-1,3);base_obs=np.repeat(conf**2,2,axis=1)
        for iteration in range(cfg.root_irls):
            system=System(n*3);res=np.einsum('tki,ti->tk',a,current)-b
            pixel_scale=f/np.maximum(current[:,2],2);robust=np.minimum(1,25/np.maximum(np.abs(res)*pixel_scale[:,None],1e-6))
            weight=base_obs*(pixel_scale[:,None]/cfg.body_sigma_px)**2*robust
            system.add(columns,a.reshape(-1,3),b.ravel(),weight.ravel())
            system.add(np.arange(n*3)[:,None],np.ones((n*3,1)),native[p].ravel(),1/cfg.root_prior_sigma_m**2)
            if physical:
                q=case['contact'][p];h=soles[p]@normal
                for foot in range(4):
                    system.add(np.arange(n)[:,None]*3+np.arange(3),normal,np.dot(normal,t)-h[:,foot],q[:,foot]**2/cfg.support_sigma_m**2)
                    lock=np.minimum(q[1:,foot],q[:-1,foot])**2/cfg.slip_step_sigma_m**2
                    for axis in [0,1,2]:
                        cols=np.stack([np.arange(n-1)*3+axis,np.arange(1,n)*3+axis],-1)
                        system.add(cols,[-1,1],-np.diff(soles[p,:,foot,axis]),lock)
                for axis in [0,1,2]:
                    cols=np.stack([np.arange(n-2)*3+axis,np.arange(1,n-1)*3+axis,np.arange(2,n)*3+axis],-1)
                    sigma=cfg.acceleration_sigma_m_s2*(1.5 if axis==1 else 1.)/fps**2
                    system.add(cols,[1,-2,1],np.zeros(n-2),1/sigma**2)
                lower=np.quantile(case['sole_vertices'][p]@normal,.03,axis=1)
                system.add(np.arange(n)[:,None]*3+np.arange(3),normal,np.dot(normal,t)-lower+case['clearance'][p],case['air'][p]/cfg.air_clearance_sigma_m**2)
                penetrating=(current@normal-np.dot(normal,t)+lower)<-.01
                system.add(np.arange(n)[:,None]*3+np.arange(3),normal,np.dot(normal,t)-lower-.005,penetrating/.02**2)
            current=system.solve().reshape(n,3)
        roots.append(current)
    return np.stack(roots),soles

def residual(raw,case,cfg,initial):
    roots,soles=solve_roots(case,raw,cfg);r,t,f,c=camera(raw);n=roots.shape[1];normal=r[:,2]
    b=(project_camera(case['local']+roots[:,:,None],f,case['pp'])-case['uv'])/cfg.body_sigma_px
    b*=case['confidence'][...,None];b*=np.sqrt(20/n)
    height=(roots[:,:,None]+soles)@normal-np.dot(normal,t);contact=height*case['contact']/cfg.support_sigma_m*np.sqrt(8/n)
    delta=np.diff(roots[:,:,None]+soles,axis=1);lock=np.minimum(case['contact'][:,1:],case['contact'][:,:-1]);slip=delta*lock[...,None]/cfg.slip_step_sigma_m*np.sqrt(3/n)
    projected=project(COURT,r,t,f,case['pp']);weights=np.sqrt(case['court_confidence']);weights[:30]*=.5
    court=(projected-case['court_uv'])*weights[:,None]/cfg.court_sigma_px
    gravity=(normal-case['up'])/cfg.gravity_sigma_rad
    scene_prior=np.r_[(c-initial[3:6])/[5.,5.,3.],(raw[6]-initial[6])/.40]
    world=(roots-t)@r;bounds=np.maximum(np.abs(world[...,:2])-[4.5,8.2],0)/.75*np.sqrt(3/n)
    return np.concatenate([b.ravel(),contact.ravel(),slip.ravel(),court.ravel(),gravity,scene_prior,bounds.ravel()])

def make_case(data,start,end):
    up=data['up_cam'][:,start:end].reshape(-1,3);up=np.median(up,axis=0);up/=np.linalg.norm(up)
    q,air,clear=contacts(data,start,end,up)
    return dict(local=data['local_joints_cam'][:,start:end,BODY_IDS].astype(float),native=data['native_roots_cam'][:,start:end].astype(float),native_focal=float(np.median(data['native_focal'][:,start:end])),
        uv=data['uv'][:,start:end,COCO_IDS].astype(float),confidence=np.clip(data['confidence'][:,start:end,COCO_IDS],0,1).astype(float),pp=np.array([float(data['width'])/2,float(data['height'])/2]),fps=float(data['fps']),
        sole_vertices=data['sole_vertices_local_cam'][:,start:end].astype(float),sole_group=data['sole_patch_group'],up=up,contact=q,air=air,clearance=clear,
        court_focal=float(np.median(data['court_focal'][start:end])) if 'court_focal' in data else float(np.median(data['native_focal'][:,start:end])),court_uv=np.median(data['court_uv'][start:end],axis=0),court_confidence=np.median(data['court_confidence'][start:end],axis=0))

def score(case,raw,roots):
    r,t,f,c=camera(raw);normal=r[:,2];soles=patches(case['sole_vertices'],case['sole_group'],normal);n=roots.shape[1]
    error=np.linalg.norm(project_camera(case['local']+roots[:,:,None],f,case['pp'])-case['uv'],axis=-1);valid=case['confidence']>.3
    floor_vertices=(case['sole_vertices']+roots[:,:,None])@normal-np.dot(normal,t);floor=np.min(floor_vertices,axis=2)
    q=case['contact'];height=(roots[:,:,None]+soles)@normal-np.dot(normal,t);mask=q>.4;support=np.minimum(q[:,1:],q[:,:-1])>.4
    velocity=np.linalg.norm(np.diff(roots[:,:,None]+soles,axis=1),axis=-1)*case['fps'];world=(roots-t)@r
    court_error=np.linalg.norm(project(COURT,r,t,f,case['pp'])-case['court_uv'],axis=-1);cv=case['court_confidence']>.05
    air_errors=np.abs(floor-case['clearance']);air=case['air']
    return dict(body12_reprojection_mean_px=float(error[valid].mean()),court_reprojection_mean_px=float(court_error[cv].mean()),
        feet_below_minus2cm_fraction=float((floor<-.02).mean()),mean_penetration_depth_m=float(np.maximum(-floor,0).mean()),
        predicted_support_height_error_m=float(np.abs(height[mask]).mean()) if mask.any() else None,
        predicted_support_slip_m_s=float(velocity[support].mean()) if support.any() else None,predicted_support_pairs=int(support.sum()),
        root_acceleration_rms_m_s2=float(np.sqrt(np.mean(np.sum((np.diff(world,n=2,axis=1)*case['fps']**2)**2,axis=-1)))),
        native_air_frames=int(air.sum()),native_air_clearance_error_m=float(air_errors[air].mean()) if air.any() else None,
        camera_center_m=c.tolist(),camera_focal_px=float(f),gravity_difference_deg=float(np.rad2deg(np.arccos(np.clip(normal@case['up'],-1,1)))),
        world_root_min_m=world.min((0,1)).tolist(),world_root_max_m=world.max((0,1)).tolist())

def run(name,cfg,workspace):
    ROOT=Path(workspace)
    data=np.load(ROOT/'inputs'/f'{name}.npz');n=data['uv'].shape[1];cuts=list(data['cut_frames'])+[n];out=ROOT/'results'/name;out.mkdir(exist_ok=True)
    baseline_root=np.zeros((2,n,3));fixed_root=np.zeros_like(baseline_root);joint_root=np.zeros_like(baseline_root);before_cam=np.zeros((n,7));after_cam=np.zeros_like(before_cam);contacts_all=np.zeros((2,n,4));air_all=np.zeros((2,n),bool);reports=[];started=time.time()
    for shot,(start,end) in enumerate(zip(cuts[:-1],cuts[1:])):
        case=make_case(data,start,end);initialize.pp=case['pp'];initial=initialize(case['court_uv'],case['court_confidence'],case['court_focal'],case['up'])
        baseline=case['native'];fixed,_=solve_roots(case,initial,cfg);print(name,'shot',shot,'initial',score(case,initial,baseline),flush=True)
        bounds_low=np.r_[[-6]*3,[-40,-40,.5],initial[6]-.65];bounds_high=np.r_[[6]*3,[40,40,18],initial[6]+.65]
        fit=least_squares(residual,initial,args=(case,cfg,initial),bounds=(bounds_low,bounds_high),x_scale=np.r_[[.1]*3,[2,2,1],.2],loss='soft_l1',f_scale=1.,max_nfev=cfg.max_camera_evaluations,ftol=2e-5,xtol=2e-5,gtol=2e-5)
        refined=fit.x;roots,_=solve_roots(case,refined,cfg);r0,t0,_,_=camera(initial);r,t,_,_=camera(refined)
        baseline_root[:,start:end]=(baseline-t0)@r0;fixed_root[:,start:end]=(fixed-t0)@r0;joint_root[:,start:end]=(roots-t)@r
        before_cam[start:end]=initial;after_cam[start:end]=refined;contacts_all[:,start:end]=case['contact'];air_all[:,start:end]=case['air']
        report=dict(shot=shot,start_frame=int(start),end_frame=int(end),baseline=score(case,initial,baseline),fixed_camera_contact=score(case,initial,fixed),joint_scene_contact=score(case,refined,roots),camera_evaluations=int(fit.nfev),solver_success=bool(fit.success),solver_message=str(fit.message))
        reports.append(report);dump(out/'progress.json',dict(status='running',completed_shots=len(reports),shots=reports));print(name,'shot',shot,'refined',report['joint_scene_contact'],flush=True)
    np.savez_compressed(out/'alignment.npz',baseline_roots_world=baseline_root.astype(np.float32),fixed_camera_roots_world=fixed_root.astype(np.float32),joint_roots_world=joint_root.astype(np.float32),initial_cameras=before_cam,refined_cameras=after_cam,contact_scores=contacts_all,native_air=air_all,cut_frames=data['cut_frames'],fps=data['fps'])
    result=dict(video=name,config=asdict(cfg),seconds=time.time()-started,shots=reports,local_body_pose_changed=False,betas_changed=False,learned_weights_used=False,physical_metrics_are_proxies=True,independent_real_3d_ground_truth=False)
    dump(out/'metrics.json',result);print('FINISHED',name,round(result['seconds'],2),'seconds',flush=True)
