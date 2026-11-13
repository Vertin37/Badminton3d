"""Assemble explicit native/court contracts, then rigid scene alignment."""
import sys,json
from pathlib import Path
import cv2,numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
from .support import COURT,template,dump,digest
from ..artifacts import read
from ..fusion import trusted

def build(job):
    ROOT=Path(job["output"]);SCENE=ROOT/"scene";NAME="scene"
    SCENE.mkdir(exist_ok=True);(SCENE/"inputs").mkdir(exist_ok=True);(SCENE/"results").mkdir(exist_ok=True)
    ASSET=Path(job["neutral_model"])
    output=SCENE/'inputs'/f'{NAME}.npz'
    cap=cv2.VideoCapture(str(Path(job['video'])));n=int(cap.get(7));width=int(cap.get(3));height=int(cap.get(4));fps=cap.get(5);cap.release()
    field=read(job['court_result'])
    samples=field['frames'];hs=[]
    if not all(trusted(g['calibration']) for g in field['groups']):raise RuntimeError('Court geometry is unsupported; refusing silent 3-D alignment')
    for i in range(n):
        nearest=min(samples,key=lambda r:abs(r['source_frame']-i));h=np.array(nearest['homography_metric_to_original'])
        q=np.c_[np.array([[-3.05,-6.7],[3.05,-6.7],[3.05,6.7],[-3.05,6.7]]),np.ones(4)]@h.T;q=q[:,:2]/q[:,2:]
        winding=np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))
        if winding>0:h=h@np.diag([-1.,1,1])
        hs.append(h)
    world=COURT[:,:2];court_uv=[]
    for h in hs:
        q=np.c_[world,np.ones(len(world))]@h.T;court_uv.append(q[:,:2]/q[:,2:])
    court_uv=np.array(court_uv);court_conf=np.ones((n,35),np.float32)*.7;court_conf[:,32:]=0
    support=min(g['calibration']['selected']['diagnostics']['forward_line_support'] for g in field['groups'])
    court_conf[:,:32]*=min(1.,support/.8)
    # No detected 3-D net-height landmarks are invented from the plane warp.
    court_uv[:,32:]=0
    patches,cluster,counts,faces=template(ASSET);sole_ids=np.concatenate(patches)
    sole_group=np.concatenate([np.full(len(p),i) for i,p in enumerate(patches)])
    model=smplx.create(str(ASSET),model_type='smplx',gender='neutral',num_pca_comps=12,flat_hand_mean=False).eval()
    values={key:[] for key in ['local_mesh_cam','local_joints_cam','native_roots_cam','sole_vertices_local_cam','up_cam','native_sole_heights','native_global_root_height','static_probabilities','native_focal','uv','confidence']}
    sources=[]
    for person in (0,1):
        source=ROOT/f'native/p{person}/1_hmr4d_results_native.pt';d=torch.load(source,map_location='cpu',weights_only=False)
        params=d['smpl_params_incam'];global_params=d['smpl_params_global'];pieces={key:[] for key in ['local_mesh_cam','local_joints_cam','native_roots_cam','sole_vertices_local_cam','native_sole_heights','native_global_root_height']}
        for first in range(0,n,48):
            last=min(n,first+48);b=last-first
            extra=dict(left_hand_pose=torch.zeros(b,12),right_hand_pose=torch.zeros(b,12),expression=torch.zeros(b,10),jaw_pose=torch.zeros(b,3),leye_pose=torch.zeros(b,3),reye_pose=torch.zeros(b,3))
            with torch.no_grad():
                incam=model(**{k:v[first:last] for k,v in params.items()},**extra)
                glob=model(**{k:v[first:last] for k,v in global_params.items()},**extra)
            vertices=incam.vertices.numpy();joints=incam.joints[:,:22].numpy();pelvis=joints[:,0];local=vertices-pelvis[:,None]
            reduced=np.stack([np.stack([np.bincount(cluster,weights=frame[:,axis])/counts for axis in range(3)],-1) for frame in local])
            for key,array in [('local_mesh_cam',reduced),('local_joints_cam',joints-pelvis[:,None]),('native_roots_cam',pelvis),
                ('sole_vertices_local_cam',local[:,sole_ids]),('native_sole_heights',glob.vertices.numpy()[:,sole_ids,1]),('native_global_root_height',glob.joints.numpy()[:,0,1])]:pieces[key].append(array.astype(np.float32))
        for key,parts in pieces.items():values[key].append(np.concatenate(parts))
        rc=Rotation.from_rotvec(params['global_orient'].numpy()).as_matrix();rg=Rotation.from_rotvec(global_params['global_orient'].numpy()).as_matrix()
        values['up_cam'].append(np.einsum('tij,j->ti',rc@rg.transpose(0,2,1),[0.,1,0]).astype(np.float32))
        values['static_probabilities'].append(d['static_confidence'][:,:4].numpy());values['native_focal'].append(d['K_fullimg'][:,0,0].numpy())
        kp=torch.load(ROOT/f'native_cache/p{person}'/Path(job['video']).stem/'preprocess/vitpose.pt',map_location='cpu').numpy()
        values['uv'].append(kp[:,:,:2]);values['confidence'].append(kp[:,:,2])
        sources.append({'person':person,'parameters':str(source),'sha256':digest(source)})
        print('CPU SMPL-X geometry',person,'prepared',flush=True)
    cuts=json.loads((ROOT/'cuts.json').read_text())['cuts']
    # Camera changes found by the frozen court module are also scene boundaries.
    cuts=sorted(set(cuts+[g['first_source_frame'] for g in field['groups']]))
    np.savez_compressed(output,**{k:np.stack(v) for k,v in values.items()},court_uv=court_uv.astype(np.float32),court_confidence=court_conf,
        court_focal=np.full(n,np.median([g['calibration']['selected']['diagnostics']['focal_length_px_proxy'] for g in field['groups']])/min(960/width,540/height)),fps=fps,width=width,height=height,cut_frames=np.array(cuts),faces=faces,sole_patch_group=sole_group,source_mapping=np.tile([0,1],(n,1)))
    dump(output.with_suffix('.json'),{'source_video':str(Path(job['video'])),'native_sources':sources,'court_source':str(job['court_result']),
        'court_model_used':False,'manual_court_points_or_cuts':False,'court_parameters_retuned':False,'native_articulation_unchanged':True,'cuts':cuts,
        'world_axis_gauge':'X sign canonicalized for proper positive-height camera; no body reflection',
        'net_height_points_not_observed':True})


def run(job):
    torch.set_num_threads(4);cv2.setNumThreads(1)
    build(job)
    from .align import run as align,Config
    scene=Path(job['output'])/'scene'
    align('scene',Config(),scene)
    if job['config']['render']['avatar']=='male_slim':
        from .avatar import prepare_avatar
        betas=read(Path(__file__).resolve().parents[3]/'configs/male_slim.json')['betas']
        prepare_avatar('scene',scene,Path(job['output'])/'native',job['neutral_model'],job['male_model'],betas)
