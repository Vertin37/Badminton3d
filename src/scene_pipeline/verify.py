"""Validate contracts and frozen articulation before considering a run complete."""
from pathlib import Path
import hashlib
import av,numpy as np,torch
from .artifacts import read,write,digest

def verify(job):
    root=Path(job['output']);data=np.load(root/'scene/inputs/scene.npz');a=np.load(root/'scene/results/scene/alignment.npz')
    assert np.isfinite(a['joint_roots_world']).all() and np.isfinite(a['refined_cameras']).all()
    for raw in a['refined_cameras']:
        from .scene.align import camera
        r,_,f,center=camera(raw)
        assert np.allclose(r@r.T,np.eye(3),atol=1e-6) and np.isclose(np.linalg.det(r),1.)
        assert np.isfinite(f) and f>0 and center[2]>0
    assert data['local_joints_cam'].shape[:2]==(2,job['video_info']['frames'])
    checks=dict(finite_world_roots=True,finite_camera_parameters=True,proper_rotation_and_positive_camera_height=True,manual_points=False,court_network_used=False)
    provenance=read(root/'scene/inputs/scene.json');native={}
    for record in provenance['native_sources']:
        assert digest(record['parameters'])==record['sha256'],'Native parameters changed during alignment'
        d=torch.load(record['parameters'],map_location='cpu',weights_only=False)
        assert d['official_postproc'] and len(d['smpl_params_incam']['transl'])==job['video_info']['frames']
        native[str(record['person'])]=dict(parameters_sha256=record['sha256'],body_pose_sha256=hashlib.sha256(d['smpl_params_incam']['body_pose'].numpy().tobytes()).hexdigest())
    checks['native_articulation_preserved']=True
    checks['native_sources']=native
    metrics=read(root/'scene/results/scene/metrics.json');assert metrics['local_body_pose_changed'] is False
    checks['shots']=len(metrics['shots'])
    court=read(job['court_result']);assert court['court_network_used'] is False
    if job.get('render_requested',True):
        cap=av.open(str(root/'reconstruction.mp4'));count=sum(1 for _ in cap.decode(video=0));assert not cap.streams.audio;cap.close()
        expected=read(root/'video_verification.json')['output_frames'];assert count==expected
        checks['video_frames']=count;checks['fully_decoded']=True;checks['audio_streams']=0
    write(root/'verification.json',dict(all_passed=True,checks=checks))
