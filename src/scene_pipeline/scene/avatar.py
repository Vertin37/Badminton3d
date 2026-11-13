import json,hashlib
from pathlib import Path
import numpy as np,torch,smplx
from .support import template,digest,dump
def model(asset,gender):
    return smplx.create(str(asset),model_type='smplx',gender=gender,use_pca=False,flat_hand_mean=True).eval()

def prepare_avatar(name,workspace,native_root,neutral_asset,male_asset,betas):
    ROOT=Path(workspace);ASSET=Path(neutral_asset);MALE=Path(male_asset)
    CONFIG=ROOT/"avatar_config.json"
    config=dict(betas=betas,asset_sha256=digest(MALE),display_only=True)
    dump(CONFIG,config)
    native_dir=lambda name,p:Path(native_root)/f"p{p}"
    output=ROOT/'inputs'/f'{name}_male_slim.npz';metadata=output.with_suffix('.json');config_hash=digest(CONFIG)
    # The orchestrator owns content-validated caching. A requested scene rebuild
    # also rebuilds this skin, because its motion, mapping or model may change.
    data=np.load(ROOT/'inputs'/f'{name}.npz');n=data['local_joints_cam'].shape[1];mapping=data['source_mapping'];neutral=model(ASSET,'neutral');male=model(MALE,'male')
    assert np.array_equal(neutral.faces,male.faces)
    z=np.load(ASSET,allow_pickle=True);left=z['hands_meanl'].astype(np.float32);right=z['hands_meanr'].astype(np.float32)
    weights=male.lbs_weights.detach();betas=torch.tensor(config['betas'],dtype=torch.float32)[None]
    rest=male.v_template.detach().numpy()+np.einsum('vck,k->vc',male.shapedirs.detach().numpy()[...,:10],np.array(config['betas']))
    _,cluster=np.unique(np.floor(rest/.032).astype(int),axis=0,return_inverse=True);counts=np.bincount(cluster);faces=cluster[male.faces]
    faces=faces[(faces[:,0]!=faces[:,1])&(faces[:,0]!=faces[:,2])&(faces[:,1]!=faces[:,2])];_,ix=np.unique(np.sort(faces,axis=1),axis=0,return_index=True);faces=faces[np.sort(ix)]
    group,_,_,_=template(ASSET);sole_ids=np.concatenate(group);meshes=[];sole_meshes=[];max_original_joint_error=0.;source_hashes=[]
    # Source-person order first, then reorder to the same per-frame slots used
    # by alignment. Neither recovered shape nor root position is overwritten.
    raw_native_joints=np.zeros((2,n,22,3),np.float32)
    for person in [0,1]:
        src=native_dir(name,person)/'1_hmr4d_results_native.pt';original=torch.load(src,map_location='cpu',weights_only=False)['smpl_params_incam'];mesh=[];soles=[]
        for first in range(0,n,48):
            last=min(n,first+48);b=last-first;batch={k:v[first:last] for k,v in original.items()}
            extra=dict(left_hand_pose=torch.from_numpy(left)[None].expand(b,-1),right_hand_pose=torch.from_numpy(right)[None].expand(b,-1),expression=torch.zeros(b,10),jaw_pose=torch.zeros(b,3),leye_pose=torch.zeros(b,3),reye_pose=torch.zeros(b,3))
            with torch.no_grad():
                native=neutral(**batch,**extra);male_batch={**batch,'betas':betas.expand(b,-1)};display=male(**male_batch,**extra)
                native_joint=native.joints[:,:55]-native.joints[:,:1];male_joint=display.joints[:,:55]-display.joints[:,:1]
                vertices=display.vertices-display.joints[:,:1]+torch.einsum('vj,bjk->bvk',weights,native_joint-male_joint)
            native_local=native_joint[:,:22].numpy();raw_native_joints[person,first:last]=native_local
            local=vertices.numpy();reduced=np.stack([np.stack([np.bincount(cluster,weights=frame[:,axis])/counts for axis in range(3)],-1) for frame in local])
            mesh.append(reduced.astype(np.float32));soles.append(local[:,sole_ids].astype(np.float32))
        meshes.append(np.concatenate(mesh));sole_meshes.append(np.concatenate(soles));source_hashes.append(dict(person=person,parameters=str(src),sha256=digest(src)))
        print(name,'male avatar source person',person,'prepared',flush=True)
    raw=np.stack(meshes);local=np.stack([raw[mapping[:,p],np.arange(n)] for p in [0,1]])
    raw_sole=np.stack(sole_meshes);sole=np.stack([raw_sole[mapping[:,p],np.arange(n)] for p in [0,1]])
    recovered=np.stack([raw_native_joints[mapping[:,p],np.arange(n)] for p in [0,1]]);max_original_joint_error=float(np.abs(recovered-data['local_joints_cam']).max());assert max_original_joint_error<3e-6,max_original_joint_error
    from .align import camera
    alignment=np.load(ROOT/'results'/name/'alignment.npz');original_sole=data['sole_vertices_local_cam'];heights=[]
    for frame in range(n):
        r,_,_,_=camera(alignment['refined_cameras'][frame]);normal=r[:,2];heights.extend(((sole[:,frame]@normal).min(-1)-(original_sole[:,frame]@normal).min(-1)).tolist())
    np.savez_compressed(output,local_mesh_cam=local,faces=faces,avatar_name='male_slim',display_only=True)
    dump(metadata,dict(avatar_name='male_slim',avatar_config=str(CONFIG),avatar_config_sha256=config_hash,asset_sha256=config['asset_sha256'],display_only=True,
        local_joint_pose_unchanged=True,original_recovered_joint_max_difference_m=max_original_joint_error,source_parameter_files=source_hashes,
        alignment_sha256=digest(ROOT/'results'/name/'alignment.npz'),inference_input_sha256=digest(ROOT/'inputs'/f'{name}.npz'),
        lowest_shoe_surface_display_change_m=dict(median=float(np.median(heights)),p05=float(np.quantile(heights,.05)),p95=float(np.quantile(heights,.95)),max_abs=float(np.max(np.abs(heights)))),
        note='Aesthetic avatar skin only. Original motion, skeleton, camera, root trajectory and quantitative alignment metrics are retained.'))
    print(output,flush=True)
