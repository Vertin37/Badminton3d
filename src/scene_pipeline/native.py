"""Streamed official HMR2/GVHMR inference; no dataset-specific frame ranges."""
import sys,json,shutil,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from .artifacts import read,write,digest,video_info,key as cache_key

def cuts_from_tracks(kps,selection_path):
    n=len(kps[0]);scores=np.zeros(n)
    for kp in kps:
        for i in range(1,n):
            ids=np.array([5,6,11,12,13,14,15,16]);good=(kp[i-1,ids,2]>.3)&(kp[i,ids,2]>.3)
            if good.sum()<4:continue
            points=kp[i,ids[good],:2];previous=kp[i-1,ids[good],:2]
            allgood=kp[i,:,2]>.3;scale=np.ptp(kp[i,allgood,1]) if allgood.any() else 100.
            scores[i]=max(scores[i],float(np.median(np.linalg.norm(points-previous,axis=1)))/max(scale,80.))
    peaks=[]
    for i in np.argsort(scores)[::-1]:
        if scores[i]<.28:break
        if i>=12 and n-i>=12 and all(abs(int(i)-j)>15 for j in peaks):peaks.append(int(i))
    selection=read(selection_path)
    cuts=list(selection['cut_frames'])
    for cut in peaks:
        if all(abs(cut-other)>12 for other in cuts):cuts.append(cut)
    cuts=sorted(cut for cut in set(cuts) if cut==0 or 12<=cut<=n-12)
    return cuts

def import_reference(job,data,cuts,n):
    source=job.get('native_reference')
    if not source:return False
    source=Path(source);v=read(source/'verification.json')
    if v['source_clip']['sha256']!=digest(job['video']):raise ValueError('Native reference video hash mismatch')
    for person,d in enumerate(data):
        folder=source/f'03_cache/p{person}'/Path(job['video']).stem/'preprocess'
        # Historical clips may have a different basename; metadata verifies
        # bytes, and the unique cache directory supplies the actual stem.
        if not folder.exists():
            candidates=list((source/f'03_cache/p{person}').glob('*/preprocess'))
            if len(candidates)!=1:raise ValueError('Ambiguous reference cache')
            folder=candidates[0]
        kp=torch.load(folder/'vitpose.pt',map_location='cpu');boxes=torch.load(folder/'bbx.pt',map_location='cpu')['bbx_xys']
        if not torch.equal(kp,d['kp']) or not torch.equal(boxes,d['boxes']):
            print('Reference inputs changed; running fresh native inference',flush=True);return False
        p=source/f'04_native/p{person}/1_hmr4d_results_native.pt'
        if digest(p)!=v['native_parameters'][str(person)]['sha256']:raise ValueError('Reference parameters changed')
        native=torch.load(p,map_location='cpu',weights_only=False)
        if list(native['cut_frames'])!=cuts or len(native['smpl_params_incam']['transl'])!=n:return False
    for person in (0,1):
        out=Path(job['output'])/f'native/p{person}';out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/f'04_native/p{person}/1_hmr4d_results_native.pt',out/'1_hmr4d_results_native.pt')
    write(Path(job['output'])/'native_import.json',dict(source=str(source),video_hash_verified=True,keypoints_and_boxes_exact=True,cuts_exact=True,parameter_hashes_verified=True))
    return True

def run(job):
    torch.set_num_threads(4);cv2.setNumThreads(1)
    ROOT=Path(job["output"]);VIDEO=Path(job["video"]);CSV=Path(job["selected_csv"])
    gv=Path(job["gvhmr_root"]);repo=Path(__file__).resolve().parents[2]
    sys.dont_write_bytecode=True;sys.path[:0]=[str(gv),str(repo/"work")]
    import rtmlib_to_gvhmr_adapter as adapter
    from hmr4d.network.hmr2.utils.preproc import crop_and_resize,IMAGE_MEAN,IMAGE_STD
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor
    cap=cv2.VideoCapture(str(VIDEO));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(3));height=int(cap.get(4));cap.release()
    data=[]
    for person in (0,1):
        parent=ROOT/f'native_cache/p{person}';path=parent/VIDEO.stem/'preprocess'
        adapter.build_adapter(CSV,VIDEO,VIDEO,parent,None,person,0,n,False,False,job['config']['native']['focal_mm'])
        data.append({'kp':torch.load(path/'vitpose.pt',map_location='cpu'),'boxes':torch.load(path/'bbx.pt',map_location='cpu')['bbx_xys'],'path':path})
    cuts=cuts_from_tracks([d['kp'].numpy() for d in data],job['selection_json']);write(ROOT/'cuts.json',dict(cuts=cuts,manual_cuts=False));print('automatic cuts',cuts,flush=True)
    if import_reference(job,data,cuts,n):return
    if not torch.cuda.is_available():raise RuntimeError('Fresh GVHMR/HMR2 inference requires CUDA; geometry and scene alignment use CPU')
    for d in data:
        signature=cache_key(dict(video=digest(VIDEO),boxes=hashlib.sha256(d['boxes'].numpy().tobytes()).hexdigest(),
            model=digest(gv/'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt'),extractor_code=digest(__file__)))
        metadata=d['path']/'feature_signature.json'
        if not metadata.exists() or read(metadata).get('signature')!=signature:
            (d['path']/'vit_features.pt').unlink(missing_ok=True)
        d['feature_signature']=signature
    missing=[p for p,d in enumerate(data) if not (d['path']/'vit_features.pt').exists()]
    if missing:
        from hmr4d.network.hmr2 import load_hmr2
        model=load_hmr2().cuda().eval()
        for person in missing:
            d=data[person];cap=cv2.VideoCapture(str(VIDEO));features=[];batch=[]
            def flush():
                if not batch:return
                images=torch.from_numpy(np.stack(batch));images=((images/255.-IMAGE_MEAN)/IMAGE_STD).permute(0,3,1,2).cuda()
                with torch.inference_mode():result=model({'img':images})
                features.append(result.cpu());batch.clear()
            for i in range(n):
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Feature video decode failed')
                image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);image=cv2.resize(image,(width//2,height//2))
                center=d['boxes'][i,:2]*.5;size=d['boxes'][i,2]*.5;factor=float(size)/256/2
                if factor>1.1:image=cv2.GaussianBlur(image,(5,5),(factor-1)/2)
                crop,_=crop_and_resize(image,center,size,256,enlarge_ratio=1.0);batch.append(crop)
                if len(batch)==job['config']['native']['feature_batch_size']:flush()
                if (i+1)%200==0:print('HMR2 person',person,i+1,'/',n,flush=True)
            flush();cap.release();torch.save(torch.cat(features),d['path']/'vit_features.pt')
            write(d['path']/'feature_signature.json',dict(signature=d['feature_signature']))
        del model;torch.cuda.empty_cache()
    from hydra import compose,initialize_config_module
    import hydra
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # registers the demo model with Hydra
    with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
        register_store_gvhmr();cfg=compose(config_name='demo',overrides=['video_name=integrated_scene'])
    model=hydra.utils.instantiate(cfg.model,_recursive_=False)
    model.load_pretrained_model(str(gv/'inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt'));model=model.eval().cuda()
    for person,d in enumerate(data):
        output=ROOT/f'native/p{person}';output.mkdir(parents=True,exist_ok=True)
        resultfile=output/'1_hmr4d_results_native.pt'
        features=torch.load(d['path']/'vit_features.pt',map_location='cpu');params={};statics=[]
        for start,end in zip(cuts,cuts[1:]+[n]):
            k=create_camera_sensor(width,height,job['config']['native']['focal_mm'])[2].repeat(end-start,1,1)
            inputs={'length':torch.tensor(end-start),'bbx_xys':d['boxes'][start:end].float(),'kp2d':d['kp'][start:end].float(),
                'K_fullimg':k.float(),'cam_angvel':torch.zeros(end-start,6),'f_imgseq':features[start:end].float()}
            with torch.inference_mode():prediction=model.predict(inputs,static_cam=True)
            for kind in ['smpl_params_global','smpl_params_incam']:
                params.setdefault(kind,{})
                for key,v in prediction[kind].items():params[kind].setdefault(key,[]).append(v.cpu())
            statics.append(torch.sigmoid(prediction['net_outputs']['static_conf_logits'][0]).cpu())
            print('GVHMR person',person,'frames',start,end,flush=True)
        joined={kind:{key:torch.cat(parts) for key,parts in values.items()} for kind,values in params.items()}
        joined.update(K_fullimg=create_camera_sensor(width,height,job['config']['native']['focal_mm'])[2].repeat(n,1,1),static_confidence=torch.cat(statics),
            cut_frames=cuts,official_postproc=True,model_weights_updated=False,method='official DemoPL.predict postproc=True per automatically split sequence')
        torch.save(joined,resultfile)
    print('Native inference complete',flush=True)
