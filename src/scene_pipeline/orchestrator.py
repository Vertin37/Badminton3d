"""One entry point with explicit contracts and bounded two-way feedback."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
from .artifacts import digest,key,read,write,video_info,cached,finish
from .fusion import accept_feedback,trusted

REPO=Path(__file__).resolve().parents[2]
def parse(argv=None):
    p=argparse.ArgumentParser(description='Two players + geometric court + frozen GVHMR + rigid scene alignment')
    p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=REPO/'configs/scene_pipeline.default.json')
    p.add_argument('--pose-python',type=Path,default=Path(sys.executable));p.add_argument('--hmr-python',type=Path,default=Path(sys.executable))
    p.add_argument('--gvhmr-root',type=Path,default=REPO/'work/GVHMR');p.add_argument('--neutral-model',type=Path);p.add_argument('--male-model',type=Path)
    p.add_argument('--detections',type=Path,help='Optional existing raw RTMLib CSV for the exact same video')
    p.add_argument('--native-reference',type=Path,help='Optional previous experiment with verified video, keypoints, boxes, cuts and parameter hashes')
    p.add_argument('--resume',action='store_true');p.add_argument('--court-only',action='store_true');p.add_argument('--no-render',action='store_true')
    args=p.parse_args(argv);args.video=args.video.resolve();args.output=args.output.resolve();args.gvhmr_root=args.gvhmr_root.resolve()
    if args.output==args.video.parent:p.error('Output must be a dedicated run directory')
    return args

def run(args):
    config=read(args.config)
    if config['fusion']['max_passes']!=2:raise ValueError('This version supports exactly two bounded passes')
    if config['native']['feature_batch_size']<1 or config['render']['fps']<=0:raise ValueError('Invalid batch size/FPS')
    info=video_info(args.video);args.output.mkdir(parents=True,exist_ok=True)
    job=dict(video=str(args.video),video_sha256=digest(args.video),video_info=info,output=str(args.output),config=config,
        gvhmr_root=str(args.gvhmr_root),neutral_model=str((args.neutral_model or args.gvhmr_root/'inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz').resolve()),
        male_model=str((args.male_model or args.gvhmr_root/'inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz').resolve()),
        native_reference=str(args.native_reference.resolve()) if args.native_reference else None,
        render_requested=not args.no_render and not args.court_only)
    if not args.court_only:
        for asset in [job['neutral_model']]+([job['male_model']] if config['render']['avatar']=='male_slim' else []):
            if not Path(asset).is_file():raise FileNotFoundError('Licensed body asset required: '+asset)
    started=time.time();status=dict(status='running',version=config['version'],video_sha256=job['video_sha256'],stages={})
    old_job=args.output/'job.json'
    if old_job.exists() and read(old_job)['video_sha256']!=job['video_sha256']:raise ValueError('Run directory belongs to different video bytes')
    write(old_job,job)
    def stage(name,worker,python,inputs,outputs,modules,fields=None):
        stage_job={**job,**(fields or {})};stage_path=args.output/f'job_{name}.json';write(stage_path,stage_job)
        sources={str(REPO/f):digest(REPO/f) for f in modules+['src/scene_pipeline/worker.py','src/scene_pipeline/artifacts.py']}
        signature=key(dict(video=job['video_sha256'],config=config,inputs={str(p):digest(p) for p in inputs},sources=sources,worker=worker,
            outputs=[str(p) for p in outputs]))
        record=args.output/f'stage_{name}.json'
        if args.resume and cached(record,signature):
            status['stages'][name]=dict(status='cache_hit',signature=signature);write(args.output/'pipeline_status.json',status);return
        status['current_stage']=name;status['stages'][name]=dict(status='running',signature=signature);write(args.output/'pipeline_status.json',status)
        command=[str(python),'-B','-u','-X','utf8',str(REPO/'run_scene_pipeline.py'),'--worker',worker,'--job',str(stage_path)]
        print('[RUN]',name,flush=True)
        with (args.output/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if result.returncode:
            status['status']='failed';status['stages'][name]['status']='failed';write(args.output/'pipeline_status.json',status)
            raise RuntimeError(f'{name} failed; see {args.output/(name+".log")}')
        finish(record,signature,outputs);status['stages'][name]=dict(status='completed',signature=signature);write(args.output/'pipeline_status.json',status)
    if args.detections:
        detections=args.detections.resolve()
        if not detections.is_file():raise FileNotFoundError(detections)
    else:
        detections=args.output/'detections/pose_data_raw.csv'
        stage('01_detection','detect',args.pose_python,[],[detections],['run_pose_test.py'],dict(detection_output=str(args.output/'detections')))
    job['detections']=str(detections)
    def selected(folder):return [args.output/folder/f for f in ['pose_data_tracked.csv','pose_data_stable.csv','selection.json']]
    selection_sources=['src/scene_pipeline/selection.py','src/scene_pipeline/fusion.py','run_pose_test.py','src/pose_biomechanics/person_tracker.py','src/pose_biomechanics/temporal_filter.py']
    stage('02_people_initial','select',sys.executable,[detections],selected('people_initial'),selection_sources,dict(selection_output=str(args.output/'people_initial'),court_reference=None))
    initial_csv=selected('people_initial')[1];initial_court=args.output/'court_initial/scene/result.json'
    geom_sources=[str(p.relative_to(REPO)) for p in (REPO/'src/scene_pipeline/geometry').glob('*.py')]
    stage('03_court_initial','court',sys.executable,[initial_csv],[initial_court],geom_sources,dict(selected_csv=str(initial_csv),court_output=str(args.output/'court_initial')))
    stage('04_people_feedback','select',sys.executable,[detections,initial_court],selected('people_feedback'),selection_sources,
        dict(selection_output=str(args.output/'people_feedback'),court_reference=str(initial_court)))
    feedback_csv=selected('people_feedback')[1]
    if digest(initial_csv)==digest(feedback_csv):
        accepted=True;reason='tracks_unchanged_no_court_refit_needed';final_csv=initial_csv;final_court=initial_court;final_selection=selected('people_initial')[2]
    else:
        feedback_court=args.output/'court_feedback/scene/result.json'
        stage('05_court_feedback','court',sys.executable,[feedback_csv],[feedback_court],geom_sources,dict(selected_csv=str(feedback_csv),court_output=str(args.output/'court_feedback')))
        accepted,reason=accept_feedback(read(initial_court),read(feedback_court),config['fusion']['max_corner_shift_px_at960'])
        final_csv=feedback_csv if accepted else initial_csv;final_court=feedback_court if accepted else initial_court
        final_selection=selected('people_feedback' if accepted else 'people_initial')[2]
    # Native input uses unfiltered coordinates; confidence remains an observation
    # mask. Stable coordinates are only the court's supplied temporal hints.
    job.update(selected_csv=str(Path(final_csv).with_name('pose_data_tracked.csv')),selection_json=str(final_selection),court_result=str(final_court))
    write(args.output/'fusion.json',dict(accepted=accepted,reason=reason,passes=2,selected_tracks=str(final_csv),selected_court=str(final_court),
        feedback=read(selected('people_feedback')[2]),manual_points=False,config_retuned_per_video=False))
    write(old_job,job)
    if not args.court_only:
        if not all(trusted(g['calibration']) for g in read(final_court)['groups']):
            status.update(status='failed',current_stage='court_quality_gate');write(args.output/'pipeline_status.json',status)
            raise RuntimeError('Court is not sufficiently supported; review court output before any native GPU inference')
        neutral=Path(job['neutral_model']);male=Path(job['male_model'])
        if not neutral.is_file():raise FileNotFoundError('Licensed SMPL-X neutral asset required: '+str(neutral))
        if config['render']['avatar']=='male_slim' and not male.is_file():raise FileNotFoundError('Male display asset required; use render.avatar=native if unavailable')
        native_outputs=[args.output/f'native/p{p}/1_hmr4d_results_native.pt' for p in (0,1)]+[args.output/'cuts.json']
        native_outputs += [args.output/f'native_cache/p{p}'/args.video.stem/'preprocess'/name for p in (0,1) for name in ('vitpose.pt','bbx.pt')]
        model_inputs=[args.gvhmr_root/'inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt',args.gvhmr_root/'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt']
        stage('06_native','native',args.hmr_python,[Path(job['selected_csv']),final_selection,*model_inputs],native_outputs,
            ['src/scene_pipeline/native.py','work/rtmlib_to_gvhmr_adapter.py'])
        scene_outputs=[args.output/'scene/inputs/scene.npz',args.output/'scene/inputs/scene.json',args.output/'scene/results/scene/alignment.npz',args.output/'scene/results/scene/metrics.json']
        if config['render']['avatar']=='male_slim':scene_outputs.append(args.output/'scene/inputs/scene_male_slim.npz')
        scene_sources=[str(p.relative_to(REPO)) for p in (REPO/'src/scene_pipeline/scene').glob('*.py')]
        stage('07_scene','scene',args.hmr_python,[*native_outputs,final_court,neutral,REPO/'configs/male_slim.json',*([male] if config['render']['avatar']=='male_slim' else [])],scene_outputs,scene_sources)
        render_outputs=[args.output/'reconstruction.mp4',args.output/'video_verification.json'] if job['render_requested'] else []
        if job['render_requested']:
            stage('08_render','render',args.hmr_python,[*scene_outputs,final_court],render_outputs,
                ['src/scene_pipeline/render.py','src/scene_pipeline/visual.py'])
        stage('09_verify','verify',args.hmr_python,[*scene_outputs,*native_outputs,final_court,*render_outputs],[args.output/'verification.json'],['src/scene_pipeline/verify.py'])
    write(args.output/'run_manifest.json',dict(version=config['version'],video_sha256=job['video_sha256'],video_info=info,config=config,
        manual_points=False,court_network_used=False,native_articulation_changed=False,feedback_decision=reason,court_result=str(final_court),
        selected_csv=job['selected_csv'],elapsed_seconds=time.time()-started,court_only=args.court_only,stages=status['stages']))
    status.update(status='complete',elapsed_seconds=time.time()-started);write(args.output/'pipeline_status.json',status)
    print('[DONE]',args.output,flush=True)

def main(argv=None):run(parse(argv))
