"""CPU court baseline: painted-line geometry and existing target-player tracks."""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,av
from .common import ROOT,PROJECT,CORNERS,WIDTH,HEIGHT,box,read,atomic,digest,project,overlay,imwrite
from .tracks import Tracks,PATHS
from .motion import features,match
from .calibrate import calibrate

VERSION='court_geometry_v1'
CORE_FILES=('common.py','tracks.py','motion.py','observations.py','seeds.py','solver.py','calibrate.py','run.py')

def fingerprint(files):return ''.join(digest(ROOT/name) for name in files)

def prepare(name,video,tracks,directory,reuse=False,motion_cache=None):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError('Cannot open video: '+str(video))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    if n<=0 or fps<=0:raise RuntimeError('Invalid frame count/FPS')
    step=max(1,round(fps/5));video_sha=digest(video)
    signature=fingerprint(('common.py','tracks.py','motion.py'))+tracks.sha256+video_sha+json.dumps(tracks.player_ids)
    saved=read(directory/'motion.json') if reuse else {}
    provenance='computed_from_video'
    if saved.get('signature')!=signature and motion_cache is not None:
        source=motion_cache/name/'motion.json';saved=read(source)
        expected=digest(ROOT/'motion.py')+tracks.sha256+video_sha
        if not saved or saved.get('signature')!=expected or tracks.player_ids!=(0,1):
            raise RuntimeError('Motion cache does not match video, tracks and motion algorithm: '+str(source))
        saved={**saved,'signature':signature};provenance='verified_previous_geometric_motion'
    cached={r['source_frame']:r for r in saved.get('frames',[])} if saved.get('signature')==signature else {}
    frames=[];anchor=None;anchor_index=0;group=0
    for i in range(n):
        ok,image=cap.read()
        if not ok:raise RuntimeError('Video decode failed at frame '+str(i))
        if i%step:continue
        small,t=box(image);hint=tracks.get(i,t)
        if i in cached:
            previous=cached[i];frames.append({**previous,'image':small,'transform':t,'motion':np.asarray(previous['motion']),'hint':hint});continue
        if cached:raise RuntimeError('Incomplete motion cache; rerun without --reuse')
        current=features(small,hint)
        if anchor is None:
            h=np.eye(3);diagnostic={'status':'initial_anchor'};anchor=current
        else:
            h,diagnostic=match(anchor,current)
            if h is None:
                difference=float(np.mean(abs(small.astype(float)-frames[-1]['image'].astype(float))))
                if difference<7:
                    h=frames[-1]['motion'].copy();diagnostic={**diagnostic,'status':'near_static_fallback','image_difference':difference}
                else:
                    group+=1;h=np.eye(3);anchor=current;anchor_index=len(frames)
                    diagnostic={**diagnostic,'status':'new_view','image_difference':difference}
        frames.append({'source_frame':i,'timestamp':i/fps,'image':small,'transform':t,'hint':hint,'motion':h,
                       'group':group,'anchor_index':anchor_index,'motion_diagnostics':diagnostic})
        if len(frames)%50==0:print(name,'motion frames',len(frames),flush=True)
    cap.release();groups=[]
    for identifier in sorted({r['group'] for r in frames}):
        indices=[i for i,r in enumerate(frames) if r['group']==identifier]
        groups.append({'id':identifier,'anchor_index':indices[0],'indices':indices,
                       'first_source_frame':frames[indices[0]]['source_frame'],'last_source_frame':frames[indices[-1]]['source_frame']})
    atomic(directory/'motion.json',{'signature':signature,'source_video':str(video),'video_sha256':video_sha,
        'tracks_sha256':tracks.sha256,'player_ids':tracks.player_ids,'source_frames':n,'source_fps':fps,'sample_fps':fps/step,
        'provenance':saved.get('provenance',provenance) if cached else provenance,'groups':groups,
        'frames':[{k:v for k,v in r.items() if k not in ('image','hint')} for r in frames]})
    return frames,groups,fps/step

def render(name,frames,directory,rate):
    path=directory/f'{name}_court_geometry.mp4';writer=av.open(str(path),'w')
    stream=writer.add_stream('libx264',rate=round(rate));stream.width=WIDTH;stream.height=HEIGHT+48
    stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'}
    selected=set(np.linspace(0,len(frames)-1,min(8,len(frames))).round().astype(int));panels=[]
    for i,row in enumerate(frames):
        picture=overlay(row['image'],row['H'])
        for p in row['hint']['feet']:
            if np.isfinite(p).all():cv2.circle(picture,tuple(p.round().astype(int)),4,(40,230,40),-1)
        panel=np.full((HEIGHT+48,WIDTH,3),24,np.uint8);panel[48:]=picture
        cv2.putText(panel,f'{VERSION} | {name} {row["timestamp"]:.1f}s | lines + same 2D tracks',(12,30),0,.6,(255,255,255),1)
        for packet in stream.encode(av.VideoFrame.from_ndarray(panel,format='bgr24')):writer.mux(packet)
        if i in selected:panels.append(panel)
    for packet in stream.encode():writer.mux(packet)
    writer.close();decoder=av.open(str(path));count=sum(1 for _ in decoder.decode(video=0));decoder.close()
    if count!=len(frames):raise RuntimeError('Export verification failed')
    imwrite(directory/'first_frame.jpg',panels[0]);imwrite(directory/'overview.jpg',np.vstack(panels))
    return {'path':str(path),'frames':count,'fps':rate,'fully_decoded_verified':True}

def run(name,video,track_path,player_ids,output,reuse=False,motion_cache=None):
    started=time.perf_counter();video=video.resolve();track_path=track_path.resolve()
    if not video.is_file() or not track_path.is_file():raise FileNotFoundError('Video or target tracks missing')
    cap=cv2.VideoCapture(str(video));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    tracks=Tracks(name,n,tuple(player_ids),track_path)
    directory=output/name;directory.mkdir(parents=True,exist_ok=True)
    frames,groups,rate=prepare(name,video,tracks,directory,reuse,motion_cache)
    signature=fingerprint(CORE_FILES)+digest(video)+tracks.sha256+json.dumps(player_ids)
    for group in groups:
        rows=[frames[i] for i in group['indices']];anchor=rows[0]
        selected=np.unique(np.r_[0,np.linspace(0,len(rows)-1,min(30,len(rows))).round().astype(int)])
        hints=[tracks.get(rows[j]['source_frame'],np.linalg.inv(rows[j]['motion'])@rows[j]['transform']) for j in selected]
        dest=directory/f'view_{group["id"]:02d}';result=read(dest/'calibration.json') if reuse else {}
        if result.get('signature')!=signature:
            result=calibrate(anchor['image'],hints,output=dest);result['signature']=signature;atomic(dest/'calibration.json',result)
        group['calibration']=result;group['fit_target_source_frames']=[hint['source_frame'] for hint in hints]
        h=np.asarray(result['homography']) if result['homography'] is not None else None
        for row in rows:row['H']=row['motion']@h if h is not None else None
        print(name,'view',group['id'],result['status'],flush=True)
    exported=render(name,frames,directory,rate)
    loaded=[k for k in ('torch','torchvision','neural') if k in sys.modules]
    result={'name':name,'module':VERSION,'version':1,'runtime':'CPU','court_network_used':False,'checkpoint':None,
        'loaded_ml_modules':loaded,'source_video':str(video),'source_video_sha256':digest(video),'target_tracks':str(track_path),
        'target_tracks_sha256':tracks.sha256,'target_player_ids':player_ids,'manual_court_points_or_cuts_used':False,
        'gvhmr_parameters_modified':False,'groups':groups,
        'frames':[{'source_frame':r['source_frame'],'timestamp':r['timestamp'],'group':r['group'],
            'homography_metric_to_original':(np.linalg.inv(r['transform'])@r['H']).tolist() if r['H'] is not None else None,
            'motion':r['motion']} for r in frames],
        'coordinate_system':{'unit':'metre','width':6.10,'length':13.40,'origin':'court_center','reflection_and_half_turn_ambiguity':True},
        'video':exported,'elapsed_seconds':time.perf_counter()-started,'core_fingerprint':fingerprint(CORE_FILES)}
    atomic(directory/'result.json',result);print(name,'saved',exported['path'],flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description='Court geometry v1; video + target-player 2D CSV; CPU only.')
    inputs=parser.add_mutually_exclusive_group();inputs.add_argument('--names',nargs='+');inputs.add_argument('--video',type=Path)
    parser.add_argument('--tracks',type=Path);parser.add_argument('--name');parser.add_argument('--player-ids',type=int,nargs=2,default=[0,1])
    parser.add_argument('--output',type=Path,default=ROOT/'results');parser.add_argument('--reuse',action='store_true');parser.add_argument('--motion-cache',type=Path)
    args=parser.parse_args();cv2.setNumThreads(1)
    if args.player_ids[0]==args.player_ids[1]:parser.error('Two different target player IDs are required')
    if args.video:
        if args.tracks is None:parser.error('--video requires --tracks from the existing target-person tracking pipeline')
        jobs=[(args.name or args.video.stem,args.video,args.tracks)]
    else:
        if args.tracks is not None or args.name:parser.error('--tracks and --name are used with --video')
        names=args.names or ['test','test2','test3']
        if any(name not in PATHS for name in names):parser.error('For a new video use --video and --tracks')
        jobs=[(name,PROJECT/f'examples/data/{name}.mp4',PATHS[name]) for name in names]
    for name,video,path in jobs:
        if name in ('.','..') or Path(name).name!=name or '/' in name or '\\' in name:parser.error('Invalid output name')
        run(name,video,path,args.player_ids,args.output,args.reuse,args.motion_cache)
