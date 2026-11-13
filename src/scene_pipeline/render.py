from pathlib import Path
from fractions import Fraction
import av,cv2,numpy as np,json
from .visual import text,camera,court,draw_mesh,line3
from .artifacts import read
COLORS=[(65,177,245),(249,164,74)]
PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

def fit(canvas,image):
    h,w=canvas.shape[:2];s=min(w/image.shape[1],h/image.shape[0]);size=(max(1,round(image.shape[1]*s)),max(1,round(image.shape[0]*s)))
    im=cv2.resize(image,size);x=(w-size[0])//2;y=(h-size[1])//2;canvas[y:y+size[1],x:x+size[0]]=im

def render(job):
    from .scene.align import camera as decode,project_camera
    ROOT=Path(job['output']);NAME='scene'
    scene=ROOT/'scene';data=dict(np.load(scene/'inputs/scene.npz'));a=dict(np.load(scene/'results/scene/alignment.npz'))
    if job['config']['render']['avatar']=='male_slim':
        appearance=np.load(scene/'inputs/scene_male_slim.npz');mesh=appearance['local_mesh_cam'];faces=appearance['faces']
    else:mesh=data['local_mesh_cam'];faces=data['faces']
    cs=read(job['court_result'])['frames']
    capture=cv2.VideoCapture(str(Path(job['video'])));n=data['uv'].shape[1];fps=float(data['fps']);pp=np.array([float(data['width'])/2,float(data['height'])/2])
    stride=max(1,round(fps/job['config']['render']['fps']));expected=len(range(0,n,stride))
    path=ROOT/'reconstruction.mp4';writer=av.open(str(path),'w');stream=writer.add_stream('libx264',rate=Fraction(str(fps/stride)).limit_denominator(1001))
    stream.width=1600;stream.height=1000;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'}
    scene_project,sr,sc=camera([13,-18,13],[0,0,.4],800,450,760)
    segments=[[[x,-6.7],[x,6.7]] for x in [-3.05,-2.59,2.59,3.05]]+[[[-3.05,y],[3.05,y]] for y in [-6.7,-5.94,-1.98,1.98,5.94,6.7]]+[[[0,-6.7],[0,-1.98]],[[0,1.98],[0,6.7]]]
    thumbnails=[];count=0
    for i in range(n):
        ok,image=capture.read()
        if not ok:raise RuntimeError('Source decode failure')
        if i%stride:continue
        original=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);overlay=original.copy();canvas=np.full((1000,1600,3),(12,20,32),np.uint8)
        row=min(cs,key=lambda row:abs(row['source_frame']-i));h=np.asarray(row['homography_metric_to_original'])
        for segment in segments:
            q=np.c_[segment,np.ones(2)]@h.T;uv=q[:,:2]/q[:,2:];cv2.line(overlay,tuple(uv[0].round().astype(int)),tuple(uv[1].round().astype(int)),(64,224,209),2,cv2.LINE_AA)
        r,t,f,c=decode(a['refined_cameras'][i]);roots=a['joint_roots_world'][:,i]
        from .scene.support import COURT
        from .scene.align import project
        projected_court=project(COURT,r,t,f,pp)
        for left,right in [(j*5,j*5+4) for j in range(6)]+[(j,25+j) for j in [0,1,3,4]]+[(2,12),(17,27)]:
            q=projected_court[[left,right]]
            if np.isfinite(q).all() and abs(q).max()<30000:cv2.line(overlay,tuple(q[0].round().astype(int)),tuple(q[1].round().astype(int)),(235,180,100),1,cv2.LINE_AA)
        for p in (0,1):
            points=project_camera(data['local_joints_cam'][p,i]+(roots[p]@r.T+t),f,pp)
            for j in range(1,22):
                q=points[[PARENTS[j],j]]
                if np.isfinite(q).all() and abs(q).max()<30000:cv2.line(overlay,tuple(q[0].round().astype(int)),tuple(q[1].round().astype(int)),COLORS[p],2,cv2.LINE_AA)
        fit(canvas[45:495,:800],overlay);view=canvas[45:495,800:]
        court(view,scene_project)
        for p in sorted((0,1),key=lambda p:np.linalg.norm(roots[p]-sc),reverse=True):draw_mesh(view,scene_project,sr,sc,mesh[p,i]@r+roots[p],faces,COLORS[p])
        for p in (0,1):
            x=p*800;uv=data['uv'][p,i];valid=data['confidence'][p,i]>.3
            if valid.sum()>=3:
                lo=uv[valid].min(0);hi=uv[valid].max(0);center=(lo+hi)/2;size=max(float(hi[1]-lo[1])*1.5,100.)
                x0,x1=np.clip([center[0]-.4*size,center[0]+.4*size],0,original.shape[1]).astype(int)
                y0,y1=np.clip([center[1]-.5*size,center[1]+.5*size],0,original.shape[0]).astype(int)
                if x1>x0 and y1>y0:fit(canvas[550:950,x:x+400],original[y0:y1,x0:x1])
            v=canvas[550:950,x+400:x+800];center=roots[p]-r[2]*3.
            def body_project(points):
                camera_points=(np.asarray(points)-center)@r.T
                return camera_points[...,:2]/np.maximum(camera_points[...,2:3],.1)*440+[200,200]
            for g in [-.5,0,.5]:
                line3(v,body_project,[[roots[p,0]-1,roots[p,1]+g,0],[roots[p,0]+1,roots[p,1]+g,0]],(42,83,85))
            draw_mesh(v,body_project,r,center,mesh[p,i]@r+roots[p],faces,COLORS[p])
        shot=int(np.searchsorted(data['cut_frames'],i,side='right')-1)
        canvas=text(canvas,[((15,7),f'Video | {i/fps:.2f}s | shot {shot+1}',22,(235,243,249)),
            ((825,7),'GVHMR motion + shared court alignment',22,(235,243,249)),
            ((15,510),'Player 1: original / recovered motion',21,COLORS[0]),((815,510),'Player 2: original / recovered motion',21,COLORS[1]),
            ((15,963),'Cyan: court detection; gold: aligned 3D court; skeleton: 3D projection. Display avatar does not change recovered articulation.',19,(177,202,218))])
        for packet in stream.encode(av.VideoFrame.from_ndarray(canvas,format='rgb24')):writer.mux(packet)
        if count in set(np.linspace(0,expected-1,7).round().astype(int)):
            cv2.imwrite(str(ROOT/f'reconstruction_{count:04d}.jpg'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR));thumbnails.append(cv2.resize(canvas,(800,500)))
        count+=1
        if count%75==0:print('rendered',count,'/',expected,flush=True)
    for packet in stream.encode():writer.mux(packet)
    writer.close();capture.release();decoder=av.open(str(path));decoded=sum(1 for _ in decoder.decode(video=0));decoder.close();assert decoded==expected
    cv2.imwrite(str(ROOT/'reconstruction_overview.jpg'),cv2.cvtColor(np.vstack(thumbnails),cv2.COLOR_RGB2BGR))
    (ROOT/'video_verification.json').write_text(json.dumps({'source_frames':n,'source_fps':fps,'output_frames':decoded,'output_fps':fps/stride,'seconds':decoded/(fps/stride),'audio_streams':0,'fully_decoded':True},indent=2))
    print('Saved',path,flush=True)
