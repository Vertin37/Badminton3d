from pathlib import Path
from .artifacts import read

def run(stage,path):
    job=read(path)
    if stage=='detect':
        import run_pose_test
        run_pose_test.run_video_inference(Path(job['video']),Path(job['detection_output']),'auto',False)
    elif stage=='select':
        from .selection import select
        select(job['video'],job['detections'],job['selection_output'],job['config']['selection'],job.get('court_reference'))
    elif stage=='court':
        from .geometry.run import run as court
        court('scene',Path(job['video']),Path(job['selected_csv']),[0,1],Path(job['court_output']))
    elif stage=='native':
        from .native import run as native
        native(job)
    elif stage=='scene':
        from .scene.build import run as scene
        scene(job)
    elif stage=='render':
        from .render import render
        render(job)
    elif stage=='verify':
        from .verify import verify
        verify(job)
    else:raise ValueError('Unknown stage '+stage)
