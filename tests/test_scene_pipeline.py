import copy,csv,io,json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from src.scene_pipeline.artifacts import cached,finish,digest,write
from src.scene_pipeline.fusion import CourtReference,near_target_court,accept_feedback,trusted
from src.scene_pipeline.selection import load_detections
from src.scene_pipeline.geometry.solver import camera_metric,refine,score,family_map
from src.scene_pipeline.geometry.observations import detect
from src.scene_pipeline.geometry.common import CORNERS,WIDTH,HEIGHT,project,homography,overlay,valid
from src.scene_pipeline.scene.align import initialize,camera
from src.scene_pipeline.scene.support import COURT

def calibrated(h=None):
    h=np.eye(3) if h is None else h
    c=dict(status='supported',homography=h.tolist(),selected={'diagnostics':dict(matched_lines=8,matched_by_direction=[4,4],camera_metric_error=.1,
        player_containment=.99,forward_line_support=.8,floor_support=.95)})
    return dict(groups=[dict(id=0,calibration=c)],frames=[dict(source_frame=0,group=0,homography_metric_to_original=h.tolist())])

def physical_camera():
    center=np.array([7.,-16,10]);z=-center/np.linalg.norm(center);x=np.cross(z,[0,0,1]);x/=np.linalg.norm(x);r=np.stack([x,np.cross(z,x),z])
    k=np.array([[700.,0,480],[0,700,270],[0,0,1.]])
    return k@np.c_[r[:,:2],-r@center],r,-r@center

class SceneFusionTests(unittest.TestCase):
    def setUp(self):cv2.setNumThreads(1)
    def test_court_feedback_rejects_spectator_but_allows_save(self):
        h=np.array([[40,0,400],[0,20,250],[0,0,1.]])
        self.assertTrue(near_target_court(h,project(h,[[4.,-3]])[0]))
        self.assertFalse(near_target_court(h,project(h,[[12.,-3]])[0]))
        self.assertTrue(near_target_court(None,[12000,12000]))
    def test_unsupported_court_does_not_gate_people(self):
        r=calibrated();r['groups'][0]['calibration']['status']='ambiguous'
        self.assertIsNone(CourtReference(r).at(5))
        r=calibrated();r['groups'][0]['calibration']['selected']['diagnostics']['camera_metric_error']=20
        self.assertIsNone(CourtReference(r).at(0))
    def test_airborne_ankle_not_forced_onto_ground(self):
        h=np.array([[40.,0,400],[0,-20,250],[0,0,1]])
        ankle=project(h,[[0.,10.]])[0]
        self.assertFalse(near_target_court(h,ankle))
        self.assertTrue(near_target_court(h,ankle,image_drop=40))
        spectator=project(h,[[12.,10.]])[0]
        self.assertFalse(near_target_court(h,spectator,image_drop=40))
    def test_feedback_is_guarded_by_lines_not_self_score(self):
        base=calibrated();candidate=copy.deepcopy(base)
        candidate['groups'][0]['calibration']['selected']['diagnostics'].update(score=100,forward_line_support=.3)
        self.assertEqual(accept_feedback(base,candidate),(False,'line_support_worse'))
        self.assertTrue(accept_feedback(base,base)[0])
    def test_feedback_drift_rejected(self):
        a=calibrated();b=calibrated(np.array([[1.,0,100],[0,1,0],[0,0,1]]))
        self.assertEqual(accept_feedback(a,b),(False,'excessive_feedback_drift'))
    def test_missing_target_pair_has_no_positive_score(self):
        image=np.full((HEIGHT,WIDTH,3),(65,155,65),np.uint8)
        hint=dict(feet=np.array([[200,200],[500,300]]),weights=np.ones(2),boxes=[],pair_valid=False)
        obs=detect(image,[hint]);h,_,_=physical_camera()
        d=score(dict(H=h,partition=18.,family_map=(0,1)),obs)
        self.assertEqual(d['score'],-1);self.assertFalse(d['eligible'])
    def test_player_constraints_actually_move_geometry(self):
        h,_,_=physical_camera();image=np.full((HEIGHT,WIDTH,3),(65,155,65),np.uint8)
        hint=dict(feet=project(h,[[-1,-3],[1,3]]),weights=np.ones(2),boxes=[],pair_valid=True)
        obs=detect(image,[hint]);seed=dict(H=h@np.array([[1.,0,0],[0,1,4.5],[0,0,1.]]),partition=18.,family_map=family_map(h,obs))
        result=refine(seed,obs,max_evaluations=80)
        self.assertGreater(score(result,obs)['opposite_half_fraction'],score(seed,obs)['opposite_half_fraction'])
    def test_metric_camera_axes_and_positive_height(self):
        h,r,t=physical_camera();metric,_=camera_metric(h);self.assertLess(metric@metric,.1)
        pts=COURT@r.T+t;uv=pts[:,:2]/pts[:,2:]*700+[480,270]
        confidence=np.zeros(len(COURT));confidence[:32]=1;initialize.pp=np.array([480,270])
        raw=initialize(uv,confidence,700,r[:,2]);rr,tt,f,c=camera(raw)
        self.assertGreater(c[2],0);self.assertAlmostEqual(np.linalg.det(rr),1,places=7)
        np.testing.assert_allclose(rr,r,atol=1e-5)
    def test_resume_validates_output_content(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'output.json';r=Path(folder)/'stage.json';p.write_text('good')
            finish(r,'inputs-v1',[p]);self.assertTrue(cached(r,'inputs-v1'));self.assertFalse(cached(r,'inputs-v2'))
            p.write_text('corrupted');self.assertFalse(cached(r,'inputs-v1'))
    def test_variable_frame_count_and_detection_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'raw.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['frame_id','person_id','keypoint_id','x','y','confidence'])
                w.writerow([1,105,15,10,20,.9]);w.writerow([7,105,16,15,25,.8])
            frames=load_detections(p,7);self.assertEqual(len(frames),7);self.assertEqual(frames[0].shape,(1,17,3))
            with self.assertRaises(ValueError):load_detections(p,6)
    def test_missing_csv_coordinates_remain_unobserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'raw.csv'
            p.write_text('frame_id,person_id,keypoint_id,x,y,confidence,raw_valid\n1,0,15,,,0,0\n1,0,16,20,30,.8,0\n')
            points=load_detections(p,1)[0];self.assertEqual(points[0,15,2],0);self.assertEqual(points[0,16,2],0)

if __name__=='__main__':unittest.main()
