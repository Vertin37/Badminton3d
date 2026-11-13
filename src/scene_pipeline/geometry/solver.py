"""Joint floor/line/target-player calibration with explicit alternative layouts."""
import numpy as np,cv2
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares,linear_sum_assignment
from .common import WIDTH,HEIGHT,CORNERS,SEGMENTS,KINDS,SAMPLED,SYMMETRIES,project,homography,valid
from .observations import family_maps,angle_delta

GRID=np.array([[x,y] for y in np.linspace(-6.3,6.3,9) for x in np.linspace(-2.9,2.9,5)])
FOCALS=np.geomspace(WIDTH*.2,WIDTH*5,64)

def camera_metric(h):
    # A metric court must be compatible with perpendicular, equally scaled
    # world axes under a pinhole camera. Fit unknown focal length; no court
    # image annotation or camera calibration is supplied.
    center=np.array([WIDTH/2,HEIGHT/2])
    a=h[:2,0]-center*h[2,0];b=h[:2,1]-center*h[2,1]
    aa=np.c_[np.broadcast_to(a,(len(FOCALS),2)),FOCALS*h[2,0]]
    bb=np.c_[np.broadcast_to(b,(len(FOCALS),2)),FOCALS*h[2,1]]
    na=np.linalg.norm(aa,axis=1).clip(1e-9);nb=np.linalg.norm(bb,axis=1).clip(1e-9)
    cosine=(aa*bb).sum(1)/(na*nb);ratio=np.log(na/nb)
    error=(cosine/.15)**2+(ratio/.2)**2;best=int(np.argmin(error))
    return np.array([cosine[best]/.15,ratio[best]/.2]),float(FOCALS[best])

def maps(obs,threshold):
    key=f'families_{threshold}'
    if key not in obs:obs[key]=family_maps(obs,threshold)
    return obs[key]

def pairs(obs):
    values=[hint['feet'] for hint in obs['hints'] if hint['pair_valid']]
    return np.stack(values) if values else np.empty((0,2,2))

def family_map(h,obs):
    p=project(h,np.array([[0.,-3],[0,3.]]));v=p[1]-p[0]
    angle=np.degrees(np.arctan2(v[1],v[0]))%180
    return (0,1) if angle_delta(angle,obs['axis'])<40 else (1,0)

def inside_polygon(q,points,margin=0.):
    edges=np.roll(q,-1,axis=0)-q
    area=np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))
    d=points[:,None]-q[None]
    cross=edges[None,:,0]*d[:,:,1]-edges[None,:,1]*d[:,:,0]
    return (cross*np.sign(area)>=-margin*np.linalg.norm(edges,axis=1)[None]).all(1)

def observation_points(h,obs,seed):
    q=project(h,CORNERS);families=maps(obs,seed['partition']);points=[];kinds=[];line_ids=[];retained=[]
    inverse_map={seed['family_map'][kind]:kind for kind in (0,1)}
    for group,lines in enumerate(families['groups']):
        for index,line in enumerate(lines):
            a,b=line['segment'].reshape(2,2);p=np.linspace(a,b,11)
            p=p[inside_polygon(q,p,3)]
            if len(p)<3:continue
            retained.append({'points':p,'angle':line['angle'],'kind':inverse_map[group]})
            selected=p[np.linspace(0,len(p)-1,min(6,len(p))).astype(int)]
            points.extend(selected);kinds.extend([inverse_map[group]]*len(selected));line_ids.extend([len(retained)-1]*len(selected))
    return np.asarray(points).reshape(-1,2),np.asarray(kinds,int),np.asarray(line_ids,int),retained

def segment_distances(points,endpoints):
    a=endpoints[:,0];v=endpoints[:,1]-a
    d=points[:,None]-a[None]
    u=np.sum(d*v[None],axis=-1)/np.sum(v*v,axis=1)[None].clip(1e-9)
    nearest=a[None]+u.clip(0,1)[:,:,None]*v[None]
    return np.linalg.norm(points[:,None]-nearest,axis=-1)

def correspondence(h,obs,seed):
    _,_,_,lines=observation_points(h,obs,seed)
    endpoints=np.stack([project(h,s) for s in SEGMENTS])
    angles=np.rad2deg(np.arctan2((endpoints[:,1]-endpoints[:,0])[:,1],(endpoints[:,1]-endpoints[:,0])[:,0]))%180
    cost=np.full((len(SEGMENTS),len(lines)),100.)
    for j,line in enumerate(lines):
        distance=np.median(segment_distances(line['points'],endpoints),axis=0)
        angle=angle_delta(angles,line['angle'])
        compatible=KINDS==line['kind']
        cost[compatible,j]=distance[compatible]/4+angle[compatible]/7
    if not lines:return {'matched_lines':0,'matched_by_direction':[0,0],'reverse_support':0.}
    row,column=linear_sum_assignment(cost)
    good=cost[row,column]<2.2
    count=[int(np.sum(good&(KINDS[row]==kind))) for kind in (0,1)]
    reverse=float(np.mean(np.min(cost,axis=0)<2.2))
    return {'matched_lines':int(good.sum()),'matched_by_direction':count,'reverse_support':reverse}

def score(seed,obs):
    h=seed['H']
    if not valid(h):return {'eligible':False,'score':-1.,'reason':'invalid_projection'}
    target_pairs=pairs(obs)
    if not len(target_pairs):return {'eligible':False,'score':-1.,'reason':'missing_target_pair'}
    world=project(np.linalg.inv(h),target_pairs.reshape(-1,2)).reshape(-1,2,2)
    excess=np.maximum(abs(world)-[3.55,7.45],0)
    inside=float(np.exp(-np.linalg.norm(excess,axis=2)/.45).mean())
    opposite=float(np.mean((np.min(world[:,:,1],axis=1)<=.15)&(np.max(world[:,:,1],axis=1)>=-.15)))
    family=maps(obs,seed['partition']);coverage=[];visible=0
    for index,samples in enumerate(SAMPLED):
        p=project(h,samples);keep=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<WIDTH)&(p[:,1]>=0)&(p[:,1]<HEIGHT)
        if keep.sum()<5:continue
        p=p[keep];d=map_coordinates(family['distances'][seed['family_map'][KINDS[index]]],[p[:,1],p[:,0]],order=1,mode='nearest')
        d=np.minimum(d,100.)
        coverage.append(float(np.exp(-d*d/18).mean()));visible+=1
    forward=float(np.mean(coverage)) if coverage else 0.
    grid=project(h,GRID);keep=np.isfinite(grid).all(1)&(grid[:,0]>=0)&(grid[:,0]<WIDTH)&(grid[:,1]>=0)&(grid[:,1]<HEIGHT)
    p=grid[keep];floor_score=0.
    if len(p):
        known=map_coordinates(obs['body_mask'],[p[:,1],p[:,0]],order=0,mode='nearest')==0
        if known.any():floor_score=float(map_coordinates(obs['floor'],[p[known,1],p[known,0]],order=0,mode='nearest').mean())
    matching=correspondence(h,obs,seed)
    metric,focal=camera_metric(h);metric_error=float(metric@metric)
    total=.30*forward+.18*matching['reverse_support']+.12*min(matching['matched_lines']/8,1)+.20*inside+.12*opposite+.08*floor_score-.035*min(metric_error,15)
    eligible=inside>.78 and opposite>.65 and matching['matched_lines']>=5 and min(matching['matched_by_direction'])>=2 and floor_score>.7 and metric_error<6
    return {'score':float(total),'eligible':bool(eligible),'forward_line_support':forward,'visible_template_lines':visible,
            'player_containment':inside,'opposite_half_fraction':opposite,'floor_support':floor_score,
            'camera_metric_error':metric_error,'focal_length_px_proxy':focal,**matching}

def refine(seed,obs,max_evaluations=45):
    h=seed['H'];initial=project(h,CORNERS);target_pairs=pairs(obs)
    if not valid(h) or len(target_pairs)==0:return seed
    family=maps(obs,seed['partition']);forward_world=[];forward_groups=[]
    for index,world in enumerate(SAMPLED):
        p=project(h,world);keep=np.isfinite(p).all(1)&(p[:,0]>1)&(p[:,0]<WIDTH-2)&(p[:,1]>1)&(p[:,1]<HEIGHT-2)
        if keep.sum()>=5:
            forward_world.extend(world[keep]);forward_groups.extend([seed['family_map'][KINDS[index]]]*keep.sum())
    forward_world=np.asarray(forward_world).reshape(-1,2);forward_groups=np.asarray(forward_groups)
    reverse_points,reverse_kinds,_,_=observation_points(h,obs,seed)
    initial_area=abs(cv2.contourArea(initial.astype(np.float32)))
    grid0=project(h,GRID);visible=np.isfinite(grid0).all(1)&(grid0[:,0]>1)&(grid0[:,0]<WIDTH-2)&(grid0[:,1]>1)&(grid0[:,1]<HEIGHT-2)
    grid_world=GRID[visible];norm=max(np.sqrt(len(target_pairs)),1)
    before=score(seed,obs)
    nres=None
    def residual(raw):
        q=raw.reshape(4,2);matrix=homography(q)
        try:inverse=np.linalg.inv(matrix)
        except np.linalg.LinAlgError:return np.full(nres or 1,100.)
        p=project(matrix,forward_world);forward=np.zeros(len(p))
        for group in (0,1):
            choose=forward_groups==group
            if choose.any():forward[choose]=map_coordinates(family['distances'][group],[p[choose,1],p[choose,0]],order=1,mode='constant',cval=15.)
        components=[np.minimum(forward,15)/3]
        endpoints=np.stack([project(matrix,s) for s in SEGMENTS])
        for kind in (0,1):
            choose=reverse_kinds==kind
            if choose.any():
                distances=segment_distances(reverse_points[choose],endpoints[KINDS==kind]).min(1)
                components.append(np.minimum(distances,15)/4)
        world=project(inverse,target_pairs.reshape(-1,2)).reshape(-1,2,2)
        components.extend([(np.maximum(abs(world[:,:,0])-3.55,0)/.4*6/norm).ravel(),
                           (np.maximum(abs(world[:,:,1])-7.45,0)/.6*6/norm).ravel(),
                           np.maximum(np.min(world[:,:,1],axis=1),0)/.3*8/norm,
                           np.maximum(-np.max(world[:,:,1],axis=1),0)/.3*8/norm])
        grid=project(matrix,grid_world)
        components.append(map_coordinates(obs['floor_distance'],[grid[:,1],grid[:,0]],order=1,mode='constant',cval=20.)/10)
        area=abs(cv2.contourArea(q.astype(np.float32)))
        components.extend([camera_metric(matrix)[0]*4,((q-initial)/55).ravel(),np.array([max(0,np.log(initial_area/max(area,1)))*10])])
        result=np.concatenate(components)
        return np.nan_to_num(result,nan=100,posinf=100,neginf=-100)
    nres=len(residual(initial.ravel()))
    bounds=np.tile([WIDTH*.13,HEIGHT*.22],4)
    optimized=least_squares(residual,initial.ravel(),bounds=(initial.ravel()-bounds,initial.ravel()+bounds),
                            diff_step=.001,max_nfev=max_evaluations,loss='soft_l1',f_scale=1.5)
    new={**seed,'H':homography(optimized.x.reshape(4,2)),'joint_optimizer_evaluations':int(optimized.nfev)}
    after=score(new,obs)
    if after['score']>=before['score'] and valid(new['H']):
        return {**new,'diagnostics':after,'refined':True,'residual_terms':['directional_lines','reverse_lines','player_containment','opposite_halves','floor_region','pinhole_metric_axes','shape_prior']}
    return {**seed,'diagnostics':before,'refined':False,'joint_optimizer_evaluations':int(optimized.nfev)}
