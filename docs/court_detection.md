# Court line detection (net/post guided version)

The independent module in `src/court_detection/` detects image-space court
structure. It does not load RTMLib, Open3D, or VGGT. The optional pole-mapping
path uses a ground-plane homography only to draw the standard court back onto
the original image; it does not export a 3-D scene or alter pose recognition.

Run it in the configured environment:

```bat
conda activate badminton3d
cd /d D:\Projects\BadmintonPose
python run_court_detection.py --video examples\data\test.mp4 --output-dir outputs\pose_test\court_detection_pole_mapping --keyframes 5 --pose-csv outputs\pose_test\pose_data_stable.csv --no-labels --no-edge-preview --all-frames --reference-frame 137 --pole-mapping
```

The pole-mapping mode first gets each player's foot center from keypoints 15
and 16. It detects compact post-base regions, tests adjacent post pairs, and
keeps the pair whose bottom-to-bottom segment intersects the player-foot
segment. The standard 13.4 m x 6.1 m court is projected with a ground-plane
Homography: the selected pole feet anchor the target net, nearby court-line
directions and a detected cross-court line provide the second depth anchor,
and standard badminton depths (back boundary and service lines) are scored
against all detected lines. A guarded net-only PnP fallback is retained for
frames where the ground-line hypothesis is unavailable. The final overlay
contains four mapped red court sides, the selected purple net segment, the
selected post pair, the player-foot intersection, and the existing filtered
player skeleton.

For a moving video, pole-mapping detects the poles and net edge on every frame
and recomputes the ground-line Homography on every valid frame.
The selected pair is tracked from the reference frame in both directions:
large one-frame jumps to a neighbouring court are rejected, while the current
frame's geometry is still used to follow camera shake. A short temporal blend
removes Hough jitter. Before rendering, an adaptive temporal filter expresses
the court relative to the selected net: net midpoint and direction follow
camera motion, while the normalized court shape changes slowly. The gain is
based on relative pole-span motion rather than fixed image coordinates, so a
real pan or zoom can be followed but isolated line-assignment errors do not
make the four-sided frame expand or contract. The people are still drawn from the existing
`pose_data_stable.csv` on every frame; no pose recognition code is changed.
The line-based path does not require an assumed focal length for the 2-D court
overlay. The PnP fallback still records its approximate focal-length
assumption in JSON.

The underlying detector uses CPU OpenCV Canny and HoughLinesP. Candidate lines
are scored using white-line support, green-court support, length, edge support,
direction families, and spatial position. The output labels are heuristic hints:

- `net_candidate`: a player-corridor net-edge hypothesis;
- `boundary_candidate`: outer candidates in the dominant cross-court family;
- `service_candidate`: interior candidates in that family;
- `court_line`: retained line without a stronger role assignment.

Outputs:

- `court_detection_frame_*.jpg`: annotated original frames;
- `court_detection_edges_*.jpg`: Canny/Hough previews;
- `court_detection_video.mp4`: per-frame mapped net/post/court overlay plus
  per-frame filtered two-player skeleton;
- `court_detection_lines.json`: `lines` contains only the four final boundary
  lines; `candidate_lines` retains the discarded Hough candidates for debugging.

For a single frame, use `--frame 150`. For semi-automatic visual confirmation,
use `--interactive`; frames are saved automatically and `Q` stops the preview.
