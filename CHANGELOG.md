# Changelog

## Unreleased

### Added

- Added cross-frame two-player identity tracking for the RTMLib 2-D pose test flow.
- Added per-player/keypoint One Euro temporal filtering with short-gap handling and anomaly rejection.
- Added the first badminton-court detection and stable-tracking pipeline using Canny/HoughLinesP, player-foot-guided net-pole selection, court-line-supported ground-plane homography, and guarded PnP fallback.

### Improved

- Improved handling of low-confidence detections, abnormal jumps, short-term missing points, and temporary missed court detections.
- Improved cross-frame court stability by tracking the reference court, rejecting jumps to neighboring courts, adapting smoothing to relative motion, and correcting isolated geometry outliers.

### Validation

- Validated on 275 frames at 1280 x 720 and 30 FPS: 273 valid court detections, stable tracked court output on all 275 frames, 273 line-based homography mappings, and 0 PnP fallback frames.
- The existing seven player-tracking and temporal-filter tests pass.

### Limitation

- The current result is a stable 2-D perspective projection in the video image. Open3D, VGGT, and human 3-D mapping have not started.
