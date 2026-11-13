# Changelog

## 2026-11-13 — Integrated person/court scene pipeline

- Added a portable, resumable entry point with versioned configuration and explicit stage contracts.
- Integrated target-person observations into geometric court fitting and bounded court-to-person feedback, including airborne-ankle protection and rejection of unsupported/drifting geometry.
- Connected official frozen GVHMR, metric scene-camera initialization, support-foot/root optimization and slim-male display; preserved recovered articulation.
- Separated detected 2-D court overlays from the aligned 3-D court projection for honest visual review.
- Added content-verified caches, regression tests, actual-video validation notes and full pipeline documentation. Excluded licensed assets, videos and caches.

## 2026-11-10 — Stable 3D workflow publication

- Added the automatic video-to-3D pipeline, fixed-camera GVHMR handling, and tracked calibration export.
- Added monocular depth alignment, Metric Video Depth Anything orchestration, multi-cue trajectory fusion, and support-foot refinement.
- Improved spectator rejection, court-coordinate compatibility, contact correction speed limits, and final trajectory bounds.
- Included the supporting geometry interfaces, regression tests, and reproducible workflow documentation.
- Excluded the separate learned court-localization, annotation/training, and multi-view recognition experiments.

## Earlier stable 2D milestone

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
