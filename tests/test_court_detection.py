import cv2
import numpy as np

from src.court_detection import CourtDetectorConfig, CourtLineDetector


def test_detector_finds_lines_on_synthetic_green_court():
    image = np.full((360, 640, 3), (45, 145, 70), dtype=np.uint8)
    cv2.line(image, (30, 100), (610, 100), (245, 245, 245), 3)
    cv2.line(image, (20, 230), (620, 210), (245, 245, 245), 3)
    cv2.line(image, (210, 350), (300, 50), (245, 245, 245), 3)
    cv2.line(image, (390, 350), (350, 50), (245, 245, 245), 3)

    detector = CourtLineDetector(
        CourtDetectorConfig(
            roi_top_ratio=0.05,
            hough_threshold=20,
            min_line_length_ratio=0.08,
            min_white_support=0.05,
            min_floor_support=0.2,
        )
    )
    result = detector.detect(image)

    assert len(result.lines) >= 2
    assert all(line.length > 40 for line in result.lines)
    assert all(0.0 <= line.white_support <= 1.0 for line in result.lines)
    assert all(0.0 <= line.floor_support <= 1.0 for line in result.lines)
    assert detector.annotate(image, result).shape == image.shape
    assert detector.edge_preview(result).shape == image.shape
