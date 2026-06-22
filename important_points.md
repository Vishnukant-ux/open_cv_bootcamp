# OpenCV Bootcamp - Module 11: Object Tracking

## Overview

This project demonstrates object tracking using OpenCV's tracking API.

Trackers explored:

- BOOSTING
- MIL
- KCF
- CSRT
- TLD
- MEDIANFLOW
- GOTURN
- MOSSE

## Goal

Given the initial location of an object, track its position in subsequent video frames.

## Tracker Comparison

| Tracker | Speed | Accuracy |
|----------|---------|-----------|
| MOSSE | Very Fast | Low |
| KCF | Fast | Good |
| CSRT | Medium | High |
| GOTURN | Medium | Very High |

## Bounding Box Initialization

The object is manually initialized using:

bbox = (1300, 405, 160, 120)


## Creating Tracker

Example:

tracker = cv2.TrackerKCF.create()


## Tracking Loop


ok, bbox = tracker.update(frame)

if ok:
    drawRectangle(frame, bbox)


## FPS Calculation


fps = cv2.getTickFrequency() / (
    cv2.getTickCount() - timer
)


