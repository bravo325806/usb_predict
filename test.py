import pyrealsense2 as rs
import cv2
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.flip(color_image, 1)
    cv2.imshow('frame', color_image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

