import numpy as np
import cv2
import pandas as pd

class EllipseFilter:
    def __init__(self, lower_area_threshold_ratio=0.95, upper_area_threshold_ratio=1.05):
        self.lower_area_threshold_ratio = lower_area_threshold_ratio
        self.upper_area_threshold_ratio = upper_area_threshold_ratio
        self.ellipse_data = []

    def filter_ellipses(self, detections):
        for mask in detections.mask:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and len(contours[0]) > 5:
                ellipse = cv2.fitEllipse(contours[0])
                mask_area = np.sum(mask)
                ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4.0
                area_ratio = mask_area / ellipse_area
                if self.lower_area_threshold_ratio <= area_ratio <= self.upper_area_threshold_ratio:
                    x, y = ellipse[0]
                    major_axis = ellipse[1][0]
                    minor_axis = ellipse[1][1]
                    angle = ellipse[2]
                    self.ellipse_data.append({'x': x, 'y': y, 'majorAxis': major_axis, 'minorAxis': minor_axis, 'angle': angle})
        return pd.DataFrame(self.ellipse_data)
