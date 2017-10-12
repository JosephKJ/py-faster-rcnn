# --------------------------------------------------------
# Licensed under The MIT License
# Written by Joseph K J
# cs17mtech01001@iith.ac.in
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
import matplotlib.pyplot as plt
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = False

class FilterLayer(caffe.Layer):
    """
    Filter thr RPNs and do back ground-subtraction on them.
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self._spatial_scale = layer_params.get('spatial_scale', 0.0625)
        self._threshold_intercept = layer_params.get('threshold_intercept', 0)
        self._active_pixel_count = layer_params.get('active_pixel_count', 10)

        top[0].reshape(*(bottom[0].data.shape))
        top[1].reshape(*(bottom[1].data.shape))

    def forward(self, bottom, top):

        roi_count, _ = bottom[1].data.shape

        feature_map = bottom[0].data[0]  # Only one image per batch
        rois = bottom[1].data

        # Generating the mask
        feature_sum = np.sum(feature_map, axis=0)
        feature_sum_normalized = (255 * (feature_sum - np.min(feature_sum)) / np.ptp(feature_sum)).astype(int)
        threshold = feature_sum_normalized.mean() + self._threshold_intercept
        binary_map = np.where(feature_sum_normalized > threshold, 1, 0)

        # Removing the background
        feature_map_filtered = feature_map * binary_map

        # Removing umwanted ROIs
        rejected_index = []
        for index, roi in enumerate (rois):
            x1 = int(np.round(roi[1] * self._spatial_scale))
            y1 = int(np.round(roi[2] * self._spatial_scale))
            x2 = int(np.round(roi[3] * self._spatial_scale))
            y2 = int(np.round(roi[4] * self._spatial_scale))

            patch = binary_map[x1:x2, y1:y2]
            if np.count_nonzero(patch) <= self._active_pixel_count:
                rois[index][3] = rois[index][1]
                rois[index][4] = rois[index][2]
                rejected_index.append(index)

        # rois = np.delete(rois, rejected_index, axis=0)

        top[0].reshape(*(bottom[0].data.shape))
        top[0].data[...] = feature_map_filtered

        top[1].reshape(*(rois.shape))
        top[1].data[...] = rois

    def backward(self, top, propagate_down, bottom):
        """Passing the gradients for the conv layers as is."""
        bottom[0].diff[...] = top[0].diff[...]

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def display_image(self, image, display=False, filename='heatmap.png'):
        # plt.axis('off')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        plt.imshow(image)
        plt.savefig("/home/joseph/workspace/sdd-py-faster-rcnn/data/staging/"+filename)
        if display:
            plt.show()