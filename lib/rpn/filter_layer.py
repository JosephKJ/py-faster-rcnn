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
    Filter thr RPNs and fo back ground subtraction on them.
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        # self._feat_stride = layer_params['feat_stride']
        # anchor_scales = layer_params.get('scales', (8, 16, 32))
        pass

    def forward(self, bottom, top):
        print 'Inside FilterLayer:forward'

        print bottom[0].data.shape
        feature_sum = np.sum(bottom[0].data[0], axis=0)
        print feature_sum.shape
        self.display_image(feature_sum)

        print bottom[1].data.shape
        print bottom[1].data

        top[0].reshape(*(bottom[1].data.shape))
        top[0].data[...] = bottom[1].data
        print 'Done FilterLayer:forward'

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def display_image(self, image):
        # plt.axis('off')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        plt.imshow(image)
        plt.savefig("/home/joseph/workspace/sdd-py-faster-rcnn/data/staging/heatmap.png")
        # plt.show()