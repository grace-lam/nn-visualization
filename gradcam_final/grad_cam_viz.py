import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pydicom
from grad_cam import GradCam
import importlib

class GradCamViz():
    def __init__(self, graph, sess, output_tensor, input_tensor, viz_dir):
        self.graph = graph
        self.sess = sess
        self.output_tensor = output_tensor
        self.input_tensor = input_tensor
        self.viz_dir = viz_dir
        
    def normalize_image(self, img, beta):
        norm_img = np.copy(img)
        A = np.percentile(img, 95)*1.05
        B = beta*A
        for r in range(len(img)):
            for c in range(len(img[r])):
                if img[r][c] < B:
                    norm_img[r][c] = 0
                else:
                    norm_img[r][c] = img[r][c] / A
        return norm_img
        
    def generate_viz(self, image, layer, output_neuron_index, viz_file_name='grad_cam', feed_dict={}, a=0.25, b=0.5):
        # compute grad_cam
        inp_img = np.expand_dims(image, 2)
        grad_cam = GradCam(self.graph, self.sess, self.output_tensor[0][output_neuron_index], self.input_tensor, layer)
        grad_cam_mask = grad_cam.GetMask(inp_img, feed_dict, should_resize = True, three_dims = False)

        # render xray image
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        # render grad_cam image
        norm_img = self.normalize_image(grad_cam_mask, b)
        norm_img = np.ma.masked_where(norm_img == 0, norm_img)
        plt.imshow(norm_img, cmap=plt.cm.jet, alpha=a, vmin=-1, vmax=1)

        # save overlay image
        file_path = os.path.join(self.viz_dir, viz_file_name+'.png')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)