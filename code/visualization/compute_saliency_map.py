import tensorflow as tf
import utils
import numpy as np
import os
import skimage
import pylab
import sys
import importlib

def main(argv):

    model_name = argv[1]
    check_point = argv[2]
    dropout_rate = argv[3]
    cost_function = argv[4]
    gpu_num = argv[5]

    model_module = importlib.import_module(model_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    sess = tf.Session()

    height = 2048
    width = 2048
    binary = '0v1v2v3'

    images = tf.placeholder(tf.float32, [1, height, width, 1])
    true_out = tf.placeholder(tf.float32, [1, 3])
    train_mode = tf.placeholder(tf.bool)

    nn_model = model_module.Vgg19('/data/vision/polina/projects/chestxray/work_space/model_para/'+model_name+'_'+dropout_rate+'_'+cost_function+'/'+check_point+'.npy')
    nn_model.build(images, train_mode)
    saliency_dir1 = '/data/vision/polina/projects/chestxray/work_space/model_para/'+model_name+'_'+dropout_rate+'_'+cost_function+'/'+check_point+'_saliency_0v123'
    saliency_dir2 = '/data/vision/polina/projects/chestxray/work_space/model_para/'+model_name+'_'+dropout_rate+'_'+cost_function+'/'+check_point+'_saliency_01v23'
    saliency_dir3 = '/data/vision/polina/projects/chestxray/work_space/model_para/'+model_name+'_'+dropout_rate+'_'+cost_function+'/'+check_point+'_saliency_012v3'

    if not os.path.exists(saliency_dir1):
        os.makedirs(saliency_dir1)
    if not os.path.exists(saliency_dir2):
        os.makedirs(saliency_dir2)
    if not os.path.exists(saliency_dir3):
        os.makedirs(saliency_dir3)

    print(nn_model.get_var_count())

    var_grad1 = tf.gradients(nn_model.output[0][0], images)
    var_grad2 = tf.gradients(nn_model.output[0][1], images)
    var_grad3 = tf.gradients(nn_model.output[0][2], images)

    sess.run(tf.global_variables_initializer())

    test_img_dir = '/data/vision/polina/projects/chestxray/work_space/labels_after_segmentation_removingLineBreaks/classification/new_sub_v2/resized_'+str(height)+'_'+str(width)+'/'+binary+'/test_resized/'
    for img_name in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_name)
        saliency_path1 = os.path.join(saliency_dir1, img_name)
        saliency_path2 = os.path.join(saliency_dir2, img_name)
        saliency_path3 = os.path.join(saliency_dir3, img_name)
        img = np.load(img_path)
        test_img = img.reshape(1, height, width, 1)
        var_grad_val1 = sess.run(var_grad1, feed_dict={images: test_img, train_mode: False})
        var_grad_val1 = np.array(var_grad_val1)
        np.save(saliency_path1, var_grad_val1)
        var_grad_val2 = sess.run(var_grad2, feed_dict={images: test_img, train_mode: False})
        var_grad_val2 = np.array(var_grad_val2)
        np.save(saliency_path2, var_grad_val2)
        var_grad_val3 = sess.run(var_grad3, feed_dict={images: test_img, train_mode: False})
        var_grad_val3 = np.array(var_grad_val3)
        np.save(saliency_path3, var_grad_val3)
        print(img_name)

if __name__ == '__main__':
    main(argv=sys.argv)