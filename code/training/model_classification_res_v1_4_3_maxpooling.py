import tensorflow as tf
import numpy as np
from functools import reduce

# Update this for every new dataset!!
XRAY_MEAN = 2189.7 

class Model:
    """
    Base class for building a residual CNN model.
    """

    def __init__(self, model_npy_path=None, trainable=True, dropout=0.0):
        """
        Initialize the model hyper-parameters (and parameters).
        """

        if model_npy_path is not None:
            self.data_dict = np.load(model_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build(self, xray, train_mode=None):
        """
        Build the model: loading model parameters from npy when given.

        Args:
            xray: A tensor representing a batch of input images.
            train_mode: A boolean. Set to True when training the model.
        """

        print('xray shape: ', xray.get_shape())
        new_xray = tf.identity((xray - XRAY_MEAN), 'input_node')

        self.block1_conv = self.conv_block(new_xray, 1, 16, "conv1_1",
                                            train_mode, filter_size=7,
                                            s_size=2, padding_mode = 'SAME')
        self.block2_res = self.res_block(self.block1_conv, 16, 16, 
                                            "res2_1", "res2_2", train_mode, 
                                            filter_size=5, s_size=1, 
                                            padding_mode = 'SAME')
        self.block3_max_pool = self.max_pool(self.block2_res, "max_pool3",
                                                k_size = 2, s_size = 2)

        self.block4_conv = self.conv_block(self.block3_max_pool, 16, 32, 
                                            "conv4_1", train_mode, 
                                            filter_size=3, s_size=2, 
                                            padding_mode = 'SAME')
        self.block5_res = self.res_block(self.block4_conv, 32, 32, 
                                            "res5_1", "res5_2", train_mode, 
                                            filter_size=3, s_size=1, 
                                            padding_mode = 'SAME')
        self.block6_max_pool = self.max_pool(self.block5_res, "max_pool6",
                                                k_size = 2, s_size = 2)

        self.block7_conv = self.conv_block(self.block6_max_pool, 32, 64, 
                                            "conv7_1", train_mode, 
                                            filter_size=3, s_size=2, 
                                            padding_mode = 'SAME')
        self.block8_res = self.res_block(self.block7_conv, 64, 64, 
                                            "res8_1", "res8_2", train_mode, 
                                            filter_size=3, s_size=1, 
                                            padding_mode = 'SAME')
        self.block9_max_pool = self.max_pool(self.block8_res, "max_pool9",
                                                k_size = 2, s_size = 2)

        self.block10_conv = self.conv_block(self.block9_max_pool, 64, 128, 
                                            "conv10_1", train_mode, 
                                            filter_size=3, s_size=2, 
                                            padding_mode = 'SAME')
        self.block11_res = self.res_block(self.block10_conv, 128, 128, 
                                            "res11_1", "res11_2", train_mode, 
                                            filter_size=3, s_size=1, 
                                            padding_mode = 'SAME')
        self.block12_max_pool = self.max_pool(self.block11_res, "max_pool12",
                                                k_size = 4, s_size = 4)

        self.block13_fc = self.fc_layer(self.block12_max_pool, 2048, 256,
                                        "fc13") 
        self.block13_bn = self.batch_norm(self.block13_fc, 'fc13_bn', 
                                            train_mode, is_fc = True)              
        self.block13_relu = tf.nn.relu(self.block13_bn)

        self.fc14 = self.fc_layer(self.block13_relu, 256, 3, "fc14")
        self.fc14_bn = self.batch_norm(self.fc14, 'fc14_bn', train_mode, 
                                        is_fc = True)
        self.output = tf.sigmoid(self.fc14_bn, name='output_node')

        # All the pre-trained parameters have been loaded.
        self.data_dict = None

    def avg_pool(self, bottom, name, k_size = 2, s_size = 1):
        return tf.nn.avg_pool(bottom, ksize=[1, k_size, k_size, 1], 
                                strides=[1, s_size, s_size, 1], 
                                padding='SAME', name=name)

    def max_pool(self, bottom, name, k_size = 2, s_size = 1):
        return tf.nn.max_pool(bottom, ksize=[1, k_size, k_size, 1], 
                                strides=[1, s_size, s_size, 1], 
                                padding='SAME', name=name)

    def batch_norm(self, bottom, name, train_mode, decay = 0.9, 
                    is_fc = False):
        epsilon = 1e-3
        
        if self.data_dict is not None and name in self.data_dict:
            scale_value = self.data_dict[name][0]
            beta_value = self.data_dict[name][1]
            pop_mean_value = self.data_dict[name][2]
            pop_var_value = self.data_dict[name][3]
        else:
            scale_value = tf.ones([bottom.get_shape()[-1]])
            beta_value = tf.zeros([bottom.get_shape()[-1]])
            pop_mean_value = tf.zeros([bottom.get_shape()[-1]])
            pop_var_value = tf.ones([bottom.get_shape()[-1]])            

        if self.trainable:
            scale = tf.Variable(scale_value, name=name+'_scale')
            beta = tf.Variable(beta_value, name=name+'_beta')
        else:
            scale = tf.constant(scale_value, dtype=tf.float32, 
                                name=name+'_scale')
            beta = tf.constant(beta_value, dtype=tf.float32, 
                                name=name+'_beta')

        pop_mean = tf.Variable(pop_mean_value, trainable=False, 
                                name=name+'_mean')
        pop_var = tf.Variable(pop_var_value, trainable=False, 
                                name=name+'_var')

        self.var_dict[(name, 0)] = scale
        self.var_dict[(name, 1)] = beta
        self.var_dict[(name, 2)] = pop_mean
        self.var_dict[(name, 3)] = pop_var

        def update_mean_var():
            if is_fc:
                batch_mean, batch_var = tf.nn.moments(bottom,[0])
            else:
                batch_mean, batch_var = tf.nn.moments(bottom,[0,1,2])
            train_mean = tf.assign(pop_mean, 
                                pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, 
                                pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(bottom, batch_mean, 
                                                    batch_var, beta, 
                                                    scale, epsilon)     

        return tf.cond(train_mode, update_mean_var, 
                        lambda: tf.nn.batch_normalization(bottom, pop_mean, 
                                                            pop_var, beta, 
                                                            scale, epsilon))

    def conv_block(self, bottom, in_channels, out_channels, name, train_mode,
                    filter_size = 3, s_size = 1, padding_mode = 'VALID', 
                    pre_trained = True, bn_relu = True):
        with tf.variable_scope(name):
            filt = self.get_conv_var(filter_size, in_channels, out_channels, 
                                        name, pre_trained = pre_trained)

            conv = tf.nn.conv2d(bottom, filt, [1, s_size, s_size, 1], 
                                padding=padding_mode)
            if bn_relu:
                conv_bn = self.batch_norm(conv, name+'_bn', train_mode)
                relu = tf.nn.relu(conv_bn)
                return relu
            else:
                return conv

    def fc_layer(self, bottom, in_size, out_size, name, pre_trained = True):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, 
                                                pre_trained = pre_trained)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def res_block(self, bottom, in_channels, out_channels, name1, name2, 
                    train_mode, filter_size = 3, s_size = 1, 
                    padding_mode = 'VALID', pre_trained = True):
        assert in_channels == out_channels

        intermediate = self.conv_block(bottom, in_channels, out_channels, 
                                        name1, train_mode, 
                                        filter_size=filter_size, 
                                        padding_mode = padding_mode,  
                                        pre_trained = pre_trained)

        with tf.variable_scope(name2):
            filt = self.get_conv_var(filter_size, in_channels, out_channels, 
                                        name2, pre_trained = pre_trained)

            conv = tf.nn.conv2d(intermediate, filt, [1, s_size, s_size, 1], 
                                padding=padding_mode)
            conv_bn = self.batch_norm(conv, name2+'_bn', train_mode)
            conv_shortcut = tf.add(conv_bn, bottom)
            relu = tf.nn.relu(conv_shortcut)

            return relu            


    def get_conv_var(self, filter_size, in_channels, out_channels, name, 
        pre_trained = True):
        initial_value = tf.truncated_normal([filter_size, filter_size,
                                            in_channels, out_channels], 
                                            0, 0.01)
        filters = self.get_var(initial_value, name, 0, name + "_filters", 
                                pre_trained = pre_trained)

        return filters

    def get_fc_var(self, in_size, out_size, name, pre_trained = True):
        initial_value = tf.truncated_normal([in_size, out_size], 0, 0.01)
        weights = self.get_var(initial_value, name, 0, name + "_weights", 
                                pre_trained = pre_trained)

        initial_value = tf.truncated_normal([out_size], 0, 0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases",
                                pre_trained = pre_trained)

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, pre_trained = True):
        if (
            self.data_dict is not None 
            and name in self.data_dict 
            and pre_trained
        ):
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        print(var_name)
        print(var.get_shape())
        print(initial_value.get_shape())
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./model-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
