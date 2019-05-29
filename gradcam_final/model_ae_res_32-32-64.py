import numpy as np
import tensorflow as tf


class Model(object):

    def __init__(self, train_mode, batch_size = 1):        
        """
        Initialize the model hyper-parameters (and parameters).
        """

        self.batch_size = batch_size
        self.train_mode = train_mode

    def _variable_on_cpu(self, name, shape, initializer, trainable=None):
        """
         Helper to create a Variable stored on CPU memory.

         Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable

        Returns:
            Variable Tensor
        """

        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, 
                                dtype=dtype, trainable=trainable)

        return var


    def _batch_norm(self, input, name, decay = 0.9, is_fc = False):

        with tf.variable_scope(name) as scope:
            epsilon = 1e-3
            
            scale = self._variable_on_cpu(name+'_scale',
                                    [input.get_shape()[-1]],
                                    tf.ones_initializer)
            beta = self._variable_on_cpu(name+'_beta',
                                    [input.get_shape()[-1]],
                                    tf.zeros_initializer)
            pop_mean = self._variable_on_cpu(name+'_mean',
                                        [input.get_shape()[-1]],
                                        tf.zeros_initializer,
                                        trainable=False)
            pop_var = self._variable_on_cpu(name+'_var',
                                        [input.get_shape()[-1]],
                                        tf.ones_initializer,
                                        trainable=False)

            def update_mean_var():
                if is_fc:
                    batch_mean, batch_var = tf.nn.moments(input,[0])
                else:
                    batch_mean, batch_var = tf.nn.moments(input,[0,1,2])
                train_mean = tf.assign(pop_mean, 
                                        pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var, 
                                        pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(input, batch_mean, 
                                                        batch_var, beta, 
                                                        scale, epsilon, name=scope.name)     

            return tf.cond(self.train_mode, update_mean_var, 
                            lambda: tf.nn.batch_normalization(input, pop_mean, pop_var, 
                                                                beta, scale, epsilon, name=scope.name))


    def _deconv_block(self, input, in_channels, out_channels, name,
                        filter_size = 3, s_size = 1, padding_mode = 'SAME', 
                        bn_relu = True, stddev = 0.01, dtype = tf.float32):
        with tf.variable_scope(name) as scope:
            initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
            kernel = self._variable_on_cpu('weights', shape = [filter_size, filter_size, out_channels, in_channels], 
                                            initializer = initializer)

            in_shape = tf.shape(input)
            out_h = in_shape[1] * s_size
            out_w = in_shape[2] * s_size
            out_shape = [in_shape[0], out_h, out_w, out_channels]
            
            deconv = tf.nn.conv2d_transpose(input, kernel, out_shape, [1, s_size, s_size, 1],
                                            padding=padding_mode)
            if bn_relu:
                deconv_bn = self._batch_norm(deconv, scope.name+'_bn')
                relu = tf.nn.relu(deconv_bn, name = scope.name+'_relu')
                return relu
            else:
                return deconv


    def _conv_block(self, input, in_channels, out_channels, name,
                    filter_size = 3, s_size = 1, padding_mode = 'SAME', 
                    bn_relu = True, stddev = 0.01, dtype = tf.float32):
        with tf.variable_scope(name) as scope:
            initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
            kernel = self._variable_on_cpu('weights', shape = [filter_size, filter_size, in_channels, out_channels], 
                                            initializer = initializer)
            conv = tf.nn.conv2d(input, kernel, [1, s_size, s_size, 1], 
                                padding=padding_mode)
            if bn_relu:
                conv_bn = self._batch_norm(conv, scope.name+'_bn')
                relu = tf.nn.relu(conv_bn, name = scope.name+'_relu')
                return relu
            else:
                return conv            


    def _res_block(self, input, in_channels, out_channels, name,
                    filter_size = 3, s_size = 1, padding_mode = 'SAME', 
                    bn_relu = True, stddev = 0.01, dtype = tf.float32):
        assert in_channels == out_channels

        with tf.variable_scope(name) as scope:
            intermediate = self._conv_block(input, in_channels, out_channels, scope.name+'_conv1',
                                            filter_size=filter_size, s_size=s_size)
            initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
            kernel = self._variable_on_cpu('conv2_weights', shape = [filter_size, filter_size, in_channels, out_channels], 
                                            initializer = initializer)
            conv = tf.nn.conv2d(intermediate, kernel, [1, s_size, s_size, 1], 
                                padding=padding_mode)
            conv_bn = self._batch_norm(conv, scope.name+'_bn')
            conv_shortcut = tf.add(conv_bn, input)
            relu = tf.nn.relu(conv_shortcut, name = scope.name+'_relu')

            return relu


    def _fc_layer(self, input, in_size, out_size, name,
                    stddev = 0.01, dtype = tf.float32):
        with tf.variable_scope(name) as scope:
            reshape = tf.reshape(input, [-1, in_size])
            initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
            weights = self._variable_on_cpu('weights', [in_size, out_size], initializer = initializer)
            biases = self._variable_on_cpu('biases', [out_size], tf.constant_initializer(0.1))

            fc = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)

            return fc


    def _relu_layer(self, input, name):
        with tf.variable_scope(name) as scope:
            relu = tf.nn.relu(input, scope.name)

            return relu


    def _sigmoid_layer(self, input, name):
        with tf.variable_scope(name) as scope:
            sigmoid = tf.nn.sigmoid(input, scope.name)

            return sigmoid


    def _max_pool(self, input, name, k_size = 2, s_size = 1, padding_mode = 'SAME'):
        with tf.variable_scope(name) as scope:
            mp = tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], 
                                strides=[1, s_size, s_size, 1], 
                                padding=padding_mode, name=scope.name)
            return mp

    
    def build(self, input_x):
        """
        Build the model: loading model parameters from npy when given.

        Args:
            input_x: A tensor representing a batch of input images.
            train_mode: A boolean. Set to True when training the model.
        """

        #network_weights = self._initialize_weights()
        #self.weights = network_weights

        self.input_x = tf.identity(input_x, 'input_x_node')

        # model: encoder

        self.ecd_block1_conv = self._conv_block(self.input_x, 1, 2, "ecd_conv1",
                                            filter_size=7, s_size=2)
        self.ecd_block1_res = self._res_block(self.ecd_block1_conv, 2, 2, "ecd_res1",
                                            filter_size=7, s_size=1)

        self.ecd_block2_conv = self._conv_block(self.ecd_block1_res, 2, 4, "ecd_conv2",
                                            filter_size=5, s_size=2)
        self.ecd_block2_res = self._res_block(self.ecd_block2_conv, 4, 4, "ecd_res2",
                                            filter_size=5, s_size=1)

        self.ecd_block3_conv = self._conv_block(self.ecd_block2_res, 4, 8, "ecd_conv3",
                                            filter_size=3, s_size=2)
        self.ecd_block3_res = self._res_block(self.ecd_block3_conv, 8, 8, "ecd_res3",
                                            filter_size=3, s_size=1)

        self.ecd_block4_conv = self._conv_block(self.ecd_block3_res, 8, 16, "ecd_conv4",
                                            filter_size=3, s_size=2)
        self.ecd_block4_res = self._res_block(self.ecd_block4_conv, 16, 16, "ecd_res4",
                                            filter_size=3, s_size=1)

        self.ecd_block5_conv = self._conv_block(self.ecd_block4_res, 16, 32, "ecd_conv5",
                                            filter_size=3, s_size=2)
        self.ecd_block5_res = self._res_block(self.ecd_block5_conv, 32, 32, "ecd_res5",
                                            filter_size=3, s_size=1)

        self.ecd_block6_conv = self._conv_block(self.ecd_block5_res, 32, 64, "ecd_conv6",
                                            filter_size=3, s_size=2)
        self.ecd_block6_res = self._res_block(self.ecd_block6_conv, 64, 64, "ecd_res6",
                                            filter_size=3, s_size=1)

        # self.block7_conv = self._conv_block(self.block6_res, 64, 128, "ecd_conv7",
        #                                     filter_size=3, s_size=2)
        # self.block7_res = self._res_block(self.block7_conv, 128, 128, "ecd_res7",
        #                                     filter_size=3, s_size=1)

        # self.block8_conv = self._conv_block(self.block7_res, 128, 256, "ecd_conv8",
        #                                     filter_size=3, s_size=2)
        # self.block8_res = self._res_block(self.block8_conv, 256, 256, "ecd_res8",
        #                                     filter_size=3, s_size=1)

        # self.block9_conv = self._conv_block(self.block8_res, 256, 512, "ecd_conv9",
        #                                     filter_size=3, s_size=2)
        # self.block9_res = self._res_block(self.block9_conv, 512, 512, "ecd_res9",
        #                                     filter_size=3, s_size=1)

        self.z_mean = tf.identity(self.ecd_block6_res, "z_mean")    
        

        #model: classifier

        self.clf_block6_res = self._res_block(self.z_mean, 64, 64, "clf_res6",
                                            filter_size=3, s_size=1)

        self.clf_block5_mp = self._max_pool(self.clf_block6_res, "clf_mp5", k_size=2, s_size=2)

        self.clf_block4_conv = self._conv_block(self.clf_block5_mp, 64, 128, "clf_conv4",
                                            filter_size=3, s_size=2)
        self.clf_block4_res = self._res_block(self.clf_block4_conv, 128, 128, "clf_res4",
                                            filter_size=3, s_size=1)

        self.clf_block3_mp = self._max_pool(self.clf_block4_res, "clf_mp3", k_size=2, s_size=2)
        
        self.clf_block2_fc = self._fc_layer(self.clf_block3_mp, 2048, 256, "clf_fc2")
        self.clf_block2_bn = self._batch_norm(self.clf_block2_fc, 'clf_fc2_bn', is_fc = True)              
        self.clf_block2_relu = self._relu_layer(self.clf_block2_bn, 'clf_fc2_relu')

        self.clf_block1_fc = self._fc_layer(self.clf_block2_relu, 256, 3, "clf_fc1")
        self.clf_block1_bn = self._batch_norm(self.clf_block1_fc, 'clf_fc1_bn', is_fc = True)              
        self.clf_block1_sigmoid = self._sigmoid_layer(self.clf_block1_bn, 'clf_fc1_sigmoid')

        self.output_y = tf.identity(self.clf_block1_sigmoid, 'output_y_node')
        

        #model: decoder
        
        self.dcd_block6_deconv = self._deconv_block(self.z_mean, 64, 32, "dcd_deconv6_1",
                                                    filter_size=3, s_size=2)
        self.dcd_block5_deconv = self._deconv_block(self.dcd_block6_deconv, 32, 16, "dcd_deconv5_1",
                                                    filter_size=3, s_size=2)
        self.dcd_block4_deconv = self._deconv_block(self.dcd_block5_deconv, 16, 8, "dcd_deconv4_1",
                                                    filter_size=3, s_size=2)
        self.dcd_block3_deconv = self._deconv_block(self.dcd_block4_deconv, 8, 4, "dcd_deconv3_1",
                                                    filter_size=3, s_size=2)
        self.dcd_block2_deconv = self._deconv_block(self.dcd_block3_deconv, 4, 2, "dcd_deconv2_1",
                                                    filter_size=5, s_size=2)
        self.dcd_block1_deconv = self._deconv_block(self.dcd_block2_deconv, 2, 1, "dcd_deconv1_1",
                                                    filter_size=7, s_size=2, bn_relu = False)
        self.dcd_block1_bn = self._batch_norm(self.dcd_block1_deconv, 'dcd_bn1_2')
        self.dcd_block1_sigmoid = tf.sigmoid(self.dcd_block1_bn, 'dcd_sigmoid1_3')
        
        self.output_x = tf.reshape(self.dcd_block1_sigmoid, [-1, 2048, 2048, 1], name='output_x_node')


        # All the pre-trained parameters have been loaded.            

    # def avg_pool(self, bottom, name, k_size = 2, s_size = 1):
    #     return tf.nn.avg_pool(bottom, ksize=[1, k_size, k_size, 1], 
    #                             strides=[1, s_size, s_size, 1], 
    #                             padding='SAME', name=name)

    # def max_pool(self, bottom, name, k_size = 2, s_size = 1):
    #     return tf.nn.max_pool(bottom, ksize=[1, k_size, k_size, 1], 
    #                             strides=[1, s_size, s_size, 1], 
    #                             padding='SAME', name=name)

    # def batch_norm(self, bottom, name, train_mode, decay = 0.9, 
    #                 is_fc = False):
    #     epsilon = 1e-3
        
    #     if self.data_dict is not None and name in self.data_dict:
    #         scale_value = self.data_dict[name][0]
    #         beta_value = self.data_dict[name][1]
    #         pop_mean_value = self.data_dict[name][2]
    #         pop_var_value = self.data_dict[name][3]
    #     else:
    #         scale_value = tf.ones([bottom.get_shape()[-1]])
    #         beta_value = tf.zeros([bottom.get_shape()[-1]])
    #         pop_mean_value = tf.zeros([bottom.get_shape()[-1]])
    #         pop_var_value = tf.ones([bottom.get_shape()[-1]])            

    #     if self.trainable:
    #         scale = tf.Variable(scale_value, name=name+'_scale')
    #         beta = tf.Variable(beta_value, name=name+'_beta')
    #     else:
    #         scale = tf.constant(scale_value, dtype=tf.float32, 
    #                             name=name+'_scale')
    #         beta = tf.constant(beta_value, dtype=tf.float32, 
    #                             name=name+'_beta')

    #     pop_mean = tf.Variable(pop_mean_value, trainable=False, 
    #                             name=name+'_mean')
    #     pop_var = tf.Variable(pop_var_value, trainable=False, 
    #                             name=name+'_var')

    #     self.var_dict[(name, 0)] = scale
    #     self.var_dict[(name, 1)] = beta
    #     self.var_dict[(name, 2)] = pop_mean
    #     self.var_dict[(name, 3)] = pop_var

    #     def update_mean_var():
    #         if is_fc:
    #             batch_mean, batch_var = tf.nn.moments(bottom,[0])
    #         else:
    #             batch_mean, batch_var = tf.nn.moments(bottom,[0,1,2])
    #         train_mean = tf.assign(pop_mean, 
    #                             pop_mean * decay + batch_mean * (1 - decay))
    #         train_var = tf.assign(pop_var, 
    #                             pop_var * decay + batch_var * (1 - decay))
    #         with tf.control_dependencies([train_mean, train_var]):
    #             return tf.nn.batch_normalization(bottom, batch_mean, 
    #                                                 batch_var, beta, 
    #                                                 scale, epsilon)     

    #     return tf.cond(train_mode, update_mean_var, 
    #                     lambda: tf.nn.batch_normalization(bottom, pop_mean, 
    #                                                         pop_var, beta, 
    #                                                         scale, epsilon))

    # def conv_block(self, bottom, in_channels, out_channels, name, train_mode,
    #                 filter_size = 3, s_size = 1, padding_mode = 'SAME', 
    #                 pre_trained = True, bn_relu = True):
    #     with tf.variable_scope(name):
    #         filt = self.get_conv_var(filter_size, in_channels, out_channels, 
    #                                     name, pre_trained = pre_trained)

    #         conv = tf.nn.conv2d(bottom, filt, [1, s_size, s_size, 1], 
    #                             padding=padding_mode)
    #         if bn_relu:
    #             conv_bn = self.batch_norm(conv, name+'_bn', train_mode)
    #             relu = tf.nn.relu(conv_bn)
    #             return relu
    #         else:
    #             return conv

    # def deconv_block(self, bottom, in_channels, out_channels, name, train_mode,
    #                 filter_size = 3, s_size = 1, padding_mode = 'SAME', 
    #                 pre_trained = True, bn_relu = True):
    #     with tf.variable_scope(name):
    #         filt = self.get_conv_var(filter_size, out_channels, in_channels, 
    #                                     name, pre_trained = pre_trained)

    #         in_shape = tf.shape(bottom)
    #         out_h = in_shape[1] * s_size
    #         out_w = in_shape[2] * s_size
    #         out_shape = [in_shape[0], out_h, out_w, out_channels]
            
    #         deconv = tf.nn.conv2d_transpose(bottom, filt, out_shape, [1, s_size, s_size, 1],
    #                                         padding=padding_mode)
    #         if bn_relu:
    #             deconv_bn = self.batch_norm(deconv, name+'_bn', train_mode)
    #             relu = tf.nn.relu(deconv_bn)
    #             return relu
    #         else:
    #             return deconv

    # def fc_layer(self, bottom, in_size, out_size, name, pre_trained = True):
    #     with tf.variable_scope(name):
    #         weights, biases = self.get_fc_var(in_size, out_size, name, 
    #                                             pre_trained = pre_trained)

    #         x = tf.reshape(bottom, [-1, in_size])
    #         fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    #         return fc

    # def res_block(self, bottom, in_channels, out_channels, name1, name2, 
    #                 train_mode, filter_size = 3, s_size = 1, 
    #                 padding_mode = 'SAME', pre_trained = True):
    #     assert in_channels == out_channels

    #     intermediate = self.conv_block(bottom, in_channels, out_channels, 
    #                                     name1, train_mode, 
    #                                     filter_size=filter_size, 
    #                                     padding_mode = padding_mode,  
    #                                     pre_trained = pre_trained)

    #     with tf.variable_scope(name2):
    #         filt = self.get_conv_var(filter_size, in_channels, out_channels, 
    #                                     name2, pre_trained = pre_trained)

    #         conv = tf.nn.conv2d(intermediate, filt, [1, s_size, s_size, 1], 
    #                             padding=padding_mode)
    #         conv_bn = self.batch_norm(conv, name2+'_bn', train_mode)
    #         conv_shortcut = tf.add(conv_bn, bottom)
    #         relu = tf.nn.relu(conv_shortcut)

    #         return relu            


    # def get_conv_var(self, filter_size, in_channels, out_channels, name, 
    #     pre_trained = True):
    #     initial_value = tf.truncated_normal([filter_size, filter_size,
    #                                         in_channels, out_channels], 
    #                                         0, 0.01)
    #     filters = self.get_var(initial_value, name, 0, name + "_filters", 
    #                             pre_trained = pre_trained)

    #     return filters

    # def get_fc_var(self, in_size, out_size, name, pre_trained = True):
    #     initial_value = tf.truncated_normal([in_size, out_size], 0, 0.01)
    #     weights = self.get_var(initial_value, name, 0, name + "_weights", 
    #                             pre_trained = pre_trained)

    #     initial_value = tf.truncated_normal([out_size], 0, 0.01)
    #     biases = self.get_var(initial_value, name, 1, name + "_biases",
    #                             pre_trained = pre_trained)

    #     return weights, biases

    # def get_var(self, initial_value, name, idx, var_name, pre_trained = True):
    #     if (
    #         self.data_dict is not None 
    #         and name in self.data_dict 
    #         and pre_trained
    #     ):
    #         value = self.data_dict[name][idx]
    #     else:
    #         value = initial_value

    #     if self.trainable:
    #         var = tf.Variable(value, name=var_name)
    #     else:
    #         var = tf.constant(value, dtype=tf.float32, name=var_name)

    #     self.var_dict[(name, idx)] = var

    #     # print var_name, var.get_shape().as_list()
    #     print(var_name)
    #     print(var.get_shape())
    #     print(initial_value.get_shape())
    #     assert var.get_shape() == initial_value.get_shape()

    #     return var