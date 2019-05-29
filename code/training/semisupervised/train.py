
# coding: utf-8

# ## Import packages and neural network model for semisupervised training

# In[4]:


import numpy as np
import os
from math import floor, ceil
import importlib

import tensorflow as tf
import keras
import keras.backend as K

from keras.layers import Input, Lambda
from keras.models import Model

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

from sklearn.metrics import roc_auc_score


# ## Data list and statistics
# 

# In[5]:


import csv

label_list_dir = '/data/vision/polina/projects/chestxray/work_space'                 '/labels_after_segmentation_removingLineBreaks/sub_img_v2/'
train_label_list = os.path.join(label_list_dir, 'yesHF_newSub_labelled.csv')
val_label_list = os.path.join(label_list_dir, 'yesHF_oldSub_val_labelled.csv')
train_yesHF_unlabeled_list = os.path.join(label_list_dir, 'yesHF_newSub_unlabelled.csv')
train_noHF_unlabeled_list = os.path.join(label_list_dir, 'noHF_newSub.csv')

train_labels = {}
train_labeled_list_IDs = []
train_yesHF_unlabeled_list_IDs = []
train_noHF_unlabeled_list_IDs = []
val_labels = {}
val_list_IDs = []

with open(train_label_list, 'r') as train_label_file:
    train_label_file_reader = csv.reader(train_label_file, delimiter = ',')
    row = next(train_label_file_reader)
    for row in train_label_file_reader:
        train_labeled_list_IDs.append(row[2])
        train_labels[row[2]] = row[6]
with open(train_yesHF_unlabeled_list, 'r') as train_yesHF_unlabeled_file:
    train_yesHF_unlabeled_file_reader = csv.reader(train_yesHF_unlabeled_file, 
                                                   delimiter = ',')
    row = next(train_yesHF_unlabeled_file_reader)
    for row in train_yesHF_unlabeled_file_reader:
        train_yesHF_unlabeled_list_IDs.append(row[2])
with open(train_noHF_unlabeled_list, 'r') as train_noHF_unlabeled_file:
    train_noHF_unlabeled_file_reader = csv.reader(train_noHF_unlabeled_file, 
                                                  delimiter = ',')
    row = next(train_noHF_unlabeled_file_reader)
    for row in train_noHF_unlabeled_file_reader:
        train_noHF_unlabeled_list_IDs.append(row[2])
with open(val_label_list, 'r') as val_label_file:
    val_label_file_reader = csv.reader(val_label_file, delimiter = ',')
    row = next(val_label_file_reader)
    for row in val_label_file_reader:
        val_list_IDs.append(row[2])
        val_labels[row[2]] = row[6]

count_class0 = 0
count_class1 = 0
count_class2 = 0
count_class3 = 0
for _, label in train_labels.items():
    if label == '0':
        count_class0 += 1
    if label == '1':
        count_class1 += 1
    if label == '2':
        count_class2 += 1
    if label == '3':
        count_class3 += 1
        
train_unlabeled_list_IDs = train_yesHF_unlabeled_list_IDs+train_noHF_unlabeled_list_IDs

print(len(train_labeled_list_IDs))
print(len(train_labels))
print(len(val_list_IDs))
print(len(val_labels))
print(len(train_yesHF_unlabeled_list_IDs))
print(len(train_noHF_unlabeled_list_IDs))
print(len(train_unlabeled_list_IDs))


# ## Learning rate schedulers

# In[6]:


def lr_schedule_wrap(init_lr=0.005):

    def lr_schedule(epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 100, 150, 200, 250 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        nonlocal init_lr

        lr = init_lr
        if epoch > 50:
            lr *= 0.5e-3
        elif epoch > 40:
            lr *= 1e-3
        elif epoch > 30:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    return lr_schedule


# ## Custom loss functions and metrics
# 

# In[7]:


def custom_binary_crossentropy(loss_weight):
    
    def loss(y_true, y_pred):
        loss_tensor = K.binary_crossentropy(y_true, y_pred)
        return K.mean(loss_tensor)*loss_weight
    
    return loss

def weighted_crossentropy(y_true, y_pred):
    count_total = count_class0+count_class1+count_class2+count_class3
    count_bit1 = count_class1+count_class2+count_class3
    count_bit2 = count_class2+count_class3
    count_bit3 = count_class3

    bit1_one_weight = float(count_total/count_bit1)
    bit1_zero_weight = float(count_total/(count_total-count_bit1))
    bit2_one_weight = float(count_total/count_bit2)
    bit2_zero_weight = float(count_total/(count_total-count_bit2))
    bit3_one_weight = float(count_total/count_bit3)
    bit3_zero_weight = float(count_total/(count_total-count_bit3))

    bit1_weight_batch = y_true[:,0]*bit1_one_weight+(1-y_true[:,0])*bit1_zero_weight
    bit2_weight_batch = y_true[:,1]*bit2_one_weight+(1-y_true[:,1])*bit2_zero_weight
    bit3_weight_batch = y_true[:,2]*bit3_one_weight+(1-y_true[:,2])*bit3_zero_weight

    bit1_binary_ce = K.binary_crossentropy(y_true[:,0], y_pred[:,0])
    bit2_binary_ce = K.binary_crossentropy(y_true[:,1], y_pred[:,1])
    bit3_binary_ce = K.binary_crossentropy(y_true[:,2], y_pred[:,2])

    bit1_weighted_ce = bit1_binary_ce*bit1_weight_batch
    bit2_weighted_ce = bit2_binary_ce*bit2_weight_batch
    bit3_weighted_ce = bit3_binary_ce*bit3_weight_batch

    return K.mean(K.concatenate([bit1_weighted_ce, bit2_weighted_ce, bit3_weighted_ce]))
    
def abs_error(y_true, y_pred):
    y_true = K.sum(y_true, axis=1)
    y_pred = K.sum(y_pred, axis=1)
    return K.mean(K.abs(y_true-y_pred))

def kl_loss(y_true, y_pred):
    batch = K.shape(y_pred)[0]
    dim = int(K.int_shape(y_pred)[1]/2)
    z_mean = y_pred[:, 0:dim]
    z_log_var = y_pred[:, dim:dim*2]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    kl_loss //= float(dim)
    
    return K.mean(kl_loss)


# ## A data generator
# 

# In[8]:


import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

class XRayDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, data_dir = '../data/', batch_size=4, 
                 dim=(2048, 2048), n_channels=1, n_classes=3, shuffle=True,
                 shift_mean=0, shift_std=100, rotation_mean=0, rotation_std=15,
                 is_training=True, latent_dim=16384, is_semisupervised=False,
                 unlabeled_list_IDs=None, labeled_list_IDs=None):
        'Initialization'
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.shift_mean = shift_mean
        self.shift_std = shift_std
        self.rotation_mean = rotation_mean
        self.rotation_std = rotation_std
        self.is_training = is_training
        self.latent_dim = latent_dim
        self.epoch = 0
        self.is_semisupervised = is_semisupervised
        if self.is_semisupervised:
            self.unlabeled_list_IDs = unlabeled_list_IDs
            self.labeled_list_IDs = labeled_list_IDs       
            self.unlabeled_subset_i = 0
            self.labeledset_len = len(labeled_list_IDs)
        
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
                               min(len(self.list_IDs), (index+1)*self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):    
        'Update indexes after each epoch'
        self.epoch += 1
        
        if self.is_semisupervised:
            if self.epoch % 2 == 0:
                unlabeled_start_index = 5*self.unlabeled_subset_i*self.labeledset_len
                unlabeled_end_index = 5*(self.unlabeled_subset_i+1)*self.labeledset_len
                if unlabeled_end_index>len(self.unlabeled_list_IDs):
                    unlabeled_end_index=len(self.unlabeled_list_IDs)
                    self.unlabeled_subset_i = -1
                
                self.list_IDs = self.unlabeled_list_IDs[unlabeled_start_index:
                                                        unlabeled_end_index]
                self.unlabeled_subset_i += 1
            else:
                self.list_IDs = self.labeled_list_IDs+self.labeled_list_IDs                                +self.labeled_list_IDs+self.labeled_list_IDs                                +self.labeled_list_IDs
                
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def translate_2Dimage(self, image, output_shape):
        'Translate 2D images as data augmentation'
        input_shape = np.shape(image)
        
        # Generate random Gaussian numbers for image shift as data augmentation
        shift_1 = int(np.random.normal(self.shift_mean, self.shift_std))
        shift_2 = int(np.random.normal(self.shift_mean, self.shift_std))
        if abs(shift_1)>2*self.shift_std:
            shift_1 = 0
        if abs(shift_2)>2*self.shift_std:
            shift_2 = 0
            
        # Pad the 2D image
        pad_1_length = max(0, float(output_shape[0]-input_shape[0]))
        pad_1_length_1 = floor(pad_1_length/2)+4 # 4 is extra padding
        pad_1_length_2 = floor(pad_1_length/2)+4 # 4 is extra padding       
        if shift_1>0:
            pad_1_length_1 = pad_1_length_1+abs(shift_1)
        else:
            pad_1_length_2 = pad_1_length_2+abs(shift_1)
        image = np.pad(image, ((pad_1_length_1, pad_1_length_2), (0, 0)), 
                       'constant', constant_values=((0, 0), (0, 0)))

        pad_2_length = max(0, float(output_shape[1]-input_shape[1]))
        pad_2_length_1 = floor(pad_2_length/2)+4 # 4 is extra padding
        pad_2_length_2 = floor(pad_2_length/2)+4 # 4 is extra padding
        if shift_2>0:
            pad_2_length_1 = pad_2_length_1+abs(shift_2)
        else:
            pad_2_length_2 = pad_2_length_2+abs(shift_2)
        image = np.pad(image, ((0, 0), (pad_2_length_1, pad_2_length_2)), 
                       'constant', constant_values=((0, 0), (0, 0)))
            
        return image
    
    def rotate_2Dimage(self, image):
        'Rotate 2D images as data augmentation'
        
        # Generate a random Gaussian number for image rotation angle as data augmentation
        angle = np.random.normal(self.rotation_mean, self.rotation_std)
        if abs(angle)>2*self.rotation_std:
            angle = 0
        return ndimage.rotate(image, angle)
    
    def pad_2Dimage(self, image, output_shape):
        'Pad 2D images to match output_shape'
        input_shape = np.shape(image)
        # Pad the 2D image
        pad_1_length = max(0, float(output_shape[0]-input_shape[0]))
        pad_1_length_1 = floor(pad_1_length/2)+4 # 4 is extra padding
        pad_1_length_2 = floor(pad_1_length/2)+4 # 4 is extra padding       
        image = np.pad(image, ((pad_1_length_1, pad_1_length_2), (0, 0)), 
                       'constant', constant_values=((0, 0), (0, 0)))

        pad_2_length = max(0, float(output_shape[1]-input_shape[1]))
        pad_2_length_1 = floor(pad_2_length/2)+4 # 4 is extra padding
        pad_2_length_2 = floor(pad_2_length/2)+4 # 4 is extra padding
        image = np.pad(image, ((0, 0), (pad_2_length_1, pad_2_length_2)), 
                       'constant', constant_values=((0, 0), (0, 0)))
            
        return image
            
    def resample_2Dimage(self, image, output_shape):
        'Resample 2D images'
        
        if self.is_training:
            image = self.translate_2Dimage(image, output_shape)
            image = self.rotate_2Dimage(image)
        else:
            image = self.pad_2Dimage(image, output_shape)
        
        input_shape = np.shape(image)
        
        if input_shape[0]-output_shape[0]<0 or input_shape[1]-output_shape[1]<0:
            raise ValueError('This image needs to be padded!')
        
        start_index1 = floor((input_shape[0]-output_shape[0])/2)
        stop_index1 = start_index1+output_shape[0]
        start_index2 = floor((input_shape[1]-output_shape[1])/2)
        stop_index2 = start_index2+output_shape[1]
        return image[start_index1:stop_index1, start_index2:stop_index2]

    def __ordinal_encoding(self, label):
        if int(label) == 0:
            return [0, 0, 0]
        if int(label) == 1:
            return [1, 0, 0]
        if int(label) == 2:
            return [1, 1, 0]        
        if int(label) == 3:
            return [1, 1, 1]
        if int(label) == -1:
            return [-1, -1, -1]
        
        print(label)
        raise ValueError('label should be in {0,1,2,3}')
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_size = len(list_IDs_temp)
        X = np.empty((batch_size, *self.dim, self.n_channels))
        y = np.empty((batch_size, self.n_classes), dtype=int)
        dummy_zeros = np.zeros((batch_size, self.latent_dim*2)) 

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = plt.imread(self.data_dir + '/' + ID + '.png')
            img = self.resample_2Dimage(img, self.dim)
            img = np.reshape(img, [*self.dim, 1])
            X[i,] = img

            # Store class
            y[i,] = self.__ordinal_encoding(self.labels.get(ID, '-1'))
                        
        return X, {'decoder': X, 'classifier': y, 
                   'encoder_variational': dummy_zeros}


# ## Custom callbacks
# 

# In[9]:


class ChangeLossWeights(keras.callbacks.Callback):
    def __init__(self, classifier_weight, clf_weight):
        self.classifier_weight = classifier_weight
        self.clf_weight = clf_weight
        self.init_clf_weight = clf_weight
    
    def on_train_begin(self, logs={}):
        print('classifier_weight:', K.get_value(self.classifier_weight))
        
    def on_epoch_end(self, epoch, logs={}):
        if logs['val_encoder_variational_loss'] >= 1000:
            self.clf_weight = self.init_clf_weight
        if logs['val_encoder_variational_loss'] < 1000:
            self.clf_weight = 5*self.init_clf_weight
        if logs['val_encoder_variational_loss'] < 100:
            self.clf_weight = 10*self.init_clf_weight
        if logs['val_encoder_variational_loss'] < 10:
            self.clf_weight = 50*self.init_clf_weight
        if epoch % 2 == 0:
            K.set_value(self.classifier_weight, 0.0)
        else:
            K.set_value(self.classifier_weight, self.clf_weight)
        print('Epoch', epoch, 'ended!')
        print('------')
        print('Next epoch classifier_weight:', K.get_value(self.classifier_weight))


# ## Sampling method for variational approach

# In[10]:


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    batch = K.shape(args)[0]
    dim = int(K.int_shape(args)[1]/2)
    # by default, random_normal has mean=0 and std=1.0
    z_mean = args[:, 0:dim]
    z_log_var = args[:, dim:dim*2]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ## Set model hyper-parameters

# In[30]:

ae_weight = 10000

model_architecture = 'model_vae_16384_clf_8_8_64_512_64'
loss = 'KL_MSE_custom_binary'
init_lr = 0.001
kl_weight = 0.001*ae_weight
clf_weight = 0.1
decoder_weight = 1.0*ae_weight
latent_dim = 128*128
num_cpus = 20


# In[31]:


print(model_architecture)
print(loss)
print(str(init_lr))
print('kl_weight:', str(kl_weight))
print('clf_weight:', str(clf_weight))
print('decoder_weight:', str(decoder_weight))
print(str(latent_dim))

classifier_weight = K.variable(clf_weight)

if loss == 'binary_crossentropy':
    loss_flag = 'binary'
elif loss == 'weighted_crossentropy':
    loss_flag = 'weighted'
elif loss == 'custom_binary_crossentropy':
    loss_flag = 'custom_binary'
elif loss == 'MSE':
    loss_flag = 'MSE'
elif loss == 'MSE_binary_crossentropy':
    loss_flag = 'MSE_binary'
elif loss == 'KL_MSE':
    loss_flag = 'KL_MSE'
elif loss == 'KL_MSE_binary':
    loss_flag = 'KL_MSE_binary'
elif loss == 'KL_MSE_custom_binary':
    loss_flag = 'KL_MSE_custom_binary'
else:
    raise ValueError('loss should be either binary_crossentropy or weighted_crossentropy')


# ## Instantiate an autoencoder

# In[32]:


input_shape = (2048, 2048, 1)
num_classes = 3

enconder_inputs = Input(shape=input_shape)

model_module = importlib.import_module(model_architecture)
encoder = model_module.encoder_variational(input_shape=input_shape)
decoder = model_module.decoder() 
classifier = model_module.classifier() 
z_mean_log_var = encoder(enconder_inputs)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
latent_z = Lambda(sampling, name='latent_z')(z_mean_log_var)

classifier_outputs = classifier(latent_z)
decoder_outputs = decoder(latent_z)
model_ae_clf = Model(enconder_inputs, 
                     [z_mean_log_var, decoder_outputs, classifier_outputs],
                     name='vae')

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

metrics=['accuracy', abs_error]

if loss_flag == 'binary':
    model_ae_clf.compile(loss='binary_crossentropy',
                              optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                              metrics=metrics)
elif loss_flag == 'weighted':
    model_ae_clf.compile(loss=weighted_crossentropy,
                              optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                              metrics=metrics)
elif loss_flag == 'custom_binary':
    model_ae_clf.compile(loss=custom_binary_crossentropy,
                              optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                              metrics=metrics)
elif loss_flag == 'MSE':
    model_ae_clf.compile(loss={'decoder':'mean_squared_error'},
                              optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                              metrics=metrics)
elif loss_flag == 'MSE_binary':
    model_ae_clf.compile(loss={'classifier': 'binary_crossentropy', 
                               'decoder': 'mean_squared_error'},
                         optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                         metrics=metrics)
elif loss_flag == 'KL_MSE':
    model_ae_clf.compile(loss={'encoder_variational': kl_loss, 
                               'decoder': 'mean_squared_error'},
                         loss_weights={'encoder_variational': kl_weight, 
                                       'decoder': decoder_weight},
                         optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                         metrics={'decoder': metrics})
elif loss_flag == 'KL_MSE_binary':
    model_ae_clf.compile(loss={'encoder_variational': kl_loss, 
                               'decoder': 'mean_squared_error',
                               'classifier': 'binary_crossentropy'},
                         loss_weights={'encoder_variational': kl_weight, 
                                       'decoder': decoder_weight,
                                       'classifier': 1.0},
                         optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                         metrics={'decoder': metrics, 'classifier': metrics}) 
elif loss_flag == 'KL_MSE_custom_binary':
    model_ae_clf.compile(loss={'encoder_variational': kl_loss, 
                               'decoder': 'mean_squared_error',
                               'classifier': custom_binary_crossentropy(classifier_weight)},
                         loss_weights={'encoder_variational': kl_weight, 
                                       'decoder': decoder_weight,
                                       'classifier': 1.0},
                         optimizer=Adam(lr=lr_schedule_wrap(init_lr=init_lr)(0)),
                         metrics={'decoder': metrics, 'classifier': metrics}) 

model_ae_clf.summary()


# ## Train a neural networks model (in a semisupervised fashion)

# ### Set training parameters

# In[33]:


batch_size = 4
epochs = 400
data_augmentation = True

model_name = '%s.{epoch:03d}.h5' % model_architecture
save_dir = '/data/vision/polina/projects/nn-visualization/code/training/semisupervised/'           +model_architecture+'_'+loss+'_'+str(init_lr)+'_'+str(kl_weight)+'_'+str(clf_weight)+'_'+str(decoder_weight)+'_'+str(ae_weight)
log_dir = os.path.join(save_dir, 'log')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
filepath = os.path.join(save_dir, model_name)


# ### Prepare callbacks for model saving and for learning rate adjustment

# In[34]:


checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_classifier_abs_error',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

lr_scheduler = LearningRateScheduler(lr_schedule_wrap(init_lr=init_lr))

lr_reducer = ReduceLROnPlateau(monitor='val_classifier_abs_error',
                               patience=4,
                               factor=np.sqrt(0.1),
                               min_lr=0.5e-6,
                               mode='min')

tensorboard = TensorBoard(log_dir=log_dir, 
                          batch_size=batch_size)

lossweight_control = ChangeLossWeights(classifier_weight=classifier_weight, clf_weight=clf_weight)

callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard, lossweight_control]

data_dir = '/data/vision/polina/projects/chestxray/data/png_16bit_v2/'


# ### Run training

# In[35]:


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    train_generator = XRayDataGenerator(train_labeled_list_IDs, train_labels, 
                                        data_dir = data_dir,
                                        batch_size = batch_size,
                                        latent_dim = latent_dim,
                                        is_semisupervised = True,
                                        labeled_list_IDs = train_labeled_list_IDs,
                                        unlabeled_list_IDs = train_unlabeled_list_IDs)
    val_generator = XRayDataGenerator(val_list_IDs, val_labels,
                                      data_dir = data_dir,
                                      batch_size = batch_size,
                                      is_training = False,
                                      latent_dim = latent_dim)   

    # Fit the model on the batches generated by train_generator.
    model_ae_clf.fit_generator(train_generator, 
                               validation_data=val_generator,
                               epochs=epochs, verbose=1, workers=num_cpus,
                               use_multiprocessing = True,
                               callbacks=callbacks)

