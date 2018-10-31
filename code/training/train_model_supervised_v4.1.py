import tensorflow as tf
import os
import importlib
import random
import sys

_NUM_TRAINING_IMAGES = 4537
_NUM_VAL_IMAGES = 628
_NUM_TEST_IMAGES = 606
_IMG_HEIGHT = 2048
_IMG_WIDTH = 2048
_BATCH_SIZE = 4

_NUM_TRAIN_FILES = 57
_NUM_VAL_FILES = 8
_NUM_TEST_FILES = 8
_NUM_UNLABELLED_FILES = 276


def get_filenames(is_training, data_dir):
    if is_training:
        if 'unlabelled' not in data_dir:
            return [os.path.join(data_dir, 'training-%05d-of-%05d.tfrecords' %
                             (i + 1, _NUM_TRAIN_FILES))
                    for i in range(_NUM_TRAIN_FILES)]
        else:
            return [os.path.join(data_dir, 'unlabelled-%05d-of-%05d.tfrecords' %
                    (i + 1, _NUM_UNLABELLED_FILES))
                    for i in range(99)]+\
                   [os.path.join(data_dir,'unlabelled-%05d-of-%05d.tfrecords' %
                    (i + 1, _NUM_UNLABELLED_FILES))
                    for i in range(103, _NUM_UNLABELLED_FILES)]
    else:
        return [os.path.join(data_dir, 'validation-%05d-of-%05d.tfrecords' %
                             (i + 1, _NUM_VAL_FILES))
                for i in range(_NUM_VAL_FILES)]


def _parse_example_proto(example_serialized):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/label': tf.FixedLenFeature([1], dtype=tf.float32,
                                          default_value=-1),
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64,
                                           default_value=-1),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                          default_value=-1),
        'image/filepath': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value='')
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    label = tf.cast(features['image/label'], dtype=tf.float32)

    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)

    return features['image/encoded'], label, height, width


def _ordinal_encoding(label):
    label = tf.reshape(label, [])
    ordinal_label = tf.constant([0, 0, 0], dtype=tf.float32)
    ordinal_label = tf.cond(tf.equal(label, tf.constant(1, dtype=tf.float32)),
                            lambda: tf.constant([1, 0, 0], dtype=tf.float32),
                            lambda: ordinal_label)
    ordinal_label = tf.cond(tf.equal(label, tf.constant(2, dtype=tf.float32)),
                            lambda: tf.constant([1, 1, 0], dtype=tf.float32),
                            lambda: ordinal_label)
    ordinal_label = tf.cond(tf.equal(label, tf.constant(3, dtype=tf.float32)),
                            lambda: tf.constant([1, 1, 1], dtype=tf.float32),
                            lambda: ordinal_label)

    return ordinal_label


def _img_shift(img, height, width):
    height_flag = random.uniform(-1, 1)
    shift_height = random.gauss(0, 100)
    shift_height = abs(shift_height)
    if shift_height > 300:
        shift_height = 0
    if height_flag >= 0:
        offset_height = 0
    else:
        offset_height = int(shift_height)
    target_height = int(shift_height) + height

    width_flag = random.uniform(-1, 1)
    shift_width = random.gauss(0, 100)
    shift_width = abs(shift_width)
    if shift_width > 300:
        shift_width = 0
    if width_flag >= 0:
        offset_width = 0
    else:
        offset_width = int(shift_width)
    target_width = int(shift_width) + width

    offset_height = tf.reshape(offset_height, [])
    offset_width = tf.reshape(offset_width, [])
    target_height = tf.reshape(target_height, [])
    target_width = tf.reshape(target_width, [])

    return tf.image.pad_to_bounding_box(img, offset_height, offset_width,
                                        target_height, target_width)


def _img_rotate(img):
    angle = random.gauss(0, 30)
    if angle > 90 or angle < -90:
        angle = 0

    return tf.contrib.image.rotate(img, angle)


def parse_record_fn(raw_record, is_training, regression):
    """
    Parses a record containing a training example of an image.
    """

    img, label, height, width = _parse_example_proto(raw_record)

    decoded_img = tf.decode_raw(bytes=img, out_type=tf.float32)
    shape = tf.concat([height, width, tf.constant([1])], axis=0)
    reshaped_img = tf.reshape(decoded_img, shape)

    if is_training:
        reshaped_img = _img_shift(reshaped_img, height, width)
        rotated_reshaped_img = _img_rotate(reshaped_img)
        cropped_img = tf.image.resize_image_with_crop_or_pad(
            rotated_reshaped_img, _IMG_HEIGHT, _IMG_WIDTH)
    else:
        cropped_img = tf.image.resize_image_with_crop_or_pad(reshaped_img,
                                                             _IMG_HEIGHT,
                                                             _IMG_WIDTH)

    if not regression:
        label = _ordinal_encoding(label)

    return {"img": cropped_img, "label": label}, label


def input_fn(is_training, data_dir, batch_size, regression=False):
    """
    Input function which provides batches for training or val.
    """

    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Convert to individual records.
    # cycle_length = 8 means 8 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small
    # systems but high enough to provide the benefits of parallelization. You
    # may want to increase this number if you have a large number of CPU cores.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=batch_size * 2))

    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

    dataset = dataset.map(
        lambda value: parse_record_fn(value, is_training, regression),
        num_parallel_calls=batch_size)
    dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset


def _learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    learning_rate = tf.train.piecewise_constant(global_step, [45400, 113500,
                                                              227000, 363200],
                                                [0.1, 0.05, 0.025, 0.0125,
                                                 0.01])
    return learning_rate


def model_fn(features, labels, mode, params):
    """
    Model function which Initializes the NN model and builds EstimatorSpec
    for training, val, or prediction
    """

    img_features = features['img']
    true_labels = features['label']
    true_labels = tf.identity(true_labels, 'true_label')

    model_name = params['model_name']
    model_module = importlib.import_module(model_name)

    model = model_module.Model()
    if mode == tf.estimator.ModeKeys.TRAIN:
        model.build(img_features, tf.constant(True, dtype=tf.bool))
    else:
        model.build(img_features, tf.constant(False, dtype=tf.bool))
    predictions = model.output
    predictions = tf.identity(predictions, 'model_output')

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

        if mode == tf.estimator.ModeKeys.TRAIN:

            if params['labelled_flag']:

                if params['cost_function'] == 'cross_entropy':
                    epsilon = 1e-6
                    training_images_number = params['training_images_number']
                    training_class0_number = params['training_class0_number']
                    training_class1_number = params['training_class1_number']
                    training_class2_number = params['training_class2_number']
                    training_class3_number = params['training_class3_number']
                    loss = tf.reduce_mean( -1*(training_images_number/training_class0_number)*tf.abs(labels[:, 0]-1)*tf.log(1-predictions[:, 0]+epsilon)
                        - (training_images_number/(training_class1_number+training_class2_number+training_class3_number))*labels[:, 0]*tf.log(predictions[:, 0]+epsilon)
                        - (training_images_number/(training_class0_number+training_class1_number))*tf.abs(labels[:, 1]-1)*tf.log(1-predictions[:, 1]+epsilon)
                        - (training_images_number/(training_class2_number+training_class3_number))*labels[:, 1]*tf.log(predictions[:, 1]+epsilon)
                        - (training_images_number/(training_class0_number+training_class1_number+training_class2_number))*tf.abs(labels[:, 2]-1)*tf.log(1-predictions[:, 2]+epsilon)
                        - (training_images_number/training_class3_number)*labels[:, 2]*tf.log(predictions[:, 2]+epsilon) )
                if params['cost_function'] == 'MSE':
                    loss = tf.reduce_mean( (predictions[:] - labels[:]) ** 2 )
                if params['cost_function'] == 'cross_entropy_reg':
                    epsilon = 1e-6
                    training_images_number = params['training_images_number']
                    training_class0_number = params['training_class0_number']
                    training_class1_number = params['training_class1_number']
                    training_class2_number = params['training_class2_number']
                    training_class3_number = params['training_class3_number']
                    reg_weight = params['reg_weight']
                    loss = tf.reduce_mean(-1 * (training_images_number / training_class0_number) * tf.abs(labels[:, 0] - 1) * tf.log(1 - predictions[:, 0] + epsilon)
                                          - (training_images_number / (training_class1_number + training_class2_number + training_class3_number)) * labels[:, 0] * tf.log(predictions[:, 0] + epsilon)
                                          - (training_images_number / (training_class0_number + training_class1_number)) * tf.abs(labels[:, 1] - 1) * tf.log(1 - predictions[:, 1] + epsilon)
                                          - (training_images_number / (training_class2_number + training_class3_number)) * labels[:, 1] * tf.log(predictions[:, 1] + epsilon)
                                          - (training_images_number / (training_class0_number + training_class1_number + training_class2_number)) * tf.abs(labels[:, 2] - 1) * tf.log(1 - predictions[:, 2] + epsilon)
                                          - (training_images_number / training_class3_number) * labels[:, 2] * tf.log(predictions[:, 2] + epsilon)
                                          + reg_weight * (
                                                  - (1 - predictions[:, 0]) * tf.log(1 - predictions[:, 0] + epsilon)
                                                  - predictions[:, 0] * tf.log(predictions[:, 0] + epsilon)
                                                  - (1 - predictions[:, 1]) * tf.log(1 - predictions[:, 1] + epsilon)
                                                  - predictions[:, 1] * tf.log(predictions[:, 1] + epsilon)
                                                  - (1 - predictions[:, 2]) * tf.log(1 - predictions[:, 2] + epsilon)
                                                  - predictions[:, 2] * tf.log(predictions[:, 2] + epsilon)
                                          ))

                global_step = tf.train.get_or_create_global_step()
                learning_rate = _learning_rate_fn(global_step)

                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=0.9)

                minimize_op = optimizer.minimize(loss, global_step)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = tf.group(minimize_op, update_ops)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op)

            else:

                if params['cost_function'] == 'entropy':
                    epsilon = 1e-6
                    loss = tf.reduce_mean((predictions[:, 0] - 1) * tf.log(
                        1 - predictions[:, 0] + epsilon) - predictions[:,
                                                           0] * tf.log(
                        predictions[:, 0] + epsilon))

                global_step = tf.train.get_or_create_global_step()
                learning_rate = _learning_rate_fn(global_step)

                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=0.9)

                minimize_op = optimizer.minimize(loss, global_step)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = tf.group(minimize_op, update_ops)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op)

        else:

            loss = tf.constant([0], dtype=tf.float32)
            train_op = None

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predicts": predictions, "labels": true_labels},
                loss=loss,
                train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predicts": predictions, "labels": true_labels},
                export_outputs={'predict': tf.estimator.export.PredictOutput(
                                            {"predicts": predictions, "labels": true_labels})})


def main(argv):

    model_name = argv[1]
    gpu_num = argv[2]
    reg_weight = argv[3]
    model_dir = argv[4]

    total_training_epoch = 480

    training_data_dir = '/data/vision/polina/projects/chestxray/work_space/labels_after_segmentation_removingLineBreaks/sub_img_v2/training_npy/tfrecords'
    training_unlabelled_data_dir = \
        '/data/vision/polina/projects/chestxray/work_space' \
        '/labels_after_segmentation_removingLineBreaks' \
        '/unlabelled/tfrecords'

    supervised_model_params = {'cost_function': 'cross_entropy_reg',
                    'training_images_number': 4537,
                    'training_class0_number': 784,
                    'training_class1_number': 2394,
                    'training_class2_number': 1122,
                    'training_class3_number': 237,
                    'model_name': model_name,
                    'labelled_flag': True,
                    'reg_weight': float(reg_weight)}
    unsupervised_model_params = {'cost_function': 'entropy',
                    'model_name': model_name,
                    'labelled_flag': False}

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    # Check gpu numbers
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    print(gpu_names)

    def input_fn_train():
        return input_fn(
            is_training=True, data_dir=training_data_dir,
            batch_size=_BATCH_SIZE)

    def input_fn_train_unlabelled():
        return input_fn(
            is_training=True, data_dir=training_unlabelled_data_dir,
            batch_size=_BATCH_SIZE)

    run_config = tf.estimator.RunConfig(keep_checkpoint_max=None)

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=model_dir,
                                        config=run_config,
                                        params=supervised_model_params)
    for epoch_index in range(total_training_epoch):
        classifier.train(input_fn=input_fn_train)

    # for epoch_index in range(total_training_epoch):
    #     if epoch_index % 10 == 0:
    #         classifier = tf.estimator.Estimator(model_fn=model_fn,
    #                                             model_dir=model_dir,
    #                                             config=run_config,
    #                                             params=unsupervised_model_params)
    #         classifier.train(input_fn=input_fn_train_unlabelled)
    #     else:
    #         classifier = tf.estimator.Estimator(model_fn=model_fn,
    #                                             model_dir=model_dir,
    #                                             config=run_config,
    #                                             params=supervised_model_params)
    #         classifier.train(input_fn=input_fn_train)

if __name__ == '__main__':
    main(argv=sys.argv)
