"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
import os
from os import listdir

from PIL import Image
from scipy.misc import imresize

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((128.96269759, 122.2720388, 116.21071466), dtype=np.float32)

NUM_CLASSES = 4
SAVE_DIR = './output/'


def num_to_str(num, num_digits=4):
    file_prefix = ''
    index_str = str(num)
    for _ in range(num_digits - len(index_str)):
        file_prefix += '0'
    return file_prefix + index_str


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("start_index", type=int,
                        help="Index to start")
    # parser.add_argument("model_weights", type=str,
    #                     help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # For bulk inference, resize all images to 640
    target_resize_length = 320
    imgs = None
    shapes = []
    names = []
    print("Loading...")
    index = args.start_index
    for fname in sorted(listdir(args.img_path))[index:]:
        print(fname)
        if fname.startswith('.'):
            continue
        if index >= args.start_index + 30:
            break
        # Prepare image.
        if fname.split('.')[1] in ('jpg', 'jpeg', 'JPG'):
            img = tf.image.decode_jpeg(tf.read_file(args.img_path + fname), channels=3)
        else:
            img = tf.image.decode_png(tf.read_file(args.img_path + fname), channels=3)
        # Convert RGB to BGR.
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
        # Extract mean.
        img -= IMG_MEAN
        # Shape save.
        im = Image.open(args.img_path + fname)
        width, height = im.size
        shapes.append((height, width))
        img = tf.image.resize_bilinear(tf.expand_dims(img, 0), (target_resize_length, target_resize_length))
        if imgs is None:
            imgs = img
        else:
            imgs = tf.concat([imgs, img], 0)
        names.append(fname)
        index += 1

    print("Model Creating...")

    # Create network.
    net = DeepLabResNetModel({'data': imgs}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    # raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])
    # raw_output_up = tf.argmax(raw_output_up, dimension=3)
    # pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    ckpt = tf.train.get_checkpoint_state('./snapshots/')
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    fname = os.path.join('./snapshots', ckpt_name)
    load(loader, sess, fname)

    # Perform inference.
    raw_outputs = sess.run(raw_output)

    for i in range(len(raw_outputs)):
        print(i)
        output = raw_outputs[i]
        shape = shapes[i]
        name = names[i]
        output_up = imresize(output, shape)
        output_up = np.argmax(output_up, 2)
        pred = np.expand_dims(output_up, 2)
        pred = np.expand_dims(pred, 0)
        msk = decode_labels(pred, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        im.save(args.save_dir + 'mask_{}.png'.format(name.split('.')[0]))
        np.save(args.save_dir + 'raw_mask_{}.npy'.format(name.split('.')[0]), pred[0])

        # print('The output file has been saved to {}'.format(args.save_dir + 'mask_{}.png'.format(num_to_str(i))))
        # print('The output file has been saved to {}'.format(args.save_dir + 'raw_mask_{}.npy'.format(num_to_str(i))))


if __name__ == '__main__':
    main()
