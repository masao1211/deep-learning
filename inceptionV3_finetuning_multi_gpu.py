from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.utils.training_utils import multi_gpu_model
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras import backend as K
import keras.utils as utils
import tensorflow as tf
import keras

import matplotlib.pyplot as plt
from imutils import contours
from sys import getsizeof
import numpy as np
import argparse
import imutils
import glob
import time
import math
import sys
import cv2
import os


def main():
    parse = argparse.ArgumentParser(description="Tensorflow train")
    parse.add_argument("--basedir", "-d", default="",
                       help="Path to directory of image classes")
    parse.add_argument("--output", "-o", default="output",
                       help="Path to directory of output")
    parse.add_argument("--batch", "-b", type=int, default=40,
                       help="Number of batch size for training")
    parse.add_argument("--img_size", "-s", type=int, default=299,
                       help="Size of image")
    parse.add_argument("--epoch", "-e", type=int, default=10,
                       help="Number of epoch")
    parse.add_argument("--gpu", "-g", type=int, default=1,
                       help="Number of gpus to use")
    parse.add_argument("--train", "-t", default="/train_dir",
                       help="Directory of train image")
    parse.add_argument("--val", "-v", default="/val_dir",
                       help="Directory of validation image")
    parse.add_argument("--prefix", "-p", default="",
                       help="Prefix of filename")
    args = parse.parse_args()

    train_dir = os.path.join(args.basedir, args.train)
    t_class_dirs = [d for d in glob.glob(train_dir+"/*") if os.path.isdir(d)]
    train_size = sum([len(glob.glob(d+"/*.jpg")) for d in t_class_dirs])

    val_dir = os.path.join(args.basedir, args.val)
    v_class_dirs = [d for d in glob.glob(val_dir+"/*") if os.path.isdir(d)]
    val_size = sum([len(glob.glob(d+"/*.jpg")) for d in v_class_dirs])

    n_classes = len(t_class_dirs)

    # NOTE: Set keras backend.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    channels_index = "channels_last"
    K.set_image_data_format(channels_index)

    # NOTE: Construct model.
    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(args.img_size, args.img_size, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers[:len(base_model.layers)]:
        layer.trainable = True
    for layer in model.layers[len(base_model.layers):]:
        layer.trainable = True
    gpu_model = multi_gpu_model(model, gpus=args.gpu)
    gpu_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])

    # NOTE: Setup generater.
    train_gen = ImageDataGenerator(
                    rescale=1. / 255,
                    # zoom_range=zoom_range,
                    # rotation_range=rotation_range
                )
    test_gen = ImageDataGenerator(rescale=1. / 255)

    # NOTE: Define callbacks.
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.2,
                                          patience=10,
                                          min_lr=0.001,
                                          verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_acc', patience=10,
                                      mode='auto', verbose=1),
    ]

    # NOTE: Execute training.
    history = gpu_model.fit_generator(
                        train_gen.flow_from_directory(
                            train_dir,
                            batch_size=args.batch*args.gpu,
                            target_size=(args.img_size, args.img_size)
                        ),
                        callbacks=callbacks,
                        epochs=args.epoch,
                        shuffle=True,
                        verbose=1,
                        validation_data=test_gen.flow_from_directory(
                            val_dir,
                            batch_size=args.batch*args.gpu,
                            target_size=(args.img_size, args.img_size)
                        ),
                        workers=32,
                        max_queue_size=64,
                        use_multiprocessing=True
              )

    # NOTE: Save model.
    model_json = model.to_json()
    save_name = os.path.join(args.output, args.prefix)
    with open("%s_model.json" % save_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("%s_model.h5" % save_name)
    model.save("%s_model.model" % save_name)

    # NOTE: Evalute model.
    score = gpu_model.evaluate_generator(
                        test_gen.flow_from_directory(
                            val_dir,
                            batch_size=args.batch,
                            target_size=(args.img_size, args.img_size)
                        )
            )
    loss, accuracy = score
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # NOTE: Save history as png image.
    history_elements = [
        ["acc", "val_acc", "Accuracy"],
        ["loss", "val_loss", "Loss"]
    ]
    for e_train, e_val, ylabel in history_elements:
        plt.plot(
            range(1, args.epoch+1),
            history.history[e_train],
            label="training"
        )
        plt.plot(
            range(1, args.epoch+1),
            history.history[e_val],
            label="validation"
        )
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig("%s.png" % ylabel)
        plt.clf()


if __name__ == '__main__':
    main()
