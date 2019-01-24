from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import model_from_json, load_model
from keras.layers import MaxPooling2D, Convolution2D
from keras.preprocessing import image
from keras import models

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import math
import cv2
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parse = argparse.ArgumentParser(description="Visualize featuremap")
    parse.add_argument("--img", "-i", default=None,
                       help="Path to predict image file")
    parse.add_argument("--model", "-m", default=None,
                       help="Path to trained model file")
    parse.add_argument("--model_json", "-j", default=None,
                       help="Path to trained model json file")
    parse.add_argument("--model_weight", "-w", default=None,
                       help="Path to trained model weighs file")
    parse.add_argument("--directory", "-d", default="./output",
                       help="Directory to output heatmap images")
    args = parse.parse_args()

    # NOTE: Load image.
    img = image.load_img(args.img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    print("IMAGE FILENAME: %s" % os.path.basename(args.img))

    # NOTE: Load trained model.
    print("[status] Loaded model...")
    if args.model is not None:
        model = load_model(args.model)
    elif not (args.model_json is None or args.model_weight is None):
        model = model_from_json(open(args.model_json).read())
        model.load_weights(args.model_weight)
    else:
        model = VGG16(weights="imagenet")

    # NOTE: Select output layer and predict.
    print("[status] Extract input image features...")
    layers = model.layers[1:19]
    layer_outputs = [layer.output for layer in layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    # activation_model.summary()
    activations = activation_model.predict(img)

    # NOTE: Select visualize layers.
    conv_and_pool_activations = []
    for layer, activation in zip(layers, activations):
        is_pooling_layer = isinstance(layer, MaxPooling2D)
        is_convolution_layer = isinstance(layer, Convolution2D)
        if is_pooling_layer or is_convolution_layer:
            conv_and_pool_activations.append([layer.name, activation])

    # NOTE: Generate heatmap.
    print("[status] Generating heatmaps...")
    os.makedirs(args.directory, exist_ok=True)
    for i, (name, activation) in enumerate(conv_and_pool_activations):
        print("[status] Processing %s layer..." % name)
        n_imgs = activation.shape[3]
        n_cols = math.ceil(math.sqrt(n_imgs))
        n_rows = math.floor(n_imgs / n_cols)
        screens = []
        for y in range(0, n_rows):
            rows = []
            for x in range(0, n_cols):
                j = y * n_cols + x
                if j < n_imgs:
                    featuremap_img = activation[0, :, :, j]
                    rows.append(featuremap_img)
                else:
                    rows.append(np.zeros())
            screens.append(np.concatenate(rows, axis=1))
        screens = np.concatenate(screens, axis=0)
        plt.figure()
        sns.heatmap(screens, xticklabels=False, yticklabels=False)
        save_name = "%s.png" % name
        save_path = os.path.join(args.directory, save_name)
        plt.savefig(save_path)
        plt.close()
    print("[status] Generating heatmap has finished...")


if __name__ == "__main__":
    main()
