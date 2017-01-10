import os.path as op
import os
from classify_image import *
import numpy as np
import time
import pandas as pd
from scipy.misc import imread
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

root = '/home/leo/Downloads/A3'
labels = op.join(root, 'val.csv')
images = op.join(root, 'val')

seed = 128
rng = np.random.RandomState(seed)

classes = pd.read_csv(labels)

temp = []
for img_name in classes.Filename:
    image_path = op.join(images, img_name)
    img = imread(image_path)
    img = img.astype('float32')
    temp.append(img)

imgs = np.stack(temp)
y = classes['Label'].values

def extract_features(file_name):
    FLAGS.model_dir = 'model/'
    maybe_download_and_extract()
    create_graph()
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        predictions = np.zeros((len(y), 1008), dtype='float32')
        representations = np.zeros((len(y), 2048), dtype='float32')
        for i in range(len(y)):
            start = time.time()
            [reps, preds] = sess.run([representation_tensor, softmax_tensor], {'DecodeJpeg:0': imgs[i]})
            if (i % 8 == 0):
                print("{}/{} Time for batch {} ".format(i, len(y), time.time() - start))
            predictions[i] = np.squeeze(preds)
            representations[i] = np.squeeze(reps)
        np.savez_compressed(file_name + ".npz", predictions=predictions, representations=representations, y=y)

if __name__ == '__main__':
    extract_features('data_batch_test')
