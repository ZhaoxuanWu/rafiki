#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import requests
import pprint
import os
import tempfile
import numpy as np
import gzip
import csv
import shutil
from tqdm import tqdm
from itertools import chain
from PIL import Image

from rafiki.model import dataset_utils

def load(train_images_url, train_labels_url, test_images_url, test_labels_url, label_to_name, \
        out_train_dataset_path, out_test_dataset_path, out_meta_csv_path, limit=None):
    '''
        Loads and converts an image dataset of the MNIST format to the DatasetType `IMAGE_FILES`.
        Refer to http://yann.lecun.com/exdb/mnist/ for the MNIST dataset format for.

        :param str train_images_url: URL to download the training set images stored in the MNIST format
        :param str train_labels_url: URL to download the training set labels stored in the MNIST format
        :param str test_images_url: URL to download the test set images stored in the MNIST format
        :param str test_labels_url: URL to download the test set labels stored in the MNIST format
        :param dict[int, str] label_to_name: Dictionary mapping label index to label name
        :param str out_train_dataset_path: Path to save the output train dataset file
        :param str out_test_dataset_path: Path to save the output test dataset file
        :param str out_meta_csv_path: Path to save the output dataset metadata .CSV file
        :param int limit: Maximum number of train & test samples (for purposes of testing)
    '''

    print('Downloading files...')
    train_images_file_path = dataset_utils.download_dataset_from_uri(train_images_url)
    train_labels_file_path = dataset_utils.download_dataset_from_uri(train_labels_url)
    test_images_file_path = dataset_utils.download_dataset_from_uri(test_images_url)
    test_labels_file_path = dataset_utils.download_dataset_from_uri(test_labels_url)

    print('Loading datasets into memory...')
    (train_images, train_labels) = _load_dataset_from_files(train_images_file_path, train_labels_file_path, limit=limit)
    (test_images, test_labels) = _load_dataset_from_files(test_images_file_path, test_labels_file_path, limit=limit)

    print('Converting and writing datasets...')

    (label_to_index) = _write_meta_csv(chain(train_labels, test_labels), label_to_name, out_meta_csv_path)
    print('Dataset metadata file is saved at {}'.format(out_meta_csv_path))

    _write_dataset(train_images, train_labels, label_to_index, out_train_dataset_path)
    print('Train dataset file is saved at {}'.format(out_train_dataset_path))

    _write_dataset(test_images, test_labels, label_to_index, out_test_dataset_path)
    print('Test dataset file is saved at {}'.format(out_test_dataset_path))

def _write_meta_csv(labels, label_to_name, out_meta_csv_path):
    label_to_index = {}
    with open(out_meta_csv_path, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'name'])
        writer.writeheader()

        for (i, label) in enumerate(sorted(set(labels))):
            label_to_index[label] = i
            writer.writerow({ 'class': i, 'name': label_to_name[label] })

    return (label_to_index)

def _write_dataset(images, labels, label_to_index, out_dataset_path):
    with tempfile.TemporaryDirectory() as d:
        # Create images.csv in temp dir for dataset
        # For each (image, label), save image as .png and add row to images.csv
        # Show a progress bar in the meantime
        images_csv_path = os.path.join(d, 'images.csv')
        n = len(images)
        with open(images_csv_path, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'class'])
            writer.writeheader()
            for (i, image, label) in tqdm(zip(range(n), images, labels), total=n, unit='images'):
                image_name = '{}-{}.png'.format(label, i)
                image_path = os.path.join(d, image_name)
                pil_image = Image.fromarray(image, mode='L')
                pil_image.save(image_path)
                writer.writerow({ 'path': image_name, 'class': label_to_index[label] })

        # Zip and export folder as dataset
        out_path = shutil.make_archive(out_dataset_path, 'zip', d)
        os.rename(out_path, out_dataset_path) # Remove additional trailing `.zip`

def _load_dataset_from_files(images_file_path, labels_file_path, limit=None):
    with gzip.open(labels_file_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_file_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
        np.reshape(images, (len(labels), 28, 28))

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    return (images, labels)

if __name__ == '__main__':
    # Loads the official Fashion MNIST dataset as `IMAGE_FILES` DatasetType
    load(
        train_images_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        train_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        test_images_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        test_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        label_to_name={
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        },
        out_train_dataset_path='data/fashion_mnist_for_image_classification_train.zip',
        out_test_dataset_path='data/fashion_mnist_for_image_classification_test.zip',
        out_meta_csv_path='data/fashion_mnist_for_image_classification_meta.csv'
    )


    
    