import urllib
import os
import sys
import tarfile
import glob
import pickle
import numpy as np
import cv2

def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.
  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

classification = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIR = 'data'


#download_and_uncompress_tarball(DATA_URL, DATA_DIR)

folders = '/home/aiserver/muke/data_manager/data/cifar-10-batches-py'

trfiles = glob.glob(folders + "/test_batch*")

data  = []
labels = []
for file in trfiles:
    dt = unpickle(file)
    data += list(dt[b"data"])
    labels += list(dt[b"labels"])

print(labels)

imgs = np.reshape(data, [-1, 3, 32, 32])

for i in range(imgs.shape[0]):
    im_data = imgs[i, ...]
    im_data = np.transpose(im_data, [1, 2, 0])
    im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR)

    f = "{}/{}".format("data/image/test", classification[labels[i]])

    if not os.path.exists(f):
        os.mkdir(f)

    cv2.imwrite("{}/{}.jpg".format(f, str(i)), im_data)













