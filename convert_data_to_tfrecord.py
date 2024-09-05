import math
import os
import random
import sys
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from lib import dataset_utils
_NUM_VALIDATION = 0
_RANDOM_SEED = 0
_NUM_SHARDS = 4
dataset_name = 'jp_2s'
seq = False
dataset_path = './dataset/action_data'
new_image_path = r'D:\New Spl\New folder\ActionNet\dataset\NewImages'
dir_list = '*'
label_name = 'label.txt'
output_path = './dataset/action_merge'
parts = ['t2', 't3']
output_dir = os.path.join(output_path, dataset_name)

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_filenames_and_classes(dataset_path):
  file_list = []
  for folder in glob.glob(os.path.join(dataset_path, dir_list)):
    with open(os.path.join(folder, label_name)) as list_file:
      file_list += list_file.readlines()
  return file_list

def _get_dataset_filename(output_dir, split_name, shard_id):
  output_filename = 'clip_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
  return os.path.join(output_dir, output_filename)

def _convert_dataset(split_name, filenames, dataset_path):
  """Converts the given filenames to a TFRecord dataset.
  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  img_files_a = []
  img_files_b = []
  class_names = []
  for filename in filenames:
    img_files_a.append(os.path.join(dataset_path, filename.split('_')[0] + '_' + filename.split('_')[1]  , parts[0], filename.strip().split()[0].split('_')[2]))
    img_files_b.append(os.path.join(dataset_path, filename.split('_')[0] + '_' + filename.split('_')[1]  , parts[1], filename.strip().split()[0].split('_')[2]))
    class_names.append(filename.strip().split()[1])
  
  img_files_a = img_files_a[:100]
  img_files_b = img_files_b[:100]
  class_names = class_names[:100]
  
  print(len(img_files_a))
  print(len(img_files_b))
  print(len(class_names))

  # total_number_of_samples = len(filenames)
  total_number_of_samples = 100
  num_per_shard = int(math.ceil(total_number_of_samples / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()
    with tf.Session('') as sess:
      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(output_dir, split_name, shard_id)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, total_number_of_samples)
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, total_number_of_samples, shard_id))
            sys.stdout.flush()
            image_a = tf.gfile.FastGFile(img_files_a[i], 'rb').read()
            image_b = tf.gfile.FastGFile(img_files_b[i], 'rb').read()
            class_id = int(class_names[i])
            example = dataset_utils.image_to_tfexample(image_a, image_b, b'jpg', class_id)
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()

def main(_):

  file_list = _get_filenames_and_classes(dataset_path)

  # Divide into train and test:
  if seq:
    file_list.sort()
    training_filenames = file_list[_NUM_VALIDATION:]
  else:
    random.seed(_RANDOM_SEED)
    random.shuffle(file_list)
  training_filenames = file_list[_NUM_VALIDATION:]
  #validation_filenames = file_list[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, new_image_path)
  # _convert_dataset('validation', validation_filenames, dataset_path)
  print('\nFinished converting the dataset!')

if __name__ == '__main__':
  tf.app.run()
