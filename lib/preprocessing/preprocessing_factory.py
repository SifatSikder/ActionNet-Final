"""Contains a factory for building various models."""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
from preprocessing import action_preprocessing

def get_preprocessing(name, is_training=False):
  """Returns preprocessing_fn(image, height, width, **kwargs).
  Args:
    name: The name of the preprocessing function.
    is_training: `True` if the model is being used for training and `False`
      otherwise.
  Returns:
    preprocessing_fn: A function that preprocessing a single image (pre-batch).
      It has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).
  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """
  preprocessing_fn_map = {
      'action_vgg_e': action_preprocessing,
  }
  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)
  def preprocessing_fn(image, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(image, output_height, output_width, is_training=is_training, **kwargs)
  return preprocessing_fn
