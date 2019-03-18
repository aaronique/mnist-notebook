import numpy
import tensorflow as tf
import time
from datetime import datetime
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflowonspark import TFNode


class StopFeedHook(tf.train.SessionRunHook):
  """SessionRunHook to terminate InputMode.SPARK RDD feeding if the training loop exits before the entire RDD is consumed."""

  def __init__(self, tf_feed):
    self._tf_feed = tf_feed

  def end(self, session):
    self._tf_feed.terminate()
    self._tf_feed.next_batch(1)


def main_keras(args, ctx):
  IMAGE_PIXELS = 28
  NUM_CLASSES = 10

  # use Keras API to load data
  from tensorflow.python.keras.datasets import mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(60000, 784)
  x_test = x_test.reshape(10000, 784)
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

  # setup a Keras model
  model = Sequential()
  model.add(Dense(512, activation='relu', input_shape=(784,)))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(10, activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
                metrics=['accuracy'])
  model.summary()

  print("model.inputs: {}".format(model.inputs))
  print("model.outputs: {}".format(model.outputs))

  # convert Keras model to tf.estimator
  estimator = tf.keras.estimator.model_to_estimator(model, model_dir=args.model)

  # setup train_input_fn
  # For InputMode.SPARK, read data from RDD
  tf_feed = TFNode.DataFeed(ctx.mgr)

  def rdd_generator():
    while not tf_feed.should_stop():
      batch = tf_feed.next_batch(1)
      if len(batch) == 0:
        return
      record = batch[0]
      image = numpy.array(record[0]).astype(numpy.float32) / 255.0
      label = numpy.array(record[1]).astype(numpy.int64)
      yield (image, label)

  def train_input_fn():
    ds = tf.data.Dataset.from_generator(rdd_generator,
                                        (tf.float32, tf.float32),
                                        (tf.TensorShape([IMAGE_PIXELS * IMAGE_PIXELS]), tf.TensorShape([10])))
    ds = ds.batch(args.batch_size)
    return ds

  # add a hook to terminate the RDD data feed when the session ends
  hooks = [StopFeedHook(tf_feed)]

  # eval_input_fn ALWAYS uses data loaded in memory, since InputMode.SPARK can only feed one RDD at a time
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"dense_input": x_test},
      y=y_test,
      num_epochs=1,
      shuffle=False)

  # setup tf.estimator.train_and_evaluate() w/ FinalExporter
  feature_spec = {'dense_input': tf.placeholder(tf.float32, shape=[None, 784])}
  exporter = tf.estimator.FinalExporter("serving", serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec))
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args.steps, hooks=hooks)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=exporter)

  # train and export model
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
  # wait for all other nodes to complete (via done files)
  done_dir = "{}/done".format(ctx.absolute_path(args.model))
  print("Writing done file to: {}".format(done_dir))
  tf.gfile.MakeDirs(done_dir)
  with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
    done_file.write("done")

  for i in range(60):
    if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
      print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
      time.sleep(1)
    else:
      print("{} All nodes done".format(datetime.now().isoformat()))
      break
