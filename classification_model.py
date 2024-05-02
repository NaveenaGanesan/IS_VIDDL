import os
import pandas as pd
# os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib
import numpy as np
import argparse
import sys
import logging
import time

logging.info("Model starting point...")

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
parser.add_argument('--lr', dest='lr',
                    default=0.001, type=float,
                    help='Learning rate.')
parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')
parser.add_argument('--steps', dest='steps',
                    default=100, type=int,
                    help='Number of steps per epoch.')
parser.add_argument('--batch_size', dest='batch_size',
                    default=16, type=int,
                    help='Size of a batch.')
parser.add_argument('--distribute', dest='distribute', type=str, default='single',
                    help='distributed training strategy')
# parser.add_argument('--param-file', dest='param_file',
#                     default='/tmp/param.txt', type=str,
#                     help='Output file for parameters')
parser.add_argument('--param-file', dest='param_file',
                    default='gs://vddl-test-vddl-419615-unique/param.csv', type=str,
                    help='Output file for parameters')
args = parser.parse_args()

logging.info('DEVICES'  + str(device_lib.list_local_devices()))

start_time = time.time()

# Single Machine, single compute device
if args.distribute == 'single':
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    strategy_type = "Single device training"
    logging.info(strategy_type)
# Single Machine, multiple compute device
elif args.distribute == 'mirrored':
    strategy = tf.distribute.MirroredStrategy()
    strategy_type = "Mirrored Strategy distributed training"
    logging.info(strategy_type)
# Multi Machine, multiple compute device
elif args.distribute == 'multiworker':
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    strategy_type = "Multi-worker Strategy distributed training"
    logging.info(strategy_type)
    logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
    # Single Machine, multiple TPU devices
elif args.distribute == 'tpu':
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

logging.info('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))


def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_dataset():
    batch_size = 32
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

# Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

def save_params_to_csv(args, training_time, strategy_type, total_parameters):
    params = {
        # 'Learning Rate': [args.lr],
        'Epochs': [args.epochs],
        'Batch Size': [args.batch_size],
        # 'Steps per Epoch': [args.steps]
        'Training Time (seconds)': [training_time],
        'Distribute Strategy': [strategy_type],
        'Model Paramters': [total_parameters]
    }
    df = pd.DataFrame(params)
    print("Output param file location: ", args.param_file)
    df.to_csv(args.param_file, index=False)
    print("Param file saved....")

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    
    logging.info("Get Model...")
    model = get_compiled_model()

    # Train the model on all available devices.
    train_dataset, val_dataset, test_dataset = get_dataset()
    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)

    # Test the model on all available devices.
    model.evaluate(test_dataset)
    
    # Print the model summary to see detailed parameters
    summary = model.summary()
    print("Model summary: ", summary)

    # Calculate and print the total number of parameters
    total_parameters = model.count_params()
    print("Total number of parameters in the model: ", total_parameters)

    training_time = time.time() - start_time
    print("Model Training time: {}", training_time)
    
    # Call the function to save parameters
    save_params_to_csv(args, training_time, strategy_type, total_parameters)