import os
import pandas as pd
from google.cloud import storage
from io import StringIO

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib
import numpy as np
import argparse
import sys
import logging
import time
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import string
import re

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
parser.add_argument('--param-file', dest='param_file',
                    default='gs://vddl-test-vddl-419615-unique/param.csv', type=str,
                    help='Output file for parameters')
args = parser.parse_args()

logging.info('DEVICES'  + str(device_lib.list_local_devices()))

start_time = time.time()

if args.distribute == 'single': # Single Machine, single compute device
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    strategy_type = "Single device training"
    logging.info(strategy_type)
elif args.distribute == 'mirrored': # Single Machine, multiple compute device
    strategy = tf.distribute.MirroredStrategy()
    strategy_type = "Mirrored Strategy distributed training"
    logging.info(strategy_type)
elif args.distribute == 'multiworker': # Multi Machine, multiple compute device
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    strategy_type = "Multi-worker Strategy distributed training"
    logging.info(strategy_type)
    logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
elif args.distribute == 'tpu':  # Single Machine, multiple TPU devices
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

logging.info('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))

#constants.
# batch_size = 32
max_features = 20000
embedding_dim = 128
sequence_length = 500

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def get_compiled_model():
    inputs = keras.Input(shape=(None,), dtype="int64")
    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = keras.Model(inputs, predictions)
    return model


print("Number of devices: {}".format(strategy.num_replicas_in_sync))

def save_params_to_csv(args, training_time, strategy_type, total_parameters, start_time, end_time):
    
    defaults = {'lr': 0.001, 'epochs': 10, 'batch_size': 16, 'steps': 100}
    
    params = {
        'Learning Rate': [args.lr if args.lr is not None else defaults['lr']],
        'Epochs': [args.epochs if args.epochs is not None else defaults['epochs']],
        'Batch Size': [args.batch_size if args.batch_size is not None else defaults['batch_size']],
        'Steps per Epoch': [args.steps if args.steps is not None else defaults['steps']],
        'Start time': [start_time],
        'End time': [end_time],
        'Training Time (seconds)': [training_time],
        'Distribute Strategy': [strategy_type],
        'Model Paramters': [total_parameters]
    }
   
    df_new = pd.DataFrame(params)
    
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = args.param_file.split('/')[2]
    blob_path = '/'.join(args.param_file.split('/')[3:])
    print("Bucket Name: ", bucket_name)
    print("Blob Path: ", blob_path) #File Path 
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Try to load existing data and append new data
    try:
        # Download the existing file to a buffer
        print("Download the existing file if exists...")
        buffer = blob.download_as_text()
        df_existing = pd.read_csv(StringIO(buffer))
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        print("Appended the new data in the existing file...")
    except Exception as e:
        print(f"Error downloading the file: {e}. A new file will be created.")
        # If the file does not exist or other errors occur, use new data
        print("Creating new file ...")
        df_final = df_new
    
    print("Output param file location: ", args.param_file)
    
    try:
        # Convert DataFrame to CSV and upload back to the GCS
        print("Will upload the file in GCS ....")
        blob.reload()
        blob.upload_from_string(df_final.to_csv(index=False), 'text/csv')
        
        # df.to_csv(args.param_file, index=False)
        print("Param file saved....")
    except google.api_core.exceptions.BadRequest as e:
        print(f"Failed to upload due to a bad request: {e.message}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.content}")

with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    
    logging.info("Get Model...")
    model = get_compiled_model()

    # Train the model on all available devices.
    # train_dataset, val_dataset, test_dataset = get_dataset()
    # model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)
    
    raw_train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size, 
                                                           validation_split=0.2, subset="training", seed=1337,)
    raw_val_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size,
                                                         validation_split=0.2, subset="validation", seed=1337,)
    raw_test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)
    
    print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
    print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
    print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")
              
    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # Do async prefetching / buffering of the data for best performance on GPU.
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)
              
    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    model.evaluate(test_ds)

    # Test the model on all available devices.
    # model.evaluate(test_dataset)
    
    summary = model.summary()
    print("Model summary: ", summary)

    total_parameters = model.count_params()
    print("Total number of parameters in the model: ", total_parameters)
              
    end_time = time.time()
    training_time = end_time - start_time
    print("Model Training time: ", training_time)
    
    save_params_to_csv(args, training_time, strategy_type, total_parameters, start_time, end_time)

# CMDARGS = ["--epochs=3", "--batch_size=32", "--distribute=multiworker"]