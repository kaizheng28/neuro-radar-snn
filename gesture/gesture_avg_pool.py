#! /usr/bin/env python3

'''

 Filename: local.py
 Author: Kai Zheng
 Date: 09/04/2022
 Function: localization for SIL Radar sensor array. Using Nengo_loihi.

'''
 
import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_dl
import nengo_loihi
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot

import tensorflow as tf
from tensorflow.keras import layers, models

seed = 2
np.random.seed(seed)
rng = np.random.RandomState(seed)
#tf.random.set_seed(seed)

t_sim = 1.5

n_samp = 2400
n_pts = 500
n_chan = 6

n_signal = n_pts * n_chan

n_train = int(0.8 * n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp, n_signal))   #  nengo dl only accepts 1D data type
y_data = np.zeros((n_samp, 1))   #  gesture label

# process the raw data
with open("gesture_real_12_2400.txt", "r")  as in_file:
    data_lines = in_file.readlines()

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(" "))
    items = items.astype(int)
    
    #dat = np.reshape(items[:n_signal], (n_pts, n_chan), 'F')
    #dat = np.transpose(dat)
    #dat = np.reshape(dat, (1, -1))
    x_data[i, :] = items[:n_signal]
    #x_data[i, :] = dat
    y_data[i] = items[n_signal]    # 0-11 for sparse_catogorial_enthropy


# first random permutation
perm = rng.permutation(x_data.shape[0])
x_data = x_data[perm, :]
y_data = y_data[perm]

# divide the dataset into training and testing group
x_train = x_data[:n_train, :]
y_train = y_data[:n_train, :]
x_test = x_data[n_train:, :]
y_test = y_data[n_train:, :]

# generating more training data by timeshifting original samples
x_train_shift_1 = np.roll(x_train, -50, axis=1)
x_train_shift_1[:, -50:] = 0
x_train_shift_2 = np.roll(x_train, 20, axis=1)
x_train_shift_2[:, :20] = 0

#x_train = np.concatenate((x_train, x_train_shift_1, x_train_shift_2), axis=0)
x_train = np.concatenate((x_train, x_train_shift_1), axis=0)
#y_train = np.tile(y_train, (3, 1))
y_train = np.tile(y_train, (2, 1))

# Second permutation
perm = rng.permutation(x_train.shape[0])
x_train = x_train[perm, :]
y_train = y_train[perm]

#perm = rng.permutation(x_test.shape[0])
#x_test = x_test[perm, :]
#y_test = y_test[perm]


t_range = np.linspace(0, t_sim, n_pts)
spikes = np.reshape(x_train[0], (n_pts, n_chan), 'F')
#print(t_range.shape)
#print(spikes.shape)
#plt.figure()
#rasterplot(t_range, spikes)
#plt.title(str(y_train[0]))
#plt.show()

x_train = x_train[:, None, :]
y_train = y_train[:, None, :]
x_test = x_test[:, None, :]
y_test = y_test[:, None, :]

print(x_train.shape)
print(y_train.shape)

# Build a CNN using Tensorflow first
inp = tf.keras.Input(shape=(6, 500, 1), name="input")

to_spikes = layers.Activation(tf.nn.relu)(inp)

pool0 = layers.AveragePooling2D((1, 2))(to_spikes)

conv0 = layers.Conv2D(
    filters=32, 
    kernel_size=(1, 16),
    #strides=(1, 4),
    activation=tf.nn.relu,
)(pool0)

# Max Pooling Layer 0
pool1 = layers.AveragePooling2D((1, 4))(conv0)

# Convolutional Layer 1 
conv1 = layers.Conv2D(
    filters=32,
    kernel_size=(3, 16),
    #strides=(1, 2),
    activation=tf.nn.relu,
)(pool1)

pool2 = layers.AveragePooling2D((1, 2))(conv1)

# Convolutional Layer 2 
conv2 = layers.Conv2D(
    filters=48,
    kernel_size=(4, 8),
    activation=tf.nn.relu,
)(conv1)

# Max Pooling Layer 1
#pool3 = layers.AveragePooling2D((1, 2))(conv2)

# Flatten
flatten = layers.Flatten()(conv2)

# Dense layer 0
dense = layers.Dense(180, 
    activation=tf.nn.relu,
)(flatten)

# Output Layer (Dense)
outp = layers.Dense(12)(dense)

model = models.Model(inputs=inp, outputs=outp)
model.summary()

# NengoDL Converter. Keras --> Nengo
converter = nengo_dl.Converter(model)

do_training = False
if do_training:
    with nengo_dl.Simulator(converter.net, minibatch_size=20) as sim:
        # run training
        sim.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],
        )
        sim.fit(
            {converter.inputs[inp]: x_train},
            {converter.outputs[outp]: y_train},
            validation_data=(
                {converter.inputs[inp]: x_test},
                {converter.outputs[outp]: y_test},
            ),
            epochs=10,
        )

        # save the parameters to file
        sim.save_params("./gesture_avg_pool_params")

def run_network(
    activation,
    params_file="gesture_avg_pool_params",
    n_steps=30,
    scale_firing_rates=1,
    synapse=None,
    n_test=400,
):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        swap_activations={tf.nn.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
    )

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[outp]

    # add a probe to the first convolutional layer to record activity.
    # we'll only record from a subset of neurons, to save memory.
    sample_neurons = np.linspace(
        0,
        np.prod(conv2.shape[1:]),
        200,
        endpoint=False,
        dtype=np.int32,
    )
    with nengo_converter.net:
        conv_probe = nengo.Probe(nengo_converter.layers[conv2][sample_neurons])

    # repeat inputs for some number of timesteps
    tiled_x_test = np.tile(x_test, (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=10, progress_bar=False
    ) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_x_test})

    # compute accuracy on test data, using output of network on
    # last timestep
    predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
    accuracy = (predictions == y_test[:, 0, 0]).mean()
    print(f"Test accuracy: {100 * accuracy:.2f}%")

    # Confusion matrix
    confusion_mat = tf.math.confusion_matrix(y_test[:, 0, 0], predictions)
    cfs_mat = np.zeros((12, 12))
    for i in range(confusion_mat.shape[1]):
        s = sum(confusion_mat[i, :])
        cfs_mat[i, :] = confusion_mat[i, :] / s
    for i in range(12):
        for j in range(12):
            print("%0.2f " % cfs_mat[i][j])
        print('\n')

    # plot the results
    for ii in range(3):
        spikes = np.reshape(x_test[ii], (n_pts, n_chan), 'F')
        
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Input Spikes")
        rasterplot(t_range, spikes)

        plt.subplot(1, 3, 2)
        scaled_data = data[conv_probe][ii] * scale_firing_rates
        if isinstance(activation, nengo.SpikingRectifiedLinear):
            scaled_data *= 0.001
            rates = np.sum(scaled_data, axis=0) / (n_steps * nengo_sim.dt)
            plt.ylabel("Number of spikes")
        else:
            rates = scaled_data
            plt.ylabel("Firing rates (Hz)")
        plt.xlabel("Timestep")
        plt.title(
            f"Neural activities (conv0 mean={rates.mean():.1f} Hz, "
            f"max={rates.max():.1f} Hz)"
        )
        plt.plot(scaled_data)

        plt.subplot(1, 3, 3)
        plt.title("Output predictions")
        plt.plot(tf.nn.softmax(data[nengo_output][ii]))
        plt.legend([str(j) for j in range(12)], loc="upper left")
        plt.xlabel("Timestep")
        plt.ylabel("Probability")

        plt.tight_layout()
        
#run_network(activation=nengo.RectifiedLinear(), n_steps=10)

run_network(activation=nengo.SpikingRectifiedLinear(), scale_firing_rates=33, 
    n_steps=80, synapse=0.01)
