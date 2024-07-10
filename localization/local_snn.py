#! /usr/bin/env python3

'''

 Filename: local_snn.py
 Author: Kai Zheng
 Date: 10/13/2022
 Function: localization for SIL Radar sensor array. 

'''
 
import numpy as np
import matplotlib.pyplot as plt
import math

import nengo
import nengo_dl
from nengo.utils.matplotlib import rasterplot

import tensorflow as tf
from tensorflow.keras import layers, models

seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

n_samp = 3594
n_pts = 500
n_chan = 12

n_total = n_pts * n_chan

n_traj = 5  # number of sampled location points

n_train = int(0.8 * n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp, n_total))   # nengo dl only accepts 1D data type
#y_data = np.zeros((n_samp, n_traj * 2))   # {x[t]}, {y[t]}
y_data = np.zeros((n_samp, n_traj * 4))   # {x[t]}, {y[t]}, {v_x[t]}, {v_y[t]}

with open("loc_3594_2s_4ms.txt", "r") as in_file:
    for i in range(n_samp):
        line = in_file.readline() 
        items = np.array(line.split(" "))

        # spike data as feature
        x_data[i, :] = items[0:n_total].astype(int)
    
        # location data as label
        locs = items[n_total:-1].astype(float)  #last item is '\n'
        
        # sample n_traj points from the data
        sel_x_start = int(n_pts / (n_traj + 1))
        sel_x_end = int(n_pts * n_traj / (n_traj + 1))
        sel_y_start = int(n_pts / (n_traj + 1))         + n_pts
        sel_y_end = int(n_pts * n_traj / (n_traj + 1))  + n_pts
        sel_vx_start = int(n_pts / (n_traj + 1))        + 2 * n_pts
        sel_vx_end = int(n_pts * n_traj / (n_traj + 1)) + 2 * n_pts
        sel_vy_start = int(n_pts / (n_traj + 1))        + 3 * n_pts
        sel_vy_end = int(n_pts * n_traj / (n_traj + 1)) + 3 * n_pts

        sel_x = (np.linspace(sel_x_start, sel_x_end, n_traj)).astype(int)
        sel_y = (np.linspace(sel_y_start, sel_y_end, n_traj)).astype(int)
        sel_vx = (np.linspace(sel_vx_start, sel_vx_end, n_traj)).astype(int)
        sel_vy = (np.linspace(sel_vy_start, sel_vy_end, n_traj)).astype(int)
        y_data[i, 0:n_traj]             = locs[sel_x]
        y_data[i, n_traj:2*n_traj]      = locs[sel_y]
        y_data[i, 2*n_traj:3*n_traj]    = locs[sel_vx]
        y_data[i, 3*n_traj:]            = locs[sel_vy]

# Random permutation
perm = rng.permutation(x_data.shape[0])
x_data = x_data[perm, :]
y_data = y_data[perm, :]

# Divide the data into training group and test group
x_train = x_data[0:n_train, :] 
y_train = y_data[0:n_train, :]
x_test = x_data[n_train:, :]
y_test = y_data[n_train:, :]

# plot sample data 
t_range = np.linspace(0, 2.0, n_pts)
for i in range(0):
    sel = i+5
    spikes = np.reshape(x_train[sel, :n_total], (n_pts, n_chan))

    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(y_train[sel, 0:n_traj], y_train[sel, n_traj:2*n_traj])
    plt.title("Object Trajectory")
    plt.xlim([-3, 3])
    plt.ylim([0, 7])

    plt.subplot(1, 2, 2)
    rasterplot(t_range, spikes)
    plt.title("Spikes")

plt.show()

# Add time dimension
x_train = x_train[:, None, :] 
y_train = y_train[:, None, :]
x_test = x_test[:, None, :]
y_test = y_test[:, None, :]

print(x_train.shape)
print(y_train.shape)

# Building a neural network
# Input
inp = tf.keras.Input(shape=(12, n_pts, 1), name="input")

#pool0 = layers.AveragePooling2D((1, 4))(inp)
to_spikes = layers.Activation(tf.nn.relu)(inp)

# Convolutional Layer 0 
conv0 = layers.Conv2D(
    filters=64, 
    kernel_size=(2, 32),
    #strides=(2, 4),
    strides=(2, 1),
    activation=tf.nn.relu,
)(to_spikes)

pool1 = layers.AveragePooling2D((1, 4))(conv0)

# Convolutional Layer 1 
conv1 = layers.Conv2D(
    filters=96,
    kernel_size=(3, 24),
    #strides=(1, 2),
    activation=tf.nn.relu,
)(pool1)

# Average Pooling Layer 0
pool2 = layers.AveragePooling2D((1, 2))(conv1)

# Convolutional Layer 2 
conv2 = layers.Conv2D(
    filters=64,
    kernel_size=(4, 16),
    #strides=(1, 2),
    activation=tf.nn.relu,
)(pool2)

# Flatten
#flatten = layers.Flatten()(pool3)
flatten = layers.Flatten()(conv2)

# Dense layer 0
dense = layers.Dense(240, 
    activation=tf.nn.relu,
)(flatten)

# Output Layer (Dense)
outp = layers.Dense(n_traj * 4)(dense)

model = models.Model(inputs=inp, outputs=outp)
model.summary()

# NengoDL Converter. Keras --> Nengo
converter = nengo_dl.Converter(model)

# It's important to note that we are using standard (non-spiking) ReLU neurons at this point.
do_training = False
if do_training:
    with nengo_dl.Simulator(converter.net, minibatch_size=41) as sim:
        #weights = np.ones(n_traj * 4)
        #weights[:n_traj*2] = 0.1
        #print(weights)
        # run training
        sim.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            #loss_weights=weights,
            metrics=['mse'],
        )
        sim.fit(
            {converter.inputs[inp]: x_train},
            {converter.outputs[outp]: y_train},
            validation_data=(
                {converter.inputs[inp]: x_test},
                {converter.outputs[outp]: y_test},
            ),
            epochs=20,
        )

        # save the parameters to file
        sim.save_params("loc_params")

def run_network(
    activation,
    params_file="loc_params",
    n_steps=5,
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
    probe_neurons = np.linspace(
        0,
        np.prod(conv2.shape[1:]),
        200,
        endpoint=False,
        dtype=np.int32,
    )
    
    with nengo_converter.net:
        conv_probe = nengo.Probe(nengo_converter.layers[conv2][probe_neurons], label="conv_p")
        #conv_probe_1 = nengo.Probe(nengo_converter.layers[conv2][conv2_neurons], label="conv_p1")
        #to_spikes_probe = nengo.Probe(nengo_converter.layers[to_spikes][to_spikes_neurons])
        #out_probe = nengo.Probe(nengo_converter.outputs[outp], label="outp")
        #out_probe_filt = nengo.Probe(nengo_converter.outputs[outp], synapse=0.1, label="outp_filt")

    # repeat inputs for some number of timesteps
    tiled_x_test = np.tile(x_test[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=20, progress_bar=False
    ) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_x_test})

    # compute mse on test data, using output of network on
    # last timestep
    #print(data[nengo_output].shape)
    #print(y_test.shape)
    loc_pred = data[nengo_output][:, -1, 0:2*n_traj]
    loc_true = y_test[:n_test, -1, 0:2*n_traj]
    speed_pred = data[nengo_output][:, -1, 2*n_traj:4*n_traj]
    speed_true = y_test[:n_test, -1, 2*n_traj:4*n_traj]
    print(loc_pred.shape)
    print(loc_true.shape)
    #mse_loc = tf.metrics.mean_squared_error(loc_true, loc_pred)
    mse_loc = tf.metrics.mean_squared_error(loc_true, loc_pred)
    #print(np.mean(mse_loc))
    #print(mse_loc)
    print("Localization mse = %f." % np.mean(mse_loc))
    mse_speed = tf.metrics.mean_squared_error(speed_true, speed_pred)
    print("Speed mse = %f." % np.mean(mse_speed))

    rmse = 0
    mse = 0
    for i in range(n_test):
        for j in range(n_traj): 
            m = (loc_true[i, j] - loc_pred[i, j]) ** 2 + \
                (loc_true[i, j + n_traj] - loc_pred[i, j + n_traj]) ** 2
            r = math.sqrt(m)
            mse += m
            rmse += r
    mse = mse / (n_test * n_traj)
    rmse = rmse / (n_test * n_traj)
    print("my_mse = %f" % mse)
    print("my_rmse = %f" % rmse)
            
    #tops = np.argpartition(mse, -5)[-5:]
    #bots = np.argpartition(a, -5)[-5:]

    # plot the results
    for ii in range(5):
        #jj = tops[ii]
        jj = ii + 10

        spikes = np.reshape(x_test[jj, :n_total], (n_pts, n_chan))

        plt.figure(figsize=(4, 4))
        rasterplot(t_range, spikes)
        plt.rcParams.update({'font.size':12})
        plt.tight_layout()

        loc_x_true = y_test[jj][-1][0:n_traj]
        loc_y_true = y_test[jj][-1][n_traj:2*n_traj]
        v_x_true = y_test[jj][-1][2*n_traj:3*n_traj]
        v_y_true = y_test[jj][-1][3*n_traj:4*n_traj]

        loc_x_pred = data[nengo_output][jj][-1][0:n_traj]
        loc_y_pred = data[nengo_output][jj][-1][n_traj:2*n_traj]
        v_x_pred = data[nengo_output][jj][-1][2*n_traj:3*n_traj]
        v_y_pred = data[nengo_output][jj][-1][3*n_traj:4*n_traj]
       
        plt.figure(figsize=(4, 4))
        #plt.title("Object Trajectory")
        plt.plot(loc_x_true, loc_y_true, label='True')
        plt.plot(loc_x_pred, loc_y_pred, '--', label='Pred')
        sel = 2
        #plt.arrow(loc_x_true[sel], loc_y_true[sel], v_x_true[sel], v_y_true[sel],
        #        width=0.015, color='green')
        #plt.arrow(loc_x_pred[sel], loc_y_pred[sel], v_x_pred[sel], v_y_pred[sel], 
        #        width=0.015, color='red')
        plt.legend(loc='lower right')
        plt.xlim([-3, 3])
        plt.ylim([0, 8])

        loc_x1_pred_t = data[nengo_output][jj][:, 0]
        loc_x2_pred_t = data[nengo_output][jj][:, n_traj - 1]
        loc_y1_pred_t = data[nengo_output][jj][:, n_traj]
        loc_y2_pred_t = data[nengo_output][jj][:, 2 * n_traj - 1]

        #plt.subplot(1, 2, 2)
        #rasterplot(t_range, spikes)
        #plt.plot(loc_x1_pred_t, loc_y1_pred_t)
        #plt.plot(loc_x2_pred_t, loc_y2_pred_t)
        #plt.scatter(loc_x_true[0], loc_y_true[0], label='Start')
        #plt.scatter(loc_x_true[n_traj-1], loc_y_true[n_traj-1], label='End')
        #plt.title("Localization Result vs. Timestep")
        #plt.xlim([-3, 3])
        #plt.ylim([0, 8])
        #plt.legend()

        #plt.subplot(1, 3, 3)
        #scaled_conv_probe_data = data[conv_probe][ii] * scale_firing_rates
        #if isinstance(activation, nengo.SpikingRectifiedLinear):
        #    scaled_conv_probe_data *= 0.001
        #    rates = np.sum(scaled_conv_probe_data, axis=0) / (n_steps * nengo_sim.dt)
        #    plt.ylabel("Number of spikes")
        #else:
        #    rates = scaled_conv_probe_data
        #    plt.ylabel("Firing rates (Hz)")
        #plt.xlabel("Timestep")
        #plt.title(
        #    f"mean={rates.mean():.1f} Hz, "
        #    f"max={rates.max():.1f} Hz"
        #)
        #plt.plot(scaled_conv_probe_data)
        plt.tight_layout()

#run_network(activation=nengo.RectifiedLinear(), n_steps=2, n_test=400) 
run_network(activation=nengo.SpikingRectifiedLinear(), n_steps=300, scale_firing_rates=180, synapse=0.015, n_test=100) 
plt.show()
