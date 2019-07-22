import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras import regularizers
from highway import *
#from keras.utils import plot_model
import numpy as NP
import matplotlib.pyplot as plt
# Load data
path_to_data = ''
n_batches_train = 100
n_batches_test = 50
train_offset = 0
n_batches_train_test = n_batches_train + n_batches_test

raw_data = []
states = []
controls = []
params = []

x_train = []
y_train = []
x_test = []
y_test = []

nx = 2
nu = 2
np = 1
nt = 1

steps_real_data = int(1/0.005 * 2)

for i in range(train_offset, train_offset + n_batches_train + n_batches_test):
    # Format is: |time|states|controls|params|
    # Try to load data in case the solution was feasible
    raw_data.append(NP.load(path_to_data+ "data_batch_v2_" + str(i+0) + ".npy"))


for i in range(n_batches_train_test):
    # remove the offset of one position in the state-control vector
    # Take points only every XXX steps because of accurate sampling
    states.append(raw_data[i][steps_real_data*3:-1:steps_real_data,nt:nt+nx])
    controls.append(raw_data[i][1+steps_real_data*3::steps_real_data,nt+nx:nt+nx+nu])
    params.append(raw_data[i][steps_real_data*3:-1:steps_real_data,nt+nx+nu:nt+nx+nu+np])


# Use all data without taking into account actual batches (later for RNN)

x_train = NP.vstack(states[0:n_batches_train])
p_train = NP.vstack(params[0:n_batches_train])
x_train = NP.append(x_train,p_train, axis = 1)
y_train = NP.vstack(controls[0:n_batches_train])

# Try including two past steps as inputs
# for i in range(len(x_train))
#     if i < n_recurrent:
#         x_train_recurrent[i] =

x_test = NP.vstack(states[n_batches_train:n_batches_train_test])
p_test = NP.vstack(params[n_batches_train:n_batches_train_test])
x_test = NP.append(x_test,p_test, axis = 1)
y_test = NP.vstack(controls[n_batches_train:n_batches_train_test])

u_lb = NP.array([0.2, 30])
u_ub = NP.array([0.8, 100])

x_lb = NP.array([-50, -100, 500]) # the third component is the scaling factor for the parameter
x_ub = NP.array([50, 400, 3000])
# Scale the outputs between 0 and 1 based on input bounds for better loss management
y_train = (y_train - u_lb) / (u_ub - u_lb)
y_test  = (y_test  - u_lb) / (u_ub - u_lb)

x_train = (x_train - x_lb) / (x_ub - x_lb)
x_test  = (x_test  - x_lb) / (x_ub - x_lb)

main_input = Input(shape=(nx+np,))
act = 'relu'
# x = Dense(100, activation = 'tanh')(main_input)
l1_pen = 0.0000
x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(main_input)
# x = Dense(10, activation = 'tanh')(x)
# x = Dense(5, activation = 'tanh')(x)

x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)
x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)
x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)
x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)

# x = highway_layers(x, 5, activation="sigmoid")

# x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)
# x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)
# x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)
# x = Dense(10, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)
# x = Dense(10, activation = act)(x)
# x = Dense(10, activation = act)(x)
# x = Dense(10, activation = act)(x)
#x = Dense(20, activation = 'sigmoid')(x)
main_output = Dense(nu, activation = 'linear', kernel_regularizer=regularizers.l2(l1_pen))(x)

model = Model(inputs = main_input, outputs = main_output)
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.


sgd = SGD(lr=0.1, decay=1e-2, momentum=0.9, nesterov=True)
model.compile(loss='mse',
              optimizer='adam')
              #optimizer='adam')

history = model.fit(x_train, y_train,
          epochs=300,
          batch_size=int(len(x_train)/15),
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=128)
print("The test score is: ", score)


# If wanted save model
model_json = model.to_json()
with open("model_revised_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_revised_1.h5")
print("Saved model to disk")


# To load
# y_pred = []
# for ii in range(len(x_train)):
#     y_pred.append(model.evaluate(x_train[ii]))
# later...
y_pred = model.predict(x_train)
train_error = abs(y_pred - y_train)
print('max error: ', NP.max(train_error,axis = 0))
print('mean error: ', NP.mean(train_error,axis = 0))

plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss (mean squared error)')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

history_np = NP.array([history.history['loss'],history.history['val_loss']])
NP.save("history_np_random" + str(n_batches_train), history_np)
print("Exporting data for history as ''" + "history_np_random" + str(n_batches_train) + "''")
# print('max error unscaled: ', NP.max(abs((y_pred - y_train) * (u_ub - u_lb) + u_lb),axis = 0))
# print('mean error unscaled: ', NP.mean(abs((y_pred - y_train) * (u_ub - u_lb) + u_lb),axis = 0))
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
