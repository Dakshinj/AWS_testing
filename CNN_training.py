import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time

start_time = time.time()
# Preparing Train data
# CNN for car = 0, person = 1, dog = 2, airplane = 3 - 600 examples each for training
# cat = 4, flower = 5, motorbike = 6, fruit = 7
# training_x = (example,d,d,d), training_y = (example,d) -- > ofter one-hot encoding
train_example = 121 # train_set size
# Car ------------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_car/car_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_car/car_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_car/car_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    if(i == 1):
        train_x = my_image
        train_y = np.array([0]).reshape(1,1)
    else:
        train_x = np.append(train_x, my_image, axis = 0)
        temp = np.array([0]).reshape(1,1)
        train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)
# Person ----------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_person/person_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_person/person_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_person/person_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    train_x = np.append(train_x, my_image, axis = 0)
    temp = np.array([1]).reshape(1,1)
    train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)
# dog --------------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_dog/dog_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_dog/dog_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_dog/dog_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    
    train_x = np.append(train_x, my_image, axis = 0)
    temp = np.array([2]).reshape(1,1)
    train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)
# airplane ----------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_airplane/airplane_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_airplane/airplane_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_airplane/airplane_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    train_x = np.append(train_x, my_image, axis = 0)
    temp = np.array([3]).reshape(1,1)
    train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)
# cat --------------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_cat/cat_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_cat/cat_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_cat/cat_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    train_x = np.append(train_x, my_image, axis = 0)
    temp = np.array([4]).reshape(1,1)
    train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)
# flower ------------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_flower/flower_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_flower/flower_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_flower/flower_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    train_x = np.append(train_x, my_image, axis = 0)
    temp = np.array([5]).reshape(1,1)
    train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)
# motorbike ----------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_motorbike/motorbike_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_motorbike/motorbike_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_motorbike/motorbike_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    train_x = np.append(train_x, my_image, axis = 0)
    temp = np.array([6]).reshape(1,1)
    train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)
# fruit -------------------------
for i in range(1, train_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_fruit/fruit_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_fruit/fruit_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_fruit/fruit_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    train_x = np.append(train_x, my_image, axis = 0)
    temp = np.array([7]).reshape(1,1)
    train_y = np.append(train_y, temp, axis = 0)
print(train_x.shape)
print(train_y.shape)

train_x_backup = train_x
train_y_backup = train_y
nb_classes = 8
targets = train_y.reshape(-1)
train_y = np.eye(nb_classes)[targets]
print(train_y.shape)
train_x = train_x/255.


# Prepare Test Data
# CNN for car = 0, person = 1, dog = 2, airplane = 3 - 100 examples each for training
# cat = 4, flower = 5, motorbike = 6, fruit = 7
# training_x = (example,d,d,d), training_y = (example,d) -- > ofter one-hot encoding
test_example = train_example + 10 + 20 # train_set size
# Car ------------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_car/car_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_car/car_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_car/car_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)
    if(i == (train_example+10)):
        test_x = my_image
        test_y = np.array([0]).reshape(1,1)
    else:
        test_x = np.append(test_x, my_image, axis = 0)
        temp = np.array([0]).reshape(1,1)
        test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)
# person ----------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_person/person_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_person/person_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_person/person_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    test_x = np.append(test_x, my_image, axis = 0)
    temp = np.array([1]).reshape(1,1)
    test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)
# dog -------------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_dog/dog_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_dog/dog_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_dog/dog_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    test_x = np.append(test_x, my_image, axis = 0)
    temp = np.array([2]).reshape(1,1)
    test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)
# airplane ---------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_airplane/airplane_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_airplane/airplane_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_airplane/airplane_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    test_x = np.append(test_x, my_image, axis = 0)
    temp = np.array([3]).reshape(1,1)
    test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)
# cat -------------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_cat/cat_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_cat/cat_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_cat/cat_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    test_x = np.append(test_x, my_image, axis = 0)
    temp = np.array([4]).reshape(1,1)
    test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)
# flower -----------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_flower/flower_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_flower/flower_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_flower/flower_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    test_x = np.append(test_x, my_image, axis = 0)
    temp = np.array([5]).reshape(1,1)
    test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)
# motorbike --------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_motorbike/motorbike_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_motorbike/motorbike_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_motorbike/motorbike_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    test_x = np.append(test_x, my_image, axis = 0)
    temp = np.array([6]).reshape(1,1)
    test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)
# fruit -------------------------
for i in range((train_example+10), test_example):
    if(i <= 999 & i > 99):
        image_address = "natural_images_v2/z_fruit/fruit_0" + str(i) + ".jpg"
    if(i <= 99 & i > 9):
        image_address = "natural_images_v2/z_fruit/fruit_00" + str(i) + ".jpg"
    if(i <= 9 & i > 0):
        image_address = "natural_images_v2/z_fruit/fruit_000" + str(i) + ".jpg"
    image = Image.open(image_address)
    my_image = np.array(image.resize((150, 150), Image.ANTIALIAS))
    my_image = my_image.reshape(1,150,150,3)

    test_x = np.append(test_x, my_image, axis = 0)
    temp = np.array([7]).reshape(1,1)
    test_y = np.append(test_y, temp, axis = 0)
print(test_x.shape)
print(test_y.shape)

test_x_backup = test_x
test_y_backup = test_y
nb_classes = 8
targets = test_y.reshape(-1)
test_y = np.eye(nb_classes)[targets]
print(test_y.shape)
test_x = test_x/255.


tf.reset_default_graph()
print(train_x.shape[1])
print(train_x.shape[2])
print(train_x.shape[3])
print(train_y.shape[1])
tf_x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1], train_x.shape[2], train_x.shape[3]), name='tf_x')
tf_y = tf.placeholder(tf.float32, shape=(None, train_y.shape[1]), name='tf_y')
print(tf_x)
print(tf_y)

# weights -----------------
# w1 - 8 filters
# w2 - 16 filter
#W1 = tf.get_variable('W1', shape=(4,4,3,8), initializer=tf.contrib.layers.xavier_initializer(seed=0))
#W2 = tf.get_variable('W2', shape=(2,2,8,16), initializer=tf.contrib.layers.xavier_initializer(seed=0))
W1 = tf.get_variable('W1', shape=(8,8,3,64), initializer=tf.contrib.layers.xavier_initializer(seed=0))
W2 = tf.get_variable('W2', shape=(6,6,64,32), initializer=tf.contrib.layers.xavier_initializer(seed=0))
W3 = tf.get_variable('W3', shape=(4,4,32,8), initializer=tf.contrib.layers.xavier_initializer(seed=0))
W4 = tf.get_variable('W4', shape=(2,2,8,12), initializer=tf.contrib.layers.xavier_initializer(seed=0))

parameters = {"W1": W1, "W2": W2, "W3" : W3, "W4" : W4}


W1 = parameters['W1']
W2 = parameters['W2']
W3 = parameters['W3']
W4 = parameters['W4']

# Block - 1
#stride of 1 and padding 'SAME'
Z1 = tf.nn.conv2d(tf_x, W1, strides=[1, 1, 1, 1], padding='SAME')
# RELU - Activation
A1 = tf.nn.relu(Z1)
# window 8x8, stride 8, padding 'SAME'
P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
# Block - 2
# stride 1, padding 'SAME'
Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
# RELU
A2 = tf.nn.relu(Z2)
# stride 4 and padding 'SAME'
P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
# Block - 3
# stride 1, padding 'SAME'
Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
# RELU
A3 = tf.nn.relu(Z3)
# stride 2 and padding 'SAME'
P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Block - 4
# stride 1, padding 'SAME'
Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
# RELU
A4 = tf.nn.relu(Z4)
# stride 2 and padding 'SAME'
P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully Connected Block
# FLATTEN 
P4 = tf.contrib.layers.flatten(P4)
# FULLY-CONNECTED without non-linear activation function.
# 8 node - 8 classes
Z5 = tf.contrib.layers.fully_connected(P4, 8, activation_fn=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=tf_y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

import math

seed = 3
(m, n_H0, n_W0, n_C0) = train_x.shape             
n_y = train_y.shape[1]                            
costs = []
num_epochs = 50
minibatch_size = 256
print_cost = True

# Initialize all the variables globally
init = tf.global_variables_initializer()
     
# Start the session to compute the tensorflow graph
with tf.Session() as sess:
        
    sess.run(init)
        
    for epoch in range(num_epochs):

        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(train_x, train_y, minibatch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            _ , temp_cost = sess.run([optimizer, cost], feed_dict={tf_x:minibatch_X, tf_y:minibatch_Y})
    
            minibatch_cost += temp_cost / num_minibatches
                
            if print_cost == True:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(0.001))
    plt.show()
    # Calculate the correct predictions
    saver = tf.train.Saver()
    saver.save(sess, 'natural_image_model') 
    predict_op = tf.argmax(Z5,1)
    correct_prediction = tf.equal(predict_op, tf.argmax(tf_y,1))
    end_time = time.time()
    print("Total time taken:")
    print(end_time-start_time)
    
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:")
    train_accuracy = accuracy.eval({tf_x: train_x, tf_y: train_y})
    print("Train Accuracy:", train_accuracy)
    test_accuracy = accuracy.eval({tf_x: test_x, tf_y: test_y})
    print("Test Accuracy:", test_accuracy)

# save the weights
saver = tf.train.import_meta_graph('natural_image_model.meta')    
