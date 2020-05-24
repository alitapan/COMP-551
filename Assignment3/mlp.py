import time
for epoch in range(2):  # loop over the dataset multiple times
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = np.array(inputs)
        labels = np.array(labels)
        num_samples = inputs.shape[0]
        inputs = np.reshape(inputs, (num_samples, inputs.shape[1]*inputs.shape[2]*inputs.shape[3]))
        labels = labels.reshape(-1,1)
        onehotencoder = OneHotEncoder(categories='auto')
        labels = onehotencoder.fit_transform(labels)
        inputs = np.c_[np.ones((inputs.shape[0])), inputs]

        W, V = SGD(inputs, labels.toarray(), 30)
#         print(f"Loss:{cost(inputs, labels.toarray(), W, V)}")  # uncommentto check if working properly - should be small positive values decreasing
    print(f"Time elapsed: {time.time()-start_time}")

class Net:

    def __init__(self):
        self.layers = [] # store layers added to net
        self.hidden_units = [] # store hidden units of those layers
        self.Ws = {} # store weights of layer i at W[i]
        self.dWs = {} # store weight changes
        self.Zs = {} # store node values of layers (Z[0] = input, Z[-1] = output)
        self.dZs = {} # store changes to layer values
        self.costs = [] # store costs for graphing
        self.best_weights = {} # store weights corresponding to lowest cost
        self.new_Ws = {}
        self.accuracies = []
        pass

    def cost(self, X, Y):
        self.costZs = {} # equivalent to Zs, with different X
        self.costZs[0] = X
        for i in range(len(self.layers)): # forward propegate
            self.costZs[i+1] = getattr(Net, self.layers[i].activation)(self, np.dot(self.costZs[i], self.Ws[i])) # Z[i+1] = activation(Z[i]*W[i])
        R = np.dot(self.costZs[len(self.costZs)-2], self.Ws[len(self.costZs)-2])
        nll = -np.mean(np.sum(R*Y, 1) - self.logsumexp(R))
        return nll

    def logsumexp(self,Z):
        Zmax = np.max(Z,axis=1)[:, None]
        lse = Zmax + np.log(np.sum(np.exp(Z-Zmax), axis=1))[:, None]
        return lse

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1.0 - (x**2)

    def relu(self, x):
        return np.maximum(x,0)

    def relu_deriv(self, x):
        return (x>0).astype(x.dtype)

    def leakyrelu(self, x):
        return np.maximum(.01 * x,x)

    def leakyrelu_deriv(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = .01
        return dx

    def softmax(self, x):
        y = x - np.max(x)
        e_x = np.exp(y)
        return e_x / e_x.sum()

    def logistic(self, x):
        return (1/(1+np.exp(-x)))

    def logistic_deriv(self, x):
        return x*(1-x)

    def gradients(self, X, Y):
        self.Zs[0] = X
        for i in range(len(self.layers)): # forward propegate
            self.Zs[i+1] = getattr(Net, self.layers[i].activation)(0, np.dot(self.Zs[i], self.Ws[i]))
        N,D = X.shape

        dY = self.Zs[len(self.layers)] - Y # calculate incorrect
        self.dZs[len(self.layers)] = dY # set last item of dZs to # incorrect in output layer
        for i in range(len(self.layers)-1, 0, -1): # backpropegate (layers) -stops at 0 so input isnt calculated
            self.dZs[i] = np.dot(self.dZs[i+1], self.Ws[i].T)

        for i in range(len(self.layers)-1, -1, -1): # backpropegate (weights)
            if i == len(self.layers)-1: # differenct calculation for first layer according to given code
                self.dWs[i] = np.dot(self.Zs[i].T, self.dZs[i+1])
            else:
                self.dWs[i] = np.dot(self.Zs[i].T, self.dZs[i+1]*getattr(Net, self.layers[i].activation+"_deriv")(self, self.Zs[i+1]))

    def multiclass_predict(self, X, Ws): # same as cost
        self.testZs = {}
        self.testZs[0] = X
        for i in range(len(self.layers)):
            self.testZs[i+1] = getattr(Net, self.layers[i].activation)(self, np.dot(self.testZs[i], Ws[i]))
        Yh = self.testZs[len(self.testZs)-1]
        predict_arr = []
        for i in Yh:
            predict_arr.append(np.argmax(i)) # find predicted value
        return predict_arr

    def SGD(self, X, Y, lr=.1, eps=1e-3, max_iters=1000, batch_size=1, beta=0.99, x_test=None, y_test=None):
        start_time = time.time()
        N,D = X.shape
        N,K = Y.shape
        self.hidden_units.insert(0, D) # number of hidden units starts at D
        self.hidden_units.append(K) # ends at K
        self.best_cost = np.inf
        for layer in range(len(self.layers)): # initialize weight dict
            self.Ws[layer] = np.random.normal(0, 1/np.sqrt(self.hidden_units[layer]), (self.hidden_units[layer], self.hidden_units[layer+1]))
            self.new_Ws[layer] = np.zeros((self.hidden_units[layer], self.hidden_units[layer+1]))

        self.dWs[len(self.layers)-1] = np.inf*np.ones_like(self.Ws[len(self.layers)-1]) # for use in while loop
        t = 0
        new_start_time = time.time()
        while  (np.linalg.norm(self.dWs[len(self.layers)-1])>eps or np.linalg.norm(self.dWs[len(self.layers)-1])) != 0 and t < max_iters : #

            minibatch = np.random.randint(N, size=(batch_size)) # create batch
            self.gradients(X[minibatch,:],Y[minibatch]) # calculate gradients with batch
            for i in range(len(self.layers)):
                self.new_Ws[i] = (1-beta)*self.dWs[i]+beta*self.new_Ws[i]
                self.Ws[i] = self.Ws[i] - lr*self.new_Ws[i]
#             new_cost = self.cost(X, Y) # calculate cost of first 10,000 samples # X[:10000, :], Y[:10000]
#             if new_cost < self.best_cost:
#                 self.best_weights = self.Ws
#             self.costs.append(new_cost)
            predictions = self.multiclass_predict(x_test, self.Ws)
            self.accuracies.append(self.get_accuracy(predictions, y_test))
            t += 1
            if t % int(max_iters/40) == 0:
                print(f"Time to compute {int(max_iters/40)} iterations ({int(t)} total calculated out of {max_iters}, {(t/max_iters)*100: 0.1f}%): {time.time() - new_start_time:0.3f}s")
                new_start_time = time.time()
        print(f"Time to fit: {time.time() - start_time:0.3f}s")
        return self.Ws

    def add(self, layer):
        self.layers.append(layer)
        self.hidden_units.append(layer.hidden_units)

    def get_accuracy(self, predictions, y_test):
        count = 0
        cat_count = {}
        cat_total = {}
        n_y_train = y_test.toarray()
        for val in range(len(predictions)):
            cat_total[predictions[val]] = cat_total.get(predictions[val], 0) + 1
            if predictions[val] == np.argmax(n_y_train[val]):
                cat_count[predictions[val]] = cat_count.get(predictions[val], 0) + 1
                count+=1

        return(count/len(predictions))

class Layer:

    def __init__(self, activation, hidden_units):
        self.hidden_units = hidden_units
        self.activation = activation

neural_net = Net()
layer_1 = Layer('logistic', 10)
layer_2 = Layer('logistic', 10)
layer_3 = Layer('softmax', 10)
neural_net.add(layer_1)
neural_net.add(layer_2)
neural_net.add(layer_3)

from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

scaler = StandardScaler()

x_train = np.array(x_train)
#x_train = np.array(x_train_aug_list)
y_train = np.array(y_train)

num_samples = x_train.shape[0]
num_samples_test = x_test.shape[0]
x_train = np.reshape(x_train, (num_samples, x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
x_test = np.reshape(x_test, (num_samples_test, x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))
y_train = y_train.reshape(-1,1)
onehotencoder = OneHotEncoder(categories='auto')
y_train = onehotencoder.fit_transform(y_train)
x_train = np.c_[np.ones((x_train.shape[0])), x_train]
x_test = np.c_[np.ones((x_test.shape[0])), x_test]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

Ws = neural_net.SGD(x_train, y_train.toarray(), lr=.0005, max_iters=100000, beta=0.99)
print(neural_net.cost(x_train[:10000, :], y_train.toarray()[:10000]))
predictions = neural_net.multiclass_predict(x_test, Ws)

count = 0
cat_count = {}
cat_total = {}
n_y_train = y_test.toarray()
for val in range(len(predictions)):
    cat_total[predictions[val]] = cat_total.get(predictions[val], 0) + 1
    if predictions[val] == np.argmax(n_y_train[val]):
        cat_count[predictions[val]] = cat_count.get(predictions[val], 0) + 1
        count+=1

print(count/len(predictions))

for key in cat_total.keys():
    print(cat_count[key] / cat_total[key])

acc_data = [46.60, 45.45, 48.14 ,45.75, 39.86]
time_data = [11.341, 5.227, 26.345, 12.222, 11.199]
n_groups = len(time_data)
# fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
plt.figure(figsize=(16,9))
rects1 = plt.bar(index+bar_width, time_data, bar_width, alpha=1, label='Train Time per 250 iterations', color='b')
rects2 = plt.bar(index, acc_data, bar_width, alpha=1, label='Test Accuracy', color='g')
plt.xlabel('Model')import time
for epoch in range(2):  # loop over the dataset multiple times
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = np.array(inputs)
        labels = np.array(labels)
        num_samples = inputs.shape[0]
        inputs = np.reshape(inputs, (num_samples, inputs.shape[1]*inputs.shape[2]*inputs.shape[3]))
        labels = labels.reshape(-1,1)
        onehotencoder = OneHotEncoder(categories='auto')
        labels = onehotencoder.fit_transform(labels)
        inputs = np.c_[np.ones((inputs.shape[0])), inputs]

        W, V = SGD(inputs, labels.toarray(), 30)
#         print(f"Loss:{cost(inputs, labels.toarray(), W, V)}")  # uncommentto check if working properly - should be small positive values decreasing
    print(f"Time elapsed: {time.time()-start_time}")

class Net:

    def __init__(self):
        self.layers = [] # store layers added to net
        self.hidden_units = [] # store hidden units of those layers
        self.Ws = {} # store weights of layer i at W[i]
        self.dWs = {} # store weight changes
        self.Zs = {} # store node values of layers (Z[0] = input, Z[-1] = output)
        self.dZs = {} # store changes to layer values
        self.costs = [] # store costs for graphing
        self.best_weights = {} # store weights corresponding to lowest cost
        self.new_Ws = {}
        self.accuracies = []
        pass

    def cost(self, X, Y):
        self.costZs = {} # equivalent to Zs, with different X
        self.costZs[0] = X
        for i in range(len(self.layers)): # forward propegate
            self.costZs[i+1] = getattr(Net, self.layers[i].activation)(self, np.dot(self.costZs[i], self.Ws[i])) # Z[i+1] = activation(Z[i]*W[i])
        R = np.dot(self.costZs[len(self.costZs)-2], self.Ws[len(self.costZs)-2])
        nll = -np.mean(np.sum(R*Y, 1) - self.logsumexp(R))
        return nll

    def logsumexp(self,Z):
        Zmax = np.max(Z,axis=1)[:, None]
        lse = Zmax + np.log(np.sum(np.exp(Z-Zmax), axis=1))[:, None]
        return lse

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1.0 - (x**2)

    def relu(self, x):
        return np.maximum(x,0)

    def relu_deriv(self, x):
        return (x>0).astype(x.dtype)

    def leakyrelu(self, x):
        return np.maximum(.01 * x,x)

    def leakyrelu_deriv(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = .01
        return dx

    def softmax(self, x):
        y = x - np.max(x)
        e_x = np.exp(y)
        return e_x / e_x.sum()

    def logistic(self, x):
        return (1/(1+np.exp(-x)))

    def logistic_deriv(self, x):
        return x*(1-x)

    def gradients(self, X, Y):
        self.Zs[0] = X
        for i in range(len(self.layers)): # forward propegate
            self.Zs[i+1] = getattr(Net, self.layers[i].activation)(0, np.dot(self.Zs[i], self.Ws[i]))
        N,D = X.shape

        dY = self.Zs[len(self.layers)] - Y # calculate incorrect
        self.dZs[len(self.layers)] = dY # set last item of dZs to # incorrect in output layer
        for i in range(len(self.layers)-1, 0, -1): # backpropegate (layers) -stops at 0 so input isnt calculated
            self.dZs[i] = np.dot(self.dZs[i+1], self.Ws[i].T)

        for i in range(len(self.layers)-1, -1, -1): # backpropegate (weights)
            if i == len(self.layers)-1: # differenct calculation for first layer according to given code
                self.dWs[i] = np.dot(self.Zs[i].T, self.dZs[i+1])
            else:
                self.dWs[i] = np.dot(self.Zs[i].T, self.dZs[i+1]*getattr(Net, self.layers[i].activation+"_deriv")(self, self.Zs[i+1]))

    def multiclass_predict(self, X, Ws): # same as cost
        self.testZs = {}
        self.testZs[0] = X
        for i in range(len(self.layers)):
            self.testZs[i+1] = getattr(Net, self.layers[i].activation)(self, np.dot(self.testZs[i], Ws[i]))
        Yh = self.testZs[len(self.testZs)-1]
        predict_arr = []
        for i in Yh:
            predict_arr.append(np.argmax(i)) # find predicted value
        return predict_arr

    def SGD(self, X, Y, lr=.1, eps=1e-3, max_iters=1000, batch_size=1, beta=0.99, x_test=None, y_test=None):
        start_time = time.time()
        N,D = X.shape
        N,K = Y.shape
        self.hidden_units.insert(0, D) # number of hidden units starts at D
        self.hidden_units.append(K) # ends at K
        self.best_cost = np.inf
        for layer in range(len(self.layers)): # initialize weight dict
            self.Ws[layer] = np.random.normal(0, 1/np.sqrt(self.hidden_units[layer]), (self.hidden_units[layer], self.hidden_units[layer+1]))
            self.new_Ws[layer] = np.zeros((self.hidden_units[layer], self.hidden_units[layer+1]))

        self.dWs[len(self.layers)-1] = np.inf*np.ones_like(self.Ws[len(self.layers)-1]) # for use in while loop
        t = 0
        new_start_time = time.time()
        while  (np.linalg.norm(self.dWs[len(self.layers)-1])>eps or np.linalg.norm(self.dWs[len(self.layers)-1])) != 0 and t < max_iters : #

            minibatch = np.random.randint(N, size=(batch_size)) # create batch
            self.gradients(X[minibatch,:],Y[minibatch]) # calculate gradients with batch
            for i in range(len(self.layers)):
                self.new_Ws[i] = (1-beta)*self.dWs[i]+beta*self.new_Ws[i]
                self.Ws[i] = self.Ws[i] - lr*self.new_Ws[i]
#             new_cost = self.cost(X, Y) # calculate cost of first 10,000 samples # X[:10000, :], Y[:10000]
#             if new_cost < self.best_cost:
#                 self.best_weights = self.Ws
#             self.costs.append(new_cost)
            predictions = self.multiclass_predict(x_test, self.Ws)
            self.accuracies.append(self.get_accuracy(predictions, y_test))
            t += 1
            if t % int(max_iters/40) == 0:
                print(f"Time to compute {int(max_iters/40)} iterations ({int(t)} total calculated out of {max_iters}, {(t/max_iters)*100: 0.1f}%): {time.time() - new_start_time:0.3f}s")
                new_start_time = time.time()
        print(f"Time to fit: {time.time() - start_time:0.3f}s")
        return self.Ws

    def add(self, layer):
        self.layers.append(layer)
        self.hidden_units.append(layer.hidden_units)

    def get_accuracy(self, predictions, y_test):
        count = 0
        cat_count = {}
        cat_total = {}
        n_y_train = y_test.toarray()
        for val in range(len(predictions)):
            cat_total[predictions[val]] = cat_total.get(predictions[val], 0) + 1
            if predictions[val] == np.argmax(n_y_train[val]):
                cat_count[predictions[val]] = cat_count.get(predictions[val], 0) + 1
                count+=1

        return(count/len(predictions))

class Layer:

    def __init__(self, activation, hidden_units):
        self.hidden_units = hidden_units
        self.activation = activation

neural_net = Net()
layer_1 = Layer('logistic', 10)
layer_2 = Layer('logistic', 10)
layer_3 = Layer('softmax', 10)
neural_net.add(layer_1)
neural_net.add(layer_2)
neural_net.add(layer_3)

from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

scaler = StandardScaler()

x_train = np.array(x_train)
#x_train = np.array(x_train_aug_list)
y_train = np.array(y_train)

num_samples = x_train.shape[0]
num_samples_test = x_test.shape[0]
x_train = np.reshape(x_train, (num_samples, x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
x_test = np.reshape(x_test, (num_samples_test, x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))
y_train = y_train.reshape(-1,1)
onehotencoder = OneHotEncoder(categories='auto')
y_train = onehotencoder.fit_transform(y_train)
x_train = np.c_[np.ones((x_train.shape[0])), x_train]
x_test = np.c_[np.ones((x_test.shape[0])), x_test]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

Ws = neural_net.SGD(x_train, y_train.toarray(), lr=.0005, max_iters=100000, beta=0.99)
print(neural_net.cost(x_train[:10000, :], y_train.toarray()[:10000]))
predictions = neural_net.multiclass_predict(x_test, Ws)

count = 0
cat_count = {}
cat_total = {}
n_y_train = y_test.toarray()
for val in range(len(predictions)):
    cat_total[predictions[val]] = cat_total.get(predictions[val], 0) + 1
    if predictions[val] == np.argmax(n_y_train[val]):
        cat_count[predictions[val]] = cat_count.get(predictions[val], 0) + 1
        count+=1

print(count/len(predictions))

for key in cat_total.keys():
    print(cat_count[key] / cat_total[key])

acc_data = [46.60, 45.45, 48.14 ,45.75, 39.86]
time_data = [11.341, 5.227, 26.345, 12.222, 11.199]
n_groups = len(time_data)
# fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
plt.figure(figsize=(16,9))
rects1 = plt.bar(index+bar_width, time_data, bar_width, alpha=1, label='Train Time per 250 iterations', color='b')
rects2 = plt.bar(index, acc_data, bar_width, alpha=1, label='Test Accuracy', color='g')
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Scores by model')
plt.xticks(index + bar_width-0.2, ('Control', 'Deeper Network', 'Wider Network', 'Deeper with same density', 'Sigmoid Activated'))
plt.legend()


plt.tight_layout()
# plt.savefig('Graphs/Models.png', dpi = 720, bbox_inches = 'tight')
plt.show()


test_arr = [0.001, 0.0005, 0.0001]

for learning_rate in test_arr:
    net = Net()
    net.add(Layer('relu', 100))
    net.add(Layer('relu', 300))
    net.add(Layer('softmax', y_train.shape[1]))
    Ws = net.SGD(x_train[:10000, :], y_train.toarray()[:10000], lr=learning_rate, max_iters=1000, beta=0.99, x_test=x_test, y_test=y_test)
    plt.plot(net.accuracies, label=f"Learning rate: {learning_rate}")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy over Iterations")
# plt.savefig('Graphs/Accuracies.png', dpi = 400, bbox_inches = 'tight')
plt.show()

"""#### Data Augmentation
In this section we applied data augmentation to part of the dataset in an attempt to improve accuracy. However, it had minimal effect on the performance, yet increased significantly the computing time. Therefore we decided to keep the original dataset for efficiency.
"""

# rotation augmentation
def rotate(x: tf.Tensor) -> tf.Tensor:
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

# flip augmentation
def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x

# zoom augmentation
def zoom(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

# function to plot images to illustrate data augmentation
def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()

augmentations = [flip, rotate]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

sample = (x_train[:8] / 255).astype(np.float32)
sampleset = tf.data.Dataset.from_tensor_slices(sample)

for f in augmentations:
  # Apply an augmentation only in 30% of the cases.
  sampleset = sampleset.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.7, lambda: f(x), lambda: x))
# Make sure that the values are still in [0, 1]
sampleset = sampleset.map(lambda x: tf.clip_by_value(x, 0, 1))

# Illustrate data augmentation with the first 8 images from CIFAR
plot_images(sampleset, n_images=8, samples_per_image=10)

# Now applying augmentation on the entire training dataset
x_train = (x_train / 255).astype(np.float32)
x_train_aug = tf.data.Dataset.from_tensor_slices(x_train)

for f in augmentations:
  # Apply an augmentation only in 30% of the cases.
  x_train_aug = x_train_aug.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.7, lambda: f(x), lambda: x))
# Make sure that the values are still in [0, 1]
x_train_aug = x_train_aug.map(lambda x: tf.clip_by_value(x, 0, 1))
x_train_aug_list = list(x_train_aug.take(-1).as_numpy_iterator()) #transform all data into list
plt.ylabel('Scores')
plt.title('Scores by model')
plt.xticks(index + bar_width-0.2, ('Control', 'Deeper Network', 'Wider Network', 'Deeper with same density', 'Sigmoid Activated'))
plt.legend()


plt.tight_layout()
# plt.savefig('Graphs/Models.png', dpi = 720, bbox_inches = 'tight')
plt.show()


test_arr = [0.001, 0.0005, 0.0001]

for learning_rate in test_arr:
    net = Net()
    net.add(Layer('relu', 100))
    net.add(Layer('relu', 300))
    net.add(Layer('softmax', y_train.shape[1]))
    Ws = net.SGD(x_train[:10000, :], y_train.toarray()[:10000], lr=learning_rate, max_iters=1000, beta=0.99, x_test=x_test, y_test=y_test)
    plt.plot(net.accuracies, label=f"Learning rate: {learning_rate}")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy over Iterations")
# plt.savefig('Graphs/Accuracies.png', dpi = 400, bbox_inches = 'tight')
plt.show()

"""#### Data Augmentation
In this section we applied data augmentation to part of the dataset in an attempt to improve accuracy. However, it had minimal effect on the performance, yet increased significantly the computing time. Therefore we decided to keep the original dataset for efficiency.
"""

# rotation augmentation
def rotate(x: tf.Tensor) -> tf.Tensor:
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

# flip augmentation
def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x

# zoom augmentation
def zoom(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

# function to plot images to illustrate data augmentation
def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()

augmentations = [flip, rotate]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

sample = (x_train[:8] / 255).astype(np.float32)
sampleset = tf.data.Dataset.from_tensor_slices(sample)

for f in augmentations:
  # Apply an augmentation only in 30% of the cases.
  sampleset = sampleset.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.7, lambda: f(x), lambda: x))
# Make sure that the values are still in [0, 1]
sampleset = sampleset.map(lambda x: tf.clip_by_value(x, 0, 1))

# Illustrate data augmentation with the first 8 images from CIFAR
plot_images(sampleset, n_images=8, samples_per_image=10)

# Now applying augmentation on the entire training dataset
x_train = (x_train / 255).astype(np.float32)
x_train_aug = tf.data.Dataset.from_tensor_slices(x_train)

for f in augmentations:
  # Apply an augmentation only in 30% of the cases.
  x_train_aug = x_train_aug.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.7, lambda: f(x), lambda: x))
# Make sure that the values are still in [0, 1]
x_train_aug = x_train_aug.map(lambda x: tf.clip_by_value(x, 0, 1))
x_train_aug_list = list(x_train_aug.take(-1).as_numpy_iterator()) #transform all data into list
