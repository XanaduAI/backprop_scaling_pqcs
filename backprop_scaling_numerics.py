import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from pennylane import pennylane as qml
import numpy as np
import optax
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, permutations, product, combinations
from model_utils import chunk_grad, chunk_loss, chunk_vmapped_fn


"""
This the numerics to accompany the article 'backpropagation scaling in parameterised quantum circuits'
The following code can be used to reproduce plots of the form of Figure 6 of the paper. 
"""

#################################### DATA GENERATION ####################################

def generate_data(dim, n, length, noise=0.):
    """
    Generate a bars and dots dataset
    :param dim: dimension of the data points
    :param n: number of data points
    :param length: length of the bars
    :param noise: std of independent gaussian noise
    :return: data (X) and labels (Y)
    """
    X = []
    Y = []
    for __ in range(n):
        start = np.random.randint(0, dim)
        x = np.ones(dim)
        if np.random.rand() < 0.5:
            bar = True
            Y.append(1)
        else:
            bar = False
            Y.append(-1)
        for i in range(length):
            if bar:
                x[(start + i) % dim] = -1
            else:
                x[(start + 2 * i) % dim] = -1
        X.append(x)
    X = np.array(X)
    X = X + np.random.normal(0, noise, X.shape)

    return X, np.array(Y)


dim = 16 #problem dimension
qubits = dim
seed = 852459
np.random.seed(seed)

#################################### FUNCTIONS USED IN MODEL GENERATION  ####################################

def cyclic_perm(a):
    "gets all cyclic permutations of a list"
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b

def seed_gens(weight, qubits=qubits, ops=['I', 'X']):
    """
    get all the seed generators up to a given pauli weight
    the seeds are fed into get_gens to get the symmetric generators
    """
    ops
    seeds = []
    for prod in product(ops, repeat=weight):
        seeds.append(list(prod) + ['I'] * (qubits - weight))
    return seeds[1:]


def seed_gens_doubles(ops=['I', 'X']):
    "get the seed generators that have weight 2 only"
    seeds = []
    for k in range(0, qubits - 1):
        seed = ops[1] + ops[0] * k + ops[1] + ops[0] * (qubits - k - 2)
        seeds.append(seed)
    return seeds


def get_gens(seeds):
    "get all unique equivariant generators from the a list of seeds"
    gens = []
    for seed in seeds:
        all_gens = cyclic_perm(seed)
        genlist = [''.join(all_gens[i]) for i in range(qubits)]
        genlist = list(dict.fromkeys(genlist))
        genlist.sort()
        if genlist not in gens:
            gens.append(genlist)
    return gens


#################################### MODEL DEFINITIONS ####################################


######## COMMUTING MODEL ##########


obs=qml.dot([1/qubits] * qubits, [qml.PauliZ(i) for i in range(qubits)])

seeds = seed_gens(qubits)
gens = get_gens(seeds)

#convert the gens to Pauli words and wires for more efficient use in pennylane
words_and_wires = [
        [(gen.replace("I", ""), [i for i, l in enumerate(gen) if l=="X"]) for gen in gen_list]
        for gen_list in gens]


#take only the generators with weight <=3
waw = []
for elem in words_and_wires:
    if len(elem[0][0])<=3:
        waw.append(elem)
words_and_wires = waw

num_gen_parallel = sum(len(sublist) for sublist in words_and_wires)
num_param_parallel = len(words_and_wires)
num_circuits_parallel = 16

print('number of generators for parallel model: ' + str(num_gen_parallel))
print('number of circuits for parallel model: ' + str(num_circuits_parallel))
print('number of parameters for parallel model: ' + str(num_param_parallel))

dev = qml.device('default.qubit',wires=qubits)
@qml.qnode(dev, interface='jax')
def parallel_model_eval(params,x):
    """
    Model used for evaluation but not for training. Sometimes it is useful to separate the two for efficiency
    reasons
    :param params: trainable parameters
    :param x: data input
    :return: expval corresponding to class label
    """
    #data encoding
    for q in range(qubits):
        qml.RY(x[q], wires=q)
    # apply the rotation for each equivariant generator
    for i, sublist in enumerate(words_and_wires):
        for word, wires in sublist:
            qml.PauliRot(params[i], pauli_word=word, wires=wires)
    return qml.expval(obs)


parallel_model_eval = jax.jit(parallel_model_eval)
parallel_model_eval = jax.vmap(parallel_model_eval,(None,0))

dev = qml.device('default.qubit',wires=qubits)
@qml.qnode(dev,interface='jax')
def parallel_model(params,x):
    for q in range(qubits):
        qml.RY(x[q],wires=q)
    #apply the rotation for each equivariant generator
    for i, sublist in enumerate(words_and_wires):
        for word, wires in sublist:
            qml.PauliRot(params[i], pauli_word=word, wires=wires)
    return qml.expval(obs)

parallel_model = jax.jit(parallel_model)
parallel_model = jax.vmap(parallel_model,(None,0))

def cost_parallel(params, input_data, labels):
    predictions = parallel_model(params['w'],input_data)
    return cross_entropy_loss(predictions,labels)

grad_parallel = jax.grad(cost_parallel)
grad_parallel = jax.jit(grad_parallel)


######## NONCOMMUTING MODEL ##########
layers = 4
obs=qml.dot([1/qubits] * qubits, [qml.PauliZ(i) for i in range(qubits)])

obsZZ = sum([qml.PauliZ(i)@qml.PauliZ((i+1)%qubits) for i in range(qubits)])

localz = get_gens(seed_gens(1,ops=['I','Z']))
localy = get_gens(seed_gens(1,ops=['I','Y']))
doublex = get_gens(seed_gens_doubles(ops=['I','X']))

z_words_and_wires = [[(gen.replace("I", ""), [i for i, l in enumerate(gen) if l=="Z"]) for gen in gen_list]
        for gen_list in localz]
y_words_and_wires = [[(gen.replace("I", ""), [i for i, l in enumerate(gen) if l=="Y"]) for gen in gen_list]
        for gen_list in localy]
x_words_and_wires = [[(gen.replace("I", ""), [i for i, l in enumerate(gen) if l=="X"]) for gen in gen_list]
        for gen_list in doublex]

general_words_and_wires = z_words_and_wires+y_words_and_wires+x_words_and_wires
num_gens_per_layer = sum(len(sublist) for sublist in general_words_and_wires)
num_gens_general = num_gens_per_layer*layers
num_param_per_layer = len(general_words_and_wires)
num_param_general = num_param_per_layer*layers
num_circuits_general = num_gens_general

print('number of generators for general model: ' + str(num_gens_general))
print('number of circuits for general model: ' + str(num_circuits_general))
print('number of parameters for general model: ' + str(num_param_general))


dev = qml.device('default.qubit',wires=qubits)
@qml.qnode(dev,  interface='jax')
def general_model_eval(params,x):
    for q in range(qubits):
        qml.RY(x[q],wires=q)
    #apply the rotation for each equivariant generator
    for l in range(layers):
        for i, sublist in enumerate(general_words_and_wires):
            for word, wires in sublist:
                qml.PauliRot(params[l*num_param_per_layer+i], pauli_word=word, wires=wires)
    return qml.expval(obs)

general_model_eval = jax.jit(general_model_eval)
general_model_eval = jax.vmap(general_model_eval,(None,0))

dev = qml.device('default.qubit',wires=qubits)
@qml.qnode(dev,  interface='jax')
def general_model(params,x):
    for q in range(qubits):
        qml.RY(x[q],wires=q)
    #apply the rotation for each equivariant generator
    for l in range(layers):
        for i, sublist in enumerate(general_words_and_wires):
            for word, wires in sublist:
                qml.PauliRot(params[l*num_param_per_layer+i], pauli_word=word, wires=wires)
    return qml.expval(obs)

general_model= jax.jit(general_model)
general_model= jax.vmap(general_model,(None,0))

def cost_general(params, input_data, labels):
    predictions = general_model(params['w'],input_data)
    return cross_entropy_loss(predictions,labels)

grad_general = jax.grad(cost_general)
grad_general = jax.jit(grad_general)

######## QUANTUM CONVOLUTIONAL MODEL ##########

def QCNN_block(params, wires):
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def pooling(params, wires):
    qml.CRZ(params[0], wires=wires)
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=wires)
    qml.PauliX(wires=wires[0])


n_params_block = 10
n_params_layer = 12
n_layers_qcnn = int(np.log2(qubits))

dev = qml.device('default.qubit', wires=qubits)

@qml.qnode(dev, interface="jax")
def QCNN_eval(params, x):
    count = 0
    wires = range(qubits)
    for q in range(qubits):
        qml.RY(x[q], wires=q)
    for j in range(n_layers_qcnn):
        for i in range(0, qubits // (2 ** j), 2):
            QCNN_block(params[count:count+10], wires=[wires[(2 ** j - 1) + 2 ** j * i], wires[
                ((2 ** j - 1) + 2 ** (j) * i + (2 ** j)) % qubits]])
        if j != int(np.log2(qubits)) - 1:
            for i in range(1, qubits // (2 ** j), 2):
                QCNN_block(params[count:count+10], wires=[wires[(2 ** j - 1) + 2 ** j * i], wires[
                    ((2 ** j - 1) + 2 ** (j) * i + (2 ** j)) % qubits]])
        count = count+10
        for i in range(0, qubits // (2 ** j), 2):
            pooling(params[count:count+2], wires=[wires[(2 ** j - 1) + 2 ** j * i], wires[
                ((2 ** j - 1) + 2 ** (j) * i + (2 ** j)) % qubits]])
        count = count+2
    return qml.expval(qml.PauliZ(qubits - 1))

QCNN_eval = jax.jit(QCNN_eval)
QCNN_eval = jax.vmap(QCNN_eval, (None, 0))

dev = qml.device('default.qubit', wires=qubits)

@qml.qnode(dev, interface="jax")
def QCNN(params, x):
    count = 0
    wires = range(qubits)
    for q in range(qubits):
        qml.RY(x[q], wires=q)
    for j in range(n_layers_qcnn):
        for i in range(0, qubits // (2 ** j), 2):
            QCNN_block(params[count:count+10], wires=[wires[(2 ** j - 1) + 2 ** j * i], wires[
                ((2 ** j - 1) + 2 ** (j) * i + (2 ** j)) % qubits]])
        if j != int(np.log2(qubits)) - 1:
            for i in range(1, qubits // (2 ** j), 2):
                QCNN_block(params[count:count+10], wires=[wires[(2 ** j - 1) + 2 ** j * i], wires[
                    ((2 ** j - 1) + 2 ** (j) * i + (2 ** j)) % qubits]])
        count = count+10
        for i in range(0, qubits // (2 ** j), 2):
            pooling(params[count:count+2], wires=[wires[(2 ** j - 1) + 2 ** j * i], wires[
                ((2 ** j - 1) + 2 ** (j) * i + (2 ** j)) % qubits]])
        count = count+2
    return qml.expval(qml.PauliZ(qubits - 1))

QCNN = jax.jit(QCNN)
QCNN = jax.vmap(QCNN, (None, 0))

def cost_QCNN(params, input_data, labels):
    predictions = QCNN(params['w'], input_data)
    return cross_entropy_loss(predictions, labels)

grad_QCNN = jax.grad(cost_QCNN)
grad_QCNN = jax.jit(grad_QCNN)

num_param_QCNN = n_layers_qcnn * n_params_layer
num_gen_QCNN = 29 * n_params_block + 15 * 2  # For the 16 qubit model
num_circuits_QCNN = 29 * (8*2+2*4) + 15*4

print('number of generators for QCNN model: ' + str(num_gen_QCNN))
print('number of parameters for QCNN model: ' + str(num_param_QCNN))

######## SEPARABLE MODEL ##########

obs=qml.dot([1/qubits] * qubits, [qml.PauliZ(i) for i in range(qubits)])

dev = qml.device('default.qubit',wires=qubits)
@qml.qnode(dev, interface='jax')
def separable_model(params,x):
    for q in range(qubits):
        qml.RY(x[q],wires=q)
    #apply the rotation for each equivariant generator
    for q in range(qubits):
        qml.Rot(params[3*q],params[3*q+1],params[3*q+2],wires=q)
    return qml.expval(obs)

separable_model = jax.jit(separable_model)
separable_model = jax.vmap(separable_model,(None,0))

def cost_separable(params, input_data, labels):
    predictions = separable_model(params['w'],input_data)
    return cross_entropy_loss(predictions,labels)

grad_separable = jax.grad(cost_separable)
grad_separable = jax.jit(grad_separable)


#################################### TRAINING AND EVAL FUNCTIONS ####################################

def square_loss(predictions, labels):
    """Square loss."""
    loss = jnp.sum((labels-predictions)**2)
    loss = loss/len(labels)
    return loss

def cross_entropy_loss(predictions, labels):
    labels = jax.nn.relu(labels)  # convert to 0,1
    return jnp.mean(optax.sigmoid_binary_cross_entropy(predictions*6, labels))

def accuracy(labels, predictions):
    return jnp.sum(predictions == labels)/len(labels)

def get_mini_batch(X,Y,n):
    """Return a random mini-batch of size n from data."""
    indices = np.random.choice(X.shape[0], size=n, replace=False)
    return X[indices, :], Y[indices]

def run_adam(grad_fn, cost_fn, lr, init_params, model, num_iter=5):
    """
    Optimises a model using the adam gradient update. We use optax.
    :param grad_func: vmapped function that returns the grads of a batch
    :param cost_fn: function that returns the cost of a batch
    :param lr: initial learning rate
    :param init_params: initial parameters
    :param model: the model function used for evaluation
    :param num_iter: the number of training steps
    :return:
    params: trained parameters
    history: training history
    """
    params = init_params.copy()
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    history = []
    # chunk the functions to save memory
    # chunk size should divide batch_size
    chunked_model = chunk_vmapped_fn(model, 1, 1)
    chunked_grad = chunk_grad(grad_fn, 1)
    chunked_loss = chunk_loss(cost_fn, 1)
    epsilon = 0.01

    for it in range(num_iter):
        X_batch, Y_batch = get_mini_batch(X, Y, batch_size)
        grads = chunked_grad(params, X_batch, Y_batch)
        grad_noise = jax.random.normal(key=jax.random.PRNGKey(np.random.randint(1000000)),
                                       shape=grads['w'].shape) * epsilon
        grads['w'] = grads['w'] + grad_noise
        cst = chunked_loss(params, X_batch, Y_batch)
        predictions = jnp.sign(chunked_model(params['w'], Xtest))
        acc = accuracy(Ytest, predictions)
        history.append((params, cst, acc))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if it % 1 == 0:
            print([cst, acc])
    return params, history

#################################### TRAINING ####################################

np.random.seed(seed)
X, Y = generate_data(qubits, 1000, dim//2,noise=1.0)
Xtest, Ytest = generate_data(dim, 100, dim//2 ,noise=1.0)

scale = 0.5
X = scale*X
Xtest = scale*Xtest

batch_size = 20
num_iter = 100
lr=0.01
trials = 20

plots_QCNN = []
plots_parallel = []
plots_general = []
plots_separable = []

for t in range(trials):
    print('trial=' + str(t))
    print('sep')
    init_params = {'w': 2 * np.pi * np.random.rand(qubits * 3)}
    params, history_separable = run_adam(grad_separable, cost_separable, lr, init_params, separable_model, num_iter=num_iter)
    plots_separable.append(history_separable)
    print('QCNN')
    init_params = {'w': 2 * np.pi * np.random.rand(n_layers_qcnn*n_params_layer)}
    params, history_QCNN = run_adam(grad_QCNN, cost_QCNN, lr, init_params, QCNN_eval, num_iter=num_iter)
    plots_QCNN.append(history_QCNN)
    print('commuting')
    init_params = {'w': 2 * np.pi * np.random.rand(num_param_parallel)}
    params, history_parallel = run_adam(grad_parallel, cost_parallel, lr, init_params, parallel_model_eval, num_iter=num_iter)
    plots_parallel.append(history_parallel)
    print('noncommuting')
    init_params = {'w': 2 * np.pi * np.random.rand(num_param_general)}
    params, history_general = run_adam(grad_general, cost_general, lr, init_params, general_model_eval, num_iter=num_iter)
    plots_general.append(history_general)


np.savetxt(f'compare_cost_separable_{dim}.txt',[[plots_separable[t][i][1] for i in range(num_iter)] for t in range(trials)])
np.savetxt(f'compare_cost_parallel_{dim}.txt',[[plots_parallel[t][i][1] for i in range(num_iter)] for t in range(trials)])
np.savetxt(f'compare_cost_general_{dim}.txt',[[plots_general[t][i][1] for i in range(num_iter)]for t in range(trials)])
np.savetxt(f'compare_cost_qcnn_{dim}.txt',[[plots_QCNN[t][i][1] for i in range(num_iter)]for t in range(trials)])

np.savetxt(f'compare_acc_separable_{dim}.txt',[[plots_separable[t][i][2] for i in range(num_iter)] for t in range(trials)])
np.savetxt(f'compare_acc_parallel_{dim}.txt',[[plots_parallel[t][i][2] for i in range(num_iter)] for t in range(trials)])
np.savetxt(f'compare_acc_general_{dim}.txt',[[plots_general[t][i][2] for i in range(num_iter)] for t in range(trials)])
np.savetxt(f'compare_acc_qcnn_{dim}.txt',[[plots_QCNN[t][i][2] for i in range(num_iter)] for t in range(trials)])

