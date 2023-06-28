import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from pennylane import pennylane as qml
import numpy as np
import optax
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, permutations, product, combinations

"""
This the numerics to accompany the article 'backpropagation scaling in parameterised quantum circuits'
The following code can be used to reproduce plots of the form of Figure 6 of the paper. 
"""

# ################################### DATA GENERATION ####################################


def generate_data(dim, n, length, noise=0.0):
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


dim = 16  # problem dimension
qubits = dim
seed = 852459
np.random.seed(seed)

# ################################### FUNCTIONS USED IN MODEL GENERATION  ####################################


def cyclic_perm(a):
    "gets all cyclic permutations of a list"
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b


def seed_gens(weight, qubits=qubits, ops=["I", "X"]):
    """
    get all the seed generators up to a given pauli weight
    the seeds are fed into get_gens to get the symmetric generators
    """
    ops
    seeds = []
    for prod in product(ops, repeat=weight):
        seeds.append(list(prod) + ["I"] * (qubits - weight))
    return seeds[1:]


def seed_gens_doubles(ops=["I", "X"]):
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
        genlist = ["".join(all_gens[i]) for i in range(qubits)]
        genlist = list(dict.fromkeys(genlist))
        genlist.sort()
        if genlist not in gens:
            gens.append(genlist)
    return gens


def square_loss(labels, predictions):
    """Square loss used to define gradient functions in models"""
    loss = jnp.sum((labels - predictions) ** 2)
    loss = loss / len(labels)
    return loss


# ################################### MODEL DEFINITIONS ####################################


# ####### COMMUTING MODEL ##########

obs = qml.dot(
    [1] * qubits, [qml.PauliZ(i) for i in range(qubits)]
)  # equivariant observable \mathcal{H}

seeds = seed_gens(qubits)
gens = get_gens(seeds)

# convert the gens to Pauli words and wires for more efficient use in pennylane
words_and_wires = [
    [(gen.replace("I", ""), [i for i, l in enumerate(gen) if l == "X"]) for gen in gen_list]
    for gen_list in gens
]

# take only the generators with weight <=3
waw = []
for elem in words_and_wires:
    if len(elem[0][0]) <= 3:
        waw.append(elem)
words_and_wires = waw

num_gen_commuting = sum(
    len(sublist) for sublist in words_and_wires
)  # total generators used in model
num_param_commuting = len(words_and_wires)  # total paramters used in model
num_circuits_commuting = qubits  # total number of circuits needed for gradient evalution

print("number of generators for commuting model: " + str(num_gen_commuting))
print("number of circuits for commuting model: " + str(num_circuits_commuting))
print("number of parameters for commuting model: " + str(num_param_commuting))


dev = qml.device("default.qubit", wires=qubits)


@qml.qnode(dev, interface="jax")
def commuting_model_eval(params, x):
    """
    Model used for evaluation but not for training. Sometimes it is useful to separate the two for efficiency
    reasons
    :param params: trainable parameters
    :param x: data input
    :return: expval corresponding to class label
    """

    # data encoding
    for q in range(qubits):
        qml.RY(x[q], wires=q)
    # apply the rotation for each equivariant generator
    for i, sublist in enumerate(words_and_wires):
        for word, wires in sublist:
            qml.PauliRot(params[i], pauli_word=word, wires=wires)
    return qml.expval(obs)


commuting_model_eval = jax.vmap(commuting_model_eval, (None, 0))  # for parallelisation over batches
commuting_model_eval = jax.jit(commuting_model_eval)


@qml.qnode(dev, interface="jax")
def commuting_model(params, x):
    """
    Model used to compute gradients. This is the same as above.
    """
    for q in range(qubits):
        qml.RY(x[q], wires=q)
    # apply the rotation for each equivariant generator
    for i, sublist in enumerate(words_and_wires):
        for word, wires in sublist:
            qml.PauliRot(params[i], pauli_word=word, wires=wires)
    return qml.expval(obs)


commuting_model = jax.vmap(commuting_model, (None, 0))


def cost_commuting(params, input_data, labels):
    predictions = commuting_model(params["w"], input_data)
    return square_loss(predictions, labels)


func_commuting = jax.value_and_grad(
    cost_commuting
)  # function used to get cost and gradient for training
func_commuting = jax.jit(func_commuting)


# ####### NONCOMMUTING MODEL ##########

layers = 4
obs = qml.dot([1] * qubits, [qml.PauliZ(i) for i in range(qubits)])

# compute seeds
localz = get_gens(seed_gens(1, ops=["I", "Z"]))
localy = get_gens(seed_gens(1, ops=["I", "Y"]))
doublex = get_gens(seed_gens_doubles(ops=["I", "X"]))

# get words and wires of equivariant generators
z_words_and_wires = [
    [(gen.replace("I", ""), [i for i, l in enumerate(gen) if l == "Z"]) for gen in gen_list]
    for gen_list in localz
]
y_words_and_wires = [
    [(gen.replace("I", ""), [i for i, l in enumerate(gen) if l == "Y"]) for gen in gen_list]
    for gen_list in localy
]
x_words_and_wires = [
    [(gen.replace("I", ""), [i for i, l in enumerate(gen) if l == "X"]) for gen in gen_list]
    for gen_list in doublex
]

noncommuting_words_and_wires = z_words_and_wires + y_words_and_wires + x_words_and_wires

num_gens_per_layer = sum(len(sublist) for sublist in noncommuting_words_and_wires)
num_gens_noncommuting = num_gens_per_layer * layers
theory_num_gens_noncommuting = layers * qubits / 2 * (qubits + 3)
assert num_gens_noncommuting == theory_num_gens_noncommuting
num_param_per_layer = len(noncommuting_words_and_wires)
num_param_noncommuting = num_param_per_layer * layers
theory_num_param_noncommuting = layers * (2 + qubits / 2)
assert num_param_noncommuting == theory_num_param_noncommuting
num_circuits_noncommuting = num_gens_noncommuting * 2  # from parameter shift rule

print("number of generators for noncommuting model: " + str(num_gens_noncommuting))
print("number of circuits for noncommuting model: " + str(num_circuits_noncommuting))
print("number of parameters for noncommuting model: " + str(num_param_noncommuting))


dev = qml.device("default.qubit", wires=qubits)


@qml.qnode(dev, interface="jax")
def noncommuting_model_eval(params, x):
    """
    used for evaluation but not for training
    """
    for q in range(qubits):
        qml.RY(x[q], wires=q)
    # apply the rotation for each equivariant generator
    for l in range(layers):
        for i, sublist in enumerate(noncommuting_words_and_wires):
            for word, wires in sublist:
                qml.PauliRot(params[l * num_param_per_layer + i], pauli_word=word, wires=wires)
    return qml.expval(obs)


noncommuting_model_eval = jax.vmap(noncommuting_model_eval, (None, 0))
noncommuting_model_eval = jax.jit(noncommuting_model_eval)


@qml.qnode(dev, interface="jax")
def noncommuting_model(params, x):
    for q in range(qubits):
        qml.RY(x[q], wires=q)
    # apply the rotation for each equivariant generator
    for l in range(layers):
        for i, sublist in enumerate(noncommuting_words_and_wires):
            for word, wires in sublist:
                qml.PauliRot(params[l * num_param_per_layer + i], pauli_word=word, wires=wires)
    return qml.expval(obs)


noncommuting_model = jax.vmap(noncommuting_model, (None, 0))


def cost_noncommuting(params, input_data, labels):
    predictions = noncommuting_model(params["w"], input_data)
    return square_loss(predictions, labels)


func_noncommuting = jax.value_and_grad(cost_noncommuting)
func_noncommuting = jax.jit(func_noncommuting)


# ####### QUANTUM CONVOLUTIONAL MODEL ##########


def QCNN_block(params, wires):
    """
    Parameterised 'convolutional' circuit. Circuit 7 from https://arxiv.org/pdf/2108.00661.pdf
    :param params: trainiable parameters
    :param wires: qires on which the block acts
    """
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
    """Trainable pooling operation from https://arxiv.org/pdf/2108.00661.pdf"""
    qml.CRZ(params[0], wires=wires)
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=wires)
    qml.PauliX(wires=wires[0])


n_params_block = 10  # number of parameters in the convolutional circuit
n_params_layer = 12  # total trainiable parameters per layer
n_layers_qcnn = int(np.log2(qubits))

dev = qml.device("default.qubit", wires=qubits)


@qml.qnode(dev, interface="jax")
def QCNN_eval(params, x):
    """
    Quantum convolutoinal neural network. Used for evaluation but not training.
    """
    wires = range(qubits)
    # data encoding
    for q in range(qubits):
        qml.RY(x[q], wires=q)

    for j in range(n_layers_qcnn):
        # first layer of convolutional circuits
        for i in range(0, qubits // (2**j), 2):
            QCNN_block(
                params[j, :n_params_block],
                wires=[
                    wires[(2**j - 1) + 2**j * i],
                    wires[((2**j - 1) + 2 ** (j) * i + (2**j)) % qubits],
                ],
            )
        # second layer of convolutional circuits
        if j != int(np.log2(qubits)) - 1:
            for i in range(1, qubits // (2**j), 2):
                QCNN_block(
                    params[j, :n_params_block],
                    wires=[
                        wires[(2**j - 1) + 2**j * i],
                        wires[((2**j - 1) + 2 ** (j) * i + (2**j)) % qubits],
                    ],
                )
        # pooling layer
        for i in range(0, qubits // (2**j), 2):
            pooling(
                params[j, -2:],
                wires=[
                    wires[(2**j - 1) + 2**j * i],
                    wires[((2**j - 1) + 2 ** (j) * i + (2**j)) % qubits],
                ],
            )

    return qml.expval(qml.PauliZ(qubits - 1))


QCNN_eval = jax.vmap(QCNN_eval, (None, 0))
QCNN_eval = jax.jit(QCNN_eval)


@qml.qnode(dev, interface="jax")
def QCNN(params, x):
    wires = range(qubits)
    for q in range(qubits):
        qml.RY(x[q], wires=q)

    for j in range(n_layers_qcnn):
        for i in range(0, qubits // (2**j), 2):
            QCNN_block(
                params[j],
                wires=[
                    wires[(2**j - 1) + 2**j * i],
                    wires[((2**j - 1) + 2 ** (j) * i + (2**j)) % qubits],
                ],
            )
        if j != int(np.log2(qubits)) - 1:
            for i in range(1, qubits // (2**j), 2):
                QCNN_block(
                    params[j],
                    wires=[
                        wires[(2**j - 1) + 2**j * i],
                        wires[((2**j - 1) + 2 ** (j) * i + (2**j)) % qubits],
                    ],
                )

        for i in range(0, qubits // (2**j), 2):
            pooling(
                params[j, -2:],
                wires=[
                    wires[(2**j - 1) + 2**j * i],
                    wires[((2**j - 1) + 2 ** (j) * i + (2**j)) % qubits],
                ],
            )

    return qml.expval(qml.PauliZ(qubits - 1))


QCNN = jax.vmap(QCNN, (None, 0))


def cost_QCNN(params, input_data, labels):
    predictions = QCNN(params["w"], input_data)
    return square_loss(predictions, labels)


func_QCNN = jax.value_and_grad(cost_QCNN)
func_QCNN = jax.jit(func_QCNN)

num_param_QCNN = n_layers_qcnn * n_params_layer
num_gen_QCNN = (2 * qubits - 3) * n_params_block + (qubits - 1) * 2
num_circuits_QCNN = (2 * qubits - 3) * (8 * 2 + 2 * 4) + (qubits - 1) * 2 * 4

print("number of generators for QCNN model: " + str(num_gen_QCNN))
print("number of circuits for QCNN model: " + str(num_circuits_QCNN))
print("number of parameters for QCNN model: " + str(num_param_QCNN))

# ####### SEPARABLE MODEL ##########

# +
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev, interface="jax")
def _separable_model(params, x):
    qml.RY(x, wires=0)
    # arbitrary single qubit rotation
    qml.RZ(params[0], wires=0)
    qml.RX(params[1], wires=0)
    qml.RY(params[2], wires=0)
    return qml.expval(qml.PauliZ(0))


def separable_model(params, x):
    return jnp.sum(
        jnp.array([_separable_model(params[3 * q : 3 * q + 3], x[q]) for q in range(qubits)])
    )


# -

separable_model = jax.vmap(separable_model, (None, 0))
separable_model_jit = jax.jit(separable_model)


def cost_separable(params, input_data, labels):
    predictions = separable_model(params["w"], input_data)
    return square_loss(predictions, labels)


func_separable = jax.value_and_grad(cost_separable)
func_separable = jax.jit(func_separable)


# ################################### TRAINING AND EVAL FUNCTIONS ####################################


def accuracy(labels, predictions):
    """class prediction accuracy"""
    return jnp.sum(predictions == labels) / len(labels)


def get_mini_batch(X, Y, n):
    """Return a random mini-batch of size n from data."""
    indices = np.random.choice(X.shape[0], size=n, replace=False)
    return X[indices, :], Y[indices]


def run_adam(func, lr, init_params, model_eval, num_iter=5):
    """
    Optimises a model using the adam gradient update. We use optax.
    :param func: vmapped function that returns the costs and grads of a batch
    :param lr: initial learning rate
    :param init_params: initial parameters
    :param model_eval: the model function used for evaluation
    :param num_iter: the number of training steps
    :return:
    params: trained parameters
    history: training history
    """
    params = init_params.copy()
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    history = []
    for it in range(num_iter):
        X_batch, Y_batch = get_mini_batch(X, Y, batch_size)
        cst, grads = func(params, X_batch, Y_batch)
        predictions = jnp.sign(model_eval(params["w"], Xtest))
        acc = accuracy(Ytest, predictions)
        history.append((params, cst, acc))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if it % 1 == 0:
            print([cst, acc])
    return params, history


# ################################### TRAINING ####################################

np.random.seed(seed)
X, Y = generate_data(qubits, 1000, dim // 2, noise=1.0)  # train dataset
Xtest, Ytest = generate_data(dim, 100, dim // 2, noise=1.0)  # test dataset

# rescale data
scale = 0.5
X = scale * X
Xtest = scale * Xtest

batch_size = 20
num_iter = 50  # training steps
lr = 0.01
trials = 2

plots_QCNN = []
plots_commuting = []
plots_noncommuting = []
plots_separable = []

for t in range(trials):
    print("trial=" + str(t))
    print("model = sep")
    init_params = {"w": 2 * np.pi * np.random.rand(qubits * 3)}
    params, history_separable = run_adam(
        func_separable, lr, init_params, separable_model_jit, num_iter=num_iter
    )
    plots_separable.append(history_separable)

    print("model = QCNN")
    init_params = {"w": 2 * np.pi * np.random.rand(n_layers_qcnn, n_params_layer)}
    params, history_QCNN = run_adam(func_QCNN, lr, init_params, QCNN_eval, num_iter=num_iter)
    plots_QCNN.append(history_QCNN)

    print("model = commuting")
    init_params = {"w": 2 * np.pi * np.random.rand(num_param_commuting)}
    params, history_commuting = run_adam(
        func_commuting, lr, init_params, commuting_model_eval, num_iter=num_iter
    )
    plots_commuting.append(history_commuting)

    print("model = noncommuting")
    init_params = {"w": 2 * np.pi * np.random.rand(num_param_noncommuting)}
    params, history_noncommuting = run_adam(
        func_noncommuting, lr, init_params, noncommuting_model_eval, num_iter=num_iter
    )
    plots_noncommuting.append(history_noncommuting)

np.savetxt(
    "cost_separable.txt",
    [[plots_separable[t][i][1] for i in range(num_iter)] for t in range(trials)],
)
np.savetxt(
    "cost_commuting.txt",
    [[plots_commuting[t][i][1] for i in range(num_iter)] for t in range(trials)],
)
np.savetxt(
    "cost_noncommuting.txt",
    [[plots_noncommuting[t][i][1] for i in range(num_iter)] for t in range(trials)],
)
np.savetxt("cost_qcnn.txt", [[plots_QCNN[t][i][1] for i in range(num_iter)] for t in range(trials)])

np.savetxt(
    "acc_separable.txt",
    [[plots_separable[t][i][2] for i in range(num_iter)] for t in range(trials)],
)
np.savetxt(
    "acc_commuting.txt",
    [[plots_commuting[t][i][2] for i in range(num_iter)] for t in range(trials)],
)
np.savetxt(
    "acc_noncommuting.txt",
    [[plots_noncommuting[t][i][2] for i in range(num_iter)] for t in range(trials)],
)
np.savetxt("acc_qcnn.txt", [[plots_QCNN[t][i][2] for i in range(num_iter)] for t in range(trials)])

costs_separable = np.loadtxt("cost_separable.txt")
costs_commuting = np.loadtxt("cost_commuting.txt")
costs_noncommuting = np.loadtxt("cost_noncommuting.txt")
costs_QCNN = np.loadtxt("cost_qcnn.txt")

accs_separable = np.loadtxt("acc_separable.txt")
accs_commuting = np.loadtxt("acc_commuting.txt")
accs_noncommuting = np.loadtxt("acc_noncommuting.txt")
accs_QCNN = np.loadtxt("acc_qcnn.txt")


# ################################### PLOTS ####################################

# shots per iteration for each model (see paper for more details)
spi_noncommuting = num_circuits_noncommuting * batch_size * 1000
spi_commuting = num_circuits_commuting * batch_size * 1000
spi_QCNN = num_circuits_QCNN * batch_size * 1000
spi_separable = 3 * batch_size * 1000

plt.figure(figsize=(8, 4), tight_layout=True)
alpha = 0.7
linewidth = 1.2

font = {"family": "serif", "weight": "normal", "size": 11}
plt.rc("font", **font)

# subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))
fig.tight_layout(pad=5.0)

########
ax1.grid(True, which="both")
ax1.set_ylabel("cost", fontsize=12)
ax1.set_xlabel("shots", fontsize=12)
ax1.set_xscale("log")
########
ax2.grid(True, which="both")
ax2.set_ylabel("test accuracy", fontsize=12)
ax2.set_xlabel("shots", fontsize=12)
ax2.set_xscale("log")

# ax2.set_ylim(0.3,1.03)
ax1.set_facecolor((0.95, 0.95, 0.95))
ax2.set_facecolor((0.95, 0.95, 0.95))

c1 = (152 / 256, 73 / 256, 66 / 256)
c2 = (71 / 256, 150 / 256, 157 / 256)
c3 = "black"

steps = np.arange(num_iter)

# plot first trial to define labels
ax1.plot(
    steps * spi_noncommuting,
    [costs_noncommuting[0][i] for i in range(num_iter)],
    alpha=alpha,
    color=c2,
    label="noncommuting circuit",
)
ax1.plot(
    steps * spi_commuting,
    [costs_commuting[0][i] for i in range(num_iter)],
    alpha=0.5,
    color=c3,
    label="commuting circuit",
)
ax1.plot(
    steps * spi_QCNN,
    [costs_QCNN[0][i] for i in range(num_iter)],
    alpha=alpha,
    color=c1,
    label="quantum convolutional",
)
# ax1.plot(steps*spi_separable,[costs_separable[0][i] for i in range(num_iter)],alpha=alpha,color=c2, label='separable model')

# plot the remaining trials
for t in range(1, trials):
    ax1.plot(
        steps * spi_noncommuting,
        [costs_noncommuting[t][i] for i in range(num_iter)],
        alpha=alpha,
        color=c2,
    )
    ax1.plot(
        steps * spi_commuting, [costs_commuting[t][i] for i in range(num_iter)], alpha=0.5, color=c3
    )
    ax1.plot(steps * spi_QCNN, [costs_QCNN[t][i] for i in range(num_iter)], alpha=alpha, color=c1)
#     ax1.plot(steps * spi_separable,[costs_separable[t][i] for i in range(num_iter)],alpha=alpha,color=c2)

for t in range(trials):
    ax2.plot(
        steps * spi_noncommuting,
        [accs_noncommuting[t][i] for i in range(num_iter)],
        alpha=alpha,
        color=c2,
    )
    ax2.plot(
        steps * spi_commuting, [accs_commuting[t][i] for i in range(num_iter)], alpha=0.5, color=c3
    )
    ax2.plot(steps * spi_QCNN, [accs_QCNN[t][i] for i in range(num_iter)], alpha=alpha, color=c1)
#     ax2.plot(steps * spi_separable,[accs_separable[t][i] for i in range(num_iter)],alpha=alpha,color=c2)

ax1.legend()

plt.savefig("plot.pdf", dpi=300)


# Plot for the separable model
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
fig.tight_layout(pad=5.0)
########
ax1.grid(True, which="both")
ax1.set_ylabel("cost", fontsize=12)
ax1.set_xlabel("shots", fontsize=12)
ax1.set_xscale("log")
########
ax2.grid(True, which="both")
ax2.set_ylabel("test accuracy", fontsize=12)
ax2.set_xlabel("shots", fontsize=12)
ax2.set_xscale("log")
# ax2.set_ylim(0.3,1.03)
ax1.set_facecolor((0.95, 0.95, 0.95))
ax2.set_facecolor((0.95, 0.95, 0.95))
c = (71 / 256, 150 / 256, 157 / 256)
steps = np.arange(num_iter)
for t in range(1, trials):
    ax1.plot(
        steps * spi_separable,
        [costs_separable[t][i] for i in range(num_iter)],
        alpha=alpha,
        color=c,
    )
    ax2.plot(
        steps * spi_separable, [accs_separable[t][i] for i in range(num_iter)], alpha=alpha, color=c
    )
fig.savefig("plot_separable.pdf", dpi=300)
