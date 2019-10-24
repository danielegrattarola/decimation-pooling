import time

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from spektral.layers import GlobalAvgPool, GraphConvSkip
from spektral.layers.ops import sp_matrix_to_sp_tensor_value
from spektral.utils import batch_iterator, log, init_logging

from utils.dataset_loader import get_graph_kernel_dataset
from utils.misc import preprocess, create_batch


def evaluate(A_list, X_list, D_list, y_list, ops):
    batches_ = batch_iterator([A_list, X_list, D_list, y_list], batch_size=P['batch_size'])
    output_ = []
    for A__, X__, D__, y__ in batches_:
        A__, X__, D__, I__ = create_batch(A__, X__, D__)
        feed_dict_ = {X_in: X__,
                      I_in: I__,
                      target: y__,
                      D_in[0]: sp_matrix_to_sp_tensor_value(D__[0]),
                      D_in[1]: sp_matrix_to_sp_tensor_value(D__[1]),
                      A_in[0]: sp_matrix_to_sp_tensor_value(A__[0]),
                      A_in[1]: sp_matrix_to_sp_tensor_value(A__[1]),
                      A_in[2]: sp_matrix_to_sp_tensor_value(A__[2]),
                      'dense_1_sample_weights:0': np.ones((1,))}
        outs_ = sess.run(ops, feed_dict=feed_dict_)
        output_.append(outs_)
    return np.mean(output_, 0)


seed = 0
np.random.seed(seed)

# Parameters
P = dict(
    coarsening_levels=[0, 1, 2],  # Levels of pooling for decimation w/ Kron
    runs=1,                       # Number of runs to repeat
    data_mode='grakel',           # grakel or synth
    pool='decim',                 # Pooling strategy
    dataset_ID='PROTEINS',        # Name of folder in data/classification/
    n_channels=32,                # Channels in GNN layers
    activ='relu',                 # Activation in GNN layers
    GNN_l2=1e-4,                  # L2 regularisation in GNN layers
    epochs=500,                   # Number of training epochs
    es_patience=50,               # Patience for early stopping
    learning_rate=5e-4,           # Learning rate
    batch_size=8                  # Size of the mini-batches
)
log_dir = init_logging()          # Create log directory and file
log(P)

################################################################################
# Load data
################################################################################
if P['data_mode'] == 'grakel':
    # Load one of the Benchmark Data Sets for Graph Kernels by C. Morris
    # https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
    A, X, y = get_graph_kernel_dataset(P['dataset_ID'], feat_norm='ohe')
    # Train/test split
    A_train, A_test, \
    X_train, X_test, \
    y_train, y_test = train_test_split(A, X, y, test_size=0.1, stratify=y)
    A_train, A_val, \
    X_train, X_val, \
    y_train, y_val = train_test_split(A_train, X_train, y_train, test_size=0.1, stratify=y_train)
elif P['data_mode'] == 'synth':
    # Load one of the Benchmark Datasets for Graph Classification by F. M. Bianchi
    # https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification
    loaded = np.load('data/hard.npz', allow_pickle=True)
    X_train, A_train, y_train = loaded['tr_feat'], list(loaded['tr_adj']), loaded['tr_class']
    X_test, A_test, y_test = loaded['te_feat'], list(loaded['te_adj']), loaded['te_class']
    X_val, A_val, y_val = loaded['val_feat'], list(loaded['val_adj']), loaded['val_class']
else:
    raise ValueError('Possible data_mode: grakel, synth')

# Parameters
F = X_train[0].shape[-1]  # Dimension of node features
n_out = y_train[0].shape[-1]  # Dimension of the target

################################################################################
# Pre-compute pooling matrices and Laplacians
################################################################################
log('Creating Laplacians pyramids and decimation matrices.')
A_train, X_train, D_train = preprocess(
    A_train, X_train, coarsening_levels=P['coarsening_levels'], pool=P['pool']
)
A_val, X_val, D_val = preprocess(
    A_val, X_val, coarsening_levels=P['coarsening_levels'], pool=P['pool']
)
A_test, X_test, D_test = preprocess(
    A_test, X_test, coarsening_levels=P['coarsening_levels'], pool=P['pool']
)

################################################################################
# Build model
################################################################################
X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, F)))
A_in = [
    Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), sparse=True)
    for _ in P['coarsening_levels']
]
D_in = [
    Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), sparse=True)
    for _ in P['coarsening_levels'][:-1]
]
I_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,)))
target = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_out)))


def decimation_pooling_fn(x_):
    X_, D_, I_ = x_
    X_pooled = K.dot(D_, X_)
    I_pooled = K.cast(
        K.dot(D_,
              K.cast(I_, tf.float32)[..., None])[..., 0],
        tf.int32
    )
    return [X_pooled, I_pooled]
decimation_pooling_op = Lambda(decimation_pooling_fn)

# Block 1
X_1 = GraphConvSkip(P['n_channels'],
                    activation=P['activ'],
                    kernel_regularizer=l2(P['GNN_l2']))([X_in, A_in[0]])
X_1, I_1 = decimation_pooling_op([X_1, D_in[0], I_in])

# Block 2
X_2 = GraphConvSkip(P['n_channels'],
                    activation=P['activ'],
                    kernel_regularizer=l2(P['GNN_l2']))([X_1, A_in[1]])
X_2, I_2 = decimation_pooling_op([X_2, D_in[1], I_1])

# Block 3
X_3 = GraphConvSkip(P['n_channels'],
                    activation=P['activ'],
                    kernel_regularizer=l2(P['GNN_l2']))([X_2, A_in[2]])

# Output block
avgpool = GlobalAvgPool()([X_3, I_2])
output = Dense(n_out, activation='softmax')(avgpool)

# Build model
model = Model([X_in, I_in] + A_in + D_in, output)
model.compile('adam', 'categorical_crossentropy', target_tensors=[target])
model.summary()

# Training setup
sess = K.get_session()
loss = model.total_loss
acc = K.mean(categorical_accuracy(target, model.output))
opt = tf.train.AdamOptimizer(learning_rate=P['learning_rate'])
train_step = opt.minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

################################################################################
# Training loop
################################################################################
log('Fitting model')
model_loss = 0               # Keep track of the current loss
model_acc = 0                # Keep track of the current accuracy
best_val_loss = np.inf       # Keep track of best validation loss for ES
patience = P['es_patience']  # Keep track of current patience for ES
epoch_time = [0]             # Keep track of epochs duration

batches_in_epoch = np.ceil(y_train.shape[0] / P['batch_size'])
total_batches = batches_in_epoch * P['epochs']
batches = batch_iterator(
    [A_train, X_train, D_train, y_train],
    batch_size=P['batch_size'], epochs=P['epochs'], shuffle=True
)
for current_batch, (A_, X_, D_, y_) in enumerate(batches, start=1):
    A_, X_, D_, I_ = create_batch(A_, X_, D_)
    tr_feed_dict = {
        X_in: X_,
        I_in: I_,
        target: y_,
        D_in[0]: sp_matrix_to_sp_tensor_value(D_[0]),
        D_in[1]: sp_matrix_to_sp_tensor_value(D_[1]),
        A_in[0]: sp_matrix_to_sp_tensor_value(A_[0]),
        A_in[1]: sp_matrix_to_sp_tensor_value(A_[1]),
        A_in[2]: sp_matrix_to_sp_tensor_value(A_[2]),
        'dense_1_sample_weights:0': np.ones((1, ))
    }
    epoch_time[-1] -= time.time()
    outs = sess.run([train_step, loss, acc], feed_dict=tr_feed_dict)
    epoch_time[-1] += time.time()

    model_loss += outs[1]
    model_acc += outs[2]
    if current_batch % batches_in_epoch == 0:
        model_loss /= batches_in_epoch
        model_acc /= batches_in_epoch

        val_loss, val_acc = evaluate(A_val, X_val, D_val, y_val, [loss, acc])
        log('Epoch: {} - Loss: {:.3f} - Acc: {:.3f} - Val. loss: {:.3f} '
            '- Val. acc: {:.3f}'
            .format(int(current_batch // batches_in_epoch), model_loss,
                    model_acc, val_loss, val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = P['es_patience']
            log('New best val_loss {:.3f}'.format(val_loss))
            model.save_weights(log_dir + 'best_model.h5')
        else:
            patience -= 1
            if patience == 0:
                log('Early stopping (best val_loss: {})'.format(best_val_loss))
                break
        epoch_time.append(0)
        model_loss = 0
        model_acc = 0
# Load best model
model.load_weights(log_dir + 'best_model.h5')

################################################################################
# Evaluation
################################################################################
log('Testing model')
test_loss, test_acc = evaluate(A_test, X_test, D_test, y_test, [loss, acc])
log('Test loss: {:.5f}; Test accuracy: {:.3f}'.format(test_loss, test_acc))
log('Average epoch duration: {:.3f} +- {:.3f}'.format(np.mean(epoch_time[1:]), np.std(epoch_time[1:])))