import tensorflow as tf
import numpy as np


def dense_layer_fn(layer_number, size_prev, size_next, activation_fn, layer_input):
    W = tf.Variable(tf.random_normal([size_prev, size_next]),
                    name=f'W_{layer_number}')
    b = tf.Variable(tf.random_normal([size_next]),
                    name=f'b_{layer_number}')
    logits = tf.add(tf.matmul(layer_input, W), b)
    activation = activation_fn(logits)
    return activation, logits, W, b


def mlp_model(features, labels, mode, params):
    regulariser_fn = params['regulariser_fn']
    #cost_fn = params['cost_fn']
    activation_fn = params['activation_fn']
    beta = params['regularisation_rate']

    prev_sizes = [params['nb_inputs']] + params['hidden_layers_sizes']
    next_sizes = params['hidden_layers_sizes'] + [params['nb_classes']]
    activation_fns = [activation_fn] * len(params['hidden_layers_sizes']) + [tf.nn.softmax]

    activation = tf.feature_column.input_layer(features, params['feature_columns'])
    regulariser = tf.constant(0.0)
    #logits = 0
    for i, (size_prev, size_next, activ_fn) in enumerate(zip(prev_sizes,
                                                             next_sizes,
                                                             activation_fns)):
        activation, logits, W, b = dense_layer_fn(i, size_prev, size_next,
                                                  activ_fn, activation)
        if (mode == tf.estimator.ModeKeys.PREDICT
                or mode == tf.estimator.ModeKeys.EVAL):
            regulariser += regulariser_fn(W) + regulariser_fn(b)

    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'class_ids': predicted_classes[:, tf.newaxis],
                       'probabilities': activation,
                       'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        regulariser = tf.multiply(beta, regulariser, name='val_regularisation_value')
        loss = tf.add(loss, regulariser, name='val_loss')
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    regulariser = tf.multiply(beta, regulariser, name='regularisation_value')
    loss = tf.add(loss, regulariser, name='loss')

    optimiser = params['optimiser']
    train_op = optimiser.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


#optimiser = tf.train.AdamOptimizer(learning_rate=1e-3)