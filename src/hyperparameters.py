import tensorflow as tf
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer, l1_l2_regularizer


REGULARISATORS = {'none': lambda arg: tf.constant(0.0),
                  'l1': l1_regularizer(1.0),
                  'l2' : l2_regularizer(1.0),
                  'l1_l2': l1_l2_regularizer(1.0, 1.0)}

NBS_EPOCHS = [5, 10, 20, 50, 100, 200]

BATCH_SIZES = [1, 16, 32, 64, 128, 512]

ARCHITECTURES = [[128] * 0,
                 [128] * 1,]
                 #[128] * 2,
                 #[128] * 3,
                 #[128] * 4]

OPTIMISERS = [tf.train.AdamOptimizer(learning_rate=1e-3),
              tf.train.GradientDescentOptimizer(learning_rate=1e-3),
              tf.train.AdadeltaOptimizer(learning_rate=1e-3),
              tf.train.RMSPropOptimizer(learning_rate=1e-3)]
