import ops
import tensorflow as tf


class ForwardModel(object):
    def __init__(self, state_size, action_size, encoding_size):
        self.state_size = state_size
        self.action_size = action_size
        self.encoding_size = encoding_size

    def forward(self, input, reuse=False):
        with tf.variable_scope('forward_model'):
            state = tf.cast(input[0], tf.float32)
            action = tf.cast(input[1], tf.float32)
            gru_state = tf.cast(input[2], tf.float32)

            # State embedding
            state_embedder1 = ops.dense(state, self.state_size, self.encoding_size, tf.nn.relu, "encoder1_state", reuse)
            gru_state = ops.gru(state_embedder1, gru_state, self.encoding_size, self.encoding_size, 'gru1', reuse)
            state_embedder2 = ops.dense(gru_state, self.encoding_size, self.encoding_size, tf.sigmoid, "encoder2_state", reuse)

            # Action embedding
            action_embedder1 = ops.dense(action, self.action_size, self.encoding_size, tf.nn.relu, "encoder1_action", reuse)
            action_embedder2 = ops.dense(action_embedder1, self.encoding_size, self.encoding_size, tf.sigmoid, "encoder2_action", reuse)

            # Joint embedding
            joint_embedding = tf.multiply(state_embedder2, action_embedder2)

            # Next state prediction
            hidden1 = ops.dense(joint_embedding, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder3", reuse)
            hidden2 = ops.dense(hidden1, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder4", reuse)
            hidden3 = ops.dense(hidden2, self.encoding_size, self.encoding_size, tf.nn.relu, "decoder1", reuse)
            next_state = ops.dense(hidden3, self.encoding_size, self.state_size, None, "decoder2", reuse)

            return next_state, gru_state
