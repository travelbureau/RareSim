import tensorflow as tf
import ops

class Policy(object):
    def __init__(self, in_dim, out_dim, size, do_keep_prob, use_separate_ego):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0],
            'n_hidden_1': size[1],
            'do_keep_prob': do_keep_prob
        }

        self.use_separate_ego = use_separate_ego

    def forward(self, state, reuse=False):
        if self.use_separate_ego:
            with tf.variable_scope('policy'):
                h0 = ops.dense(state, self.arch_params['in_dim'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0', reuse)
                h1 = ops.dense(h0, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1', reuse)
                relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
                a = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2', reuse)
                logstd = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2b', reuse)

            with tf.variable_scope('ego_policy'):
                relu1_do_ego = tf.expand_dims(relu1_do[0], axis=0)
                a_ego = ops.dense(relu1_do_ego, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2', reuse)
                logstd_ego = ops.dense(relu1_do_ego, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2b', reuse)
            a = tf.concat([a_ego, a[1:,:]], axis=0)
            logstd = tf.concat([logstd_ego, logstd[1:,:]], axis=0)

        else:
            with tf.variable_scope('policy'):
                h0 = ops.dense(state, self.arch_params['in_dim'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0', reuse)
                h1 = ops.dense(h0, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1', reuse)
                relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
                a = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2', reuse)
                logstd = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2b', reuse)

        return a, logstd