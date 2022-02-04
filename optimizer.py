import tensorflow as tf
from utils import *

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

SMALL = 1e-16

class Optimizer(object):
    def __init__(self, labels, model, epoch, model_str):

        epoch = tf.cast(epoch, tf.float32)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        
        # S MC samples:
        self.nll = tf.constant(0.0)
        for s in range(model.S):
            preds = model.reconstructions_list[s]
            neg_ll = self.binary_weighted_crossentropy(preds, labels)
            self.nll += tf.reduce_mean(neg_ll)
        self.nll = self.nll / model.S 
        self.check = model.reconstructions_list

        # KL-divergence loss
        self.kl = 0
        for k in range(model.S):
            for idx in range(model.num_decoder_layers):
                
                mean_posterior1 = model.posterior_theta_param1_list[k][idx][0]
                log_std_posterior1 = model.posterior_theta_param1_list[k][idx][1]
                self.kl_z = kl_normal(mean_posterior1, log_std_posterior1) / model.batch_size
                mean_posterior2 = model.posterior_theta_param2_list[k][idx][0]
                log_std_posterior2 = model.posterior_theta_param2_list[k][idx][1]
                self.kl_z += kl_normal(mean_posterior2, log_std_posterior2) / model.batch_size
                
                if model_str == 'dlsm':
                    pi_logit_prior = model.prior_theta_param_list[k][idx][0]
                    pi_logit_posterior1 = model.posterior_theta_param1_list[k][idx][2]
                    s_logit1 = model.theta1_list[k][idx][1]
                    self.kl_s = kl_binconcrete(pi_logit_posterior1, pi_logit_prior, s_logit1, FLAGS.temp_post, FLAGS.temp_prior) / model.batch_size
                    pi_logit_posterior2 = model.posterior_theta_param2_list[k][idx][2]
                    s_logit2 = model.theta2_list[k][idx][1]
                    self.kl_s += kl_binconcrete(pi_logit_posterior2, pi_logit_prior, s_logit2, FLAGS.temp_post, FLAGS.temp_prior) / model.batch_size
                
                    alpha_gam_prior = model.prior_theta_param_list[k][idx][1]
                    alpha_gam_posterior = model.posterior_theta_param1_list[k][idx][3]
                    self.kl_alpha_gam = kl_gamma(alpha_gam_posterior, alpha_gam_prior) / model.batch_size
                    if FLAGS.directed == 1:
                        alpha_del_prior = model.prior_theta_param_list[k][idx][2]
                        alpha_del_posterior = model.posterior_theta_param2_list[k][idx][3]
                        self.kl_alpha_del = kl_gamma(alpha_del_posterior, alpha_del_prior) / model.batch_size
                    else:
                        self.kl_alpha_del = 0.
                else:
                    self.kl_s = 0.
                    
                    alpha_gam_prior = model.prior_theta_param_list[k][idx][0]
                    alpha_gam_posterior = model.posterior_theta_param1_list[k][idx][2]
                    self.kl_alpha_gam = kl_gamma(alpha_gam_posterior, alpha_gam_prior) / model.batch_size
                    if FLAGS.directed == 1:
                        alpha_del_prior = model.prior_theta_param_list[k][idx][1]
                        alpha_del_posterior = model.posterior_theta_param2_list[k][idx][2]
                        self.kl_alpha_del = kl_gamma(alpha_del_posterior, alpha_del_prior) / model.batch_size
                    else:
                        self.kl_alpha_del = 0.
                
                self.kl += self.kl_z + self.kl_s + self.kl_alpha_gam + self.kl_alpha_del
        # Average
        self.kl = self.kl/model.S
        
        self.wu_beta = epoch / FLAGS.epochs
        if FLAGS.use_kl_warmup == 0:
            self.wu_beta = 1.

        self.cost = self.nll + self.wu_beta * self.kl
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # gradient clipping
        self.clipped_grads_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else 0, var) for grad, var in self.grads_vars]

        self.opt_op = self.optimizer.apply_gradients(self.clipped_grads_vars)
       
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds, 0.5), tf.int32),
                                           tf.cast(labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def binary_weighted_crossentropy(self, preds, labels):
        """
        Expects probabilities preds
        pos weight: scaling factor for 1-labels for use in unbalanced datasets with lots of zeros(?)
        See this thread for more: https://github.com/tensorflow/tensorflow/issues/2462
        """
        SMALL_VAL = 10e-8
        epsilon = tf.constant(SMALL_VAL)
        preds = tf.clip_by_value(preds, epsilon, 1-epsilon)
        loss = labels * -tf.math.log(preds) + (1 - labels) * -tf.math.log(1 - preds)

        return loss
