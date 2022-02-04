from __future__ import division
import tensorflow as tf
import numpy as np

SMALL = 1e-16
SMALL2 = 1e-8
EULER_GAMMA = 0.5772156649015329


def logit(x):
    return tf.math.log(x + SMALL2) - tf.math.log(1. - x + SMALL2)

def log_density_logistic(logalpha, sample, temperature):
    """
    log-density of the Logistic distribution, from 
    Maddison et. al. (2017) (right after equation 26)
    Input logalpha is a logit (alpha is a probability ratio)
    """
    exp_term = logalpha - sample * temperature
    log_prob = exp_term + np.log(temperature) - 2. * tf.nn.softplus(exp_term)
    return log_prob

def sample_normal(mean, log_std):

    # mu + standard_samples * stand_deviation
    x = mean + tf.random.normal(tf.shape(mean)) * tf.exp(log_std)
    return x

def sample_bernoulli(p_logit):

    p = tf.nn.sigmoid(p_logit)
    u = tf.random.uniform(tf.shape(p_logit), 1e-4, 1. - 1e-4)
    x = tf.round(u > p)
        
    return x

def sample_binconcrete(pi_logit, temperature):

    # Concrete instead of Bernoulli
    u = tf.random.uniform(tf.shape(pi_logit), 1e-4, 1. - 1e-4)
    L = tf.math.log(u) - tf.math.log(1. - u)
    x_logit = (pi_logit + L) / temperature
    #s = tf.sigmoid(logit)
        
    return x_logit

def sample_gamma(alpha, beta):

    u = tf.random.uniform(tf.shape(alpha), 1e-4, 1. - 1e-4)
    x = tf.exp(-tf.math.log(beta + SMALL) + (tf.math.log(u) + tf.math.log(alpha + SMALL) + tf.math.lgamma(alpha + SMALL)) / (alpha + SMALL))
        
    return x

def kl_normal(mean_posterior, log_std, mean_prior = 0.):
    #mean, log_std: d × N × K
    kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + 2 * log_std - tf.square(mean_posterior - mean_prior) - tf.square(tf.exp(log_std)), axis = 1))
    return kl

def kl_binconcrete(logit_posterior, logit_prior, sample, temp_posterior, temp_prior):
    """
    KL divergence between the prior and posterior
    inputs are in logit-space
    """
    log_prior = log_density_logistic(logit_prior, sample, temp_prior)
    log_posterior = log_density_logistic(logit_posterior, sample, temp_posterior)
    kl = log_posterior - log_prior
    return tf.reduce_mean(tf.reduce_sum(kl, axis=1))

def kl_gamma(alpha_posterior, alpha_prior):
    """
    KL divergence between the prior and posterior
    """
    kl = tf.math.lgamma(alpha_prior) - tf.math.lgamma(alpha_posterior + SMALL) + (alpha_posterior - alpha_prior) * tf.math.digamma(alpha_posterior + SMALL)
    return tf.reduce_mean(tf.reduce_sum(kl, axis = 1))

def scaled_distance(x1, x2, scale):

    return tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(x1 - x2, scale)), axis = 1) + SMALL2)
