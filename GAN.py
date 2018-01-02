import tensorflow as tf
import numpy as np


def _parse_function(example_proto):
    features = {"sentence": tf.FixedLenFeature([10], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["sentence"]

def get_iterator(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(100000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def getWeightsAndBiases(sent_dim, gen_hidden_dim, dis_hidden_dim, noise_dim):

    """ weights & biases dictionaries """

    weights = {
        "gen_hidden1": tf.Variable(tf.truncated_normal([noise_dim, gen_hidden_dim])),
        "gen_out": tf.Variable(tf.truncated_normal([gen_hidden_dim, sent_dim])),
        "disc_hidden1": tf.Variable(tf.truncated_normal([sent_dim, dis_hidden_dim])),
        "disc_out": tf.Variable(tf.truncated_normal([dis_hidden_dim, 1]))
    }

    biases = {
        "gen_hidden1": tf.Variable(tf.zeros([gen_hidden_dim])),
        "gen_out": tf.Variable(tf.zeros([sent_dim])),
        "disc_hidden1": tf.Variable(tf.zeros([dis_hidden_dim])),
        "disc_out": tf.Variable(tf.zeros([1]))
    }

    return weights, biases


def generator(x, weights, biases):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def discriminator(x, weights, biases):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer





def main():

    """ Training Params """
    num_steps = 10000
    batch_size = 10
    learning_rate = 0.001

    """ Network Params """

    sent_dim = 10
    gen_hidden_dim = 64
    dis_hidden_dim = 64
    noise_dim = 1


    """ Configure iterator of dataset """
    filename = "train.tfrecords"
    iterator = get_iterator(filename, batch_size)


    """ Set necessary params """
    weights, biases = getWeightsAndBiases(sent_dim, gen_hidden_dim, dis_hidden_dim, noise_dim)

    """ Set inputs """
    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, sent_dim], name='disc_input')

    """ Build generator network """
    gen_sample = generator(gen_input, weights, biases)

    """ Build discriminator networks """
    disc_real = discriminator(disc_input, weights, biases)
    disc_fake = discriminator(gen_sample, weights, biases)

    """ Calculations of losses """
    gen_loss = -tf.reduce_mean(tf.log(disc_fake + 0.0001))
    disc_loss = -tf.reduce_mean(tf.log(disc_real + 0.0001) + tf.log(1. - disc_fake))

    """ Set which variables to update """
    gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]

    disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                biases['disc_hidden1'], biases['disc_out']]

    """ Build Optimizers """
    train_gen = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(gen_loss, var_list=gen_vars)
    train_disc = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(disc_loss, var_list=disc_vars)


    """ Initialize variables """
    init = tf.global_variables_initializer()



    with tf.Session() as sess:

        sess.run(init)
        sentences_batch = iterator.get_next()

        for i in range(num_steps):
            batch = sess.run(sentences_batch)
            noise = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            _, _, gen_loss_new, disc_loss_new = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict={disc_input: batch, gen_input: noise})
            if i % 100 == 0 or i == 1:
                print('Step %i: Generator Loss: %s, Discriminator Loss: %s' % (i, gen_loss_new, disc_loss_new))


        for i in range(10):
            noise = np.random.uniform(-1., 1., size=[1, 1])
            g = sess.run([gen_sample], feed_dict={gen_input: noise})

            print(g)


if __name__ == '__main__':
    main()
