import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
import utility.Util as Util

credit_data = Util.open_file("../data/credit_data.csv")

# create lists of types of features
numerical = ["Duration", 'InstallmentRatePecnt', 'PresentResidenceTime', 'Age']
target = ['CreditStatus']

credit_data = credit_data.sample(frac=1).reset_index(drop=True)
train_x, train_y = Util.pre_process_data(credit_data, numerical, target)

# dividing the dataset into training and test sets
x_train, y_train, x_test, y_test = Util.split_data(0.8, train_x, train_y)

n_hidden_1 = 8
n_input = train_x.shape[1]
n_classes = train_y.shape[1]

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")

training_epochs = 1000
display_step = 200
batch_size = 32

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

predictions = Util.multilayer_perceptron(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

print("Training Beginning...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(x_train) / batch_size)
        x_batches = np.array_split(x_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, cost],
                            feed_dict={
                                x: batch_x,
                                y: batch_y,
                                keep_prob: 0.8
                            })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    preds = accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0})
    print("Testing complete", preds)
