import pandas
import numpy as np

learning_rate = 1

I_dim = 2
H_dim = 4

epoch_count = 2

weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
weights_HtoO = np.random.uniform(-1, 1, H_dim)

preActivation_H = np.zeros(H_dim)
postActivation_H = np.zeros(H_dim)


# Activation functions
def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


def print_mat(mat):
    for rows in mat:
        for x in rows:
            print('{:.2f}\t'.format(x), end='')
        print()


def print_vec(mat):
    for x in mat:
        print('{:.2f}\t'.format(x),  end='')


input_file = 'data/classification/data.simple.test.100.csv'
print("Import file {} ...".format(input_file))
training_data = pandas.read_csv(input_file)
target_cls = training_data.cls
training_data = training_data.drop(['cls'], axis=1)
training_data = np.asarray(training_data)
training_count = len(training_data[:,0])

#####################
#training
#####################
for epoch in range(epoch_count):
    print("\n\nEpoch {}".format(epoch))
    for sample in range(training_count):
        for node in range(H_dim):
            preActivation_H[node] = np.dot(training_data[sample,:], weights_ItoH[:, node])
            postActivation_H[node] = logistic(preActivation_H[node])

        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)

        FE = postActivation_O - target_cls[sample]

        for H_node in range(H_dim):
            S_error = FE * logistic_deriv(preActivation_O)
            gradient_HtoO = S_error * postActivation_H[H_node]
            for I_node in range(I_dim):
                input_value = training_data[sample, I_node]
                gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivation_H[H_node]) * input_value

                weights_ItoH[I_node, H_node] -= learning_rate * gradient_ItoH

            weights_HtoO[H_node] -= learning_rate * gradient_HtoO

    print('I to H:')
    print_mat(weights_ItoH)

    print()
    print('H to O:')
    print_vec(weights_HtoO)
