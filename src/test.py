import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2

net = network2.Network([784, 30, 10])

net.SGD(training_data[:5000], 30, 10,
        lmbda=1.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)

# with out linesearch eta
# net.SGD(training_data, 30, 10, 1.0,
#         lmbda=1.0,
#         evaluation_data=validation_data,
#         monitor_evaluation_accuracy=True)
