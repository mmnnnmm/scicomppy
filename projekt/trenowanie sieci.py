import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

sizes = [784] + [10]*10
learning_rates = {"leaky":0.4, "tanh":0.4,\
                  "relu": 0.2,"sigmoid" :1, "identity":0.35}
for s in learning_rates.keys():
    print(s)
    net_leaky1 = network.Network(sizes, 1, s)
    net_leaky1.SGD(training_data, 30, 30, learning_rates[s],\
                   test_data=test_data)
    if s != "sigmoid":
        net_leaky2 = network.Network(sizes, 200, s)
        net_leaky2.SGD(training_data, 30, 30, learning_rates[s],\
                       test_data=test_data)


# ==============================================================================
