from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # initialize weights randomly with mean 0
        self.synaptic_weights_1 = 2 * random.random((3, 3)) - 1
        self.synaptic_weights_2 = 2 * random.random((3, 3)) - 1
        self.synaptic_weights_3 = 2 * random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            output_1 = self.think(training_set_inputs, self.synaptic_weights_1)
            output_2 = self.think(output_1, self.synaptic_weights_2)
            output_3 = self.think(output_2, self.synaptic_weights_3)

            error_3 = training_set_outputs - output_3
            error_2 = dot(self.synaptic_weights_3, error_3.T) * (self.__sigmoid_derivative(output_2).T)
            error_1 = dot(self.synaptic_weights_2, error_2) * (self.__sigmoid_derivative(output_1).T)

            adjustment_3 = dot(output_2.T, error_3)
            adjustment_2 = dot(output_1.T, error_2.T)
            adjustment_1 = dot(training_set_inputs.T, error_1.T)

            self.synaptic_weights_1 += adjustment_1
            self.synaptic_weights_2 += adjustment_2
            self.synaptic_weights_3 += adjustment_3


    # The neural network thinks.
    def think(self, inputs, weights):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, weights))

    def predict(self, inputs):
        l1 = self.__sigmoid(dot(inputs, self.synaptic_weights_1))
        l2 = self.__sigmoid(dot(l1, self.synaptic_weights_2))
        l3 = self.__sigmoid(dot(l2, self.synaptic_weights_3))
        return l3


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights_1)
    print(neural_network.synaptic_weights_2)
    print(neural_network.synaptic_weights_3)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after traning")
    print(neural_network.synaptic_weights_1, neural_network.synaptic_weights_2, neural_network.synaptic_weights_3)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.predict(array([1, 0, 0])))