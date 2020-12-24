import numpy
import scipy.special

class NeuralNet:
    def __init__(self, inputnodes, hidennodes, outputnodes, E, input = 'test'):
        self.inodes = inputnodes
        self.hnodes = hidennodes
        self.onodes = outputnodes
        self.E = E
        try:
            self.wih = numpy.load("wih" + str(input) + ".npy")
            self.who = numpy.load("who" + str(input) + ".npy")
        except:
            self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        
    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin = 2).T
        targets = numpy.array(target_list, ndmin = 2).T
        hiden_inputs = numpy.dot(self.wih, inputs)
        hiden_outputs = self.activation_function(hiden_inputs)
        final_inputs = numpy.dot(self.who, hiden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hiden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.E * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hiden_outputs))
        self.wih += self.E * numpy.dot((hiden_errors * hiden_outputs * (1 - hiden_outputs)), numpy.transpose(inputs))
        
    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin = 2).T
        hiden_inputs = numpy.dot(self.wih, inputs)
        hiden_outputs = self.activation_function(hiden_inputs)
        final_inputs = numpy.dot(self.who, hiden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        
    def outputW(self):
        return [self.who, self.wih]