# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2, 1]

nn_cross = [2, 2, 1]

nn_stripe = [3, 1]

nn_hexagon = [6, 1]

nn_grid = [4, 2, 1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return 1*(x>=threshold)

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+e**(-steepness*(x-midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(x,0)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -((desired_output-actual_output)**2)/2


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    out = {}
    res = 0
    neurons = net.topological_sort()
    for nrn in neurons:
        s = 0
        for inp in net.get_incoming_neighbors(nrn):
            x = node_value(inp, input_values, out)
            w = net.get_wires(inp, nrn)[0].get_weight()
            s += w*x
        out[nrn]=threshold_fn(s)
        if net.is_output_neuron(nrn):
            res=threshold_fn(s)
    return (res, out)



#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    vs = [[x-step_size, x, x+step_size] for x in inputs]
    al = [[x,y,z] for x in vs[0] for y in vs[1] for z in vs[2]]
    best = max(al, key=lambda k:func(*k))
    return func(*best), best

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    end = wire.endNode
    if net.is_output_neuron(end):
        return set({wire.startNode, wire, end})
    ret = set({wire.startNode, wire})
    for wr in net.get_wires(end):
        ret = ret.union(get_back_prop_dependencies(net, wr))
    return ret


def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    deltas = {}
    res = 0
    neurons = net.topological_sort()
    neurons.reverse()
    for nrn in neurons:
        out = neuron_outputs[nrn]
        if net.is_output_neuron(nrn):
            deltas[nrn]=out*(1-out)*(desired_output-out)
        else:
            sm = sum([wr.get_weight()*deltas[wr.endNode] for wr in net.get_wires(nrn)])
            deltas[nrn] = out*(1-out)*sm
    return deltas

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    deltas = calculate_deltas(net, desired_output, neuron_outputs)
    for wr in net.get_wires():
        w = wr.get_weight()
        w = w + r * node_value(wr.startNode, input_values, neuron_outputs) * deltas[wr.endNode]
        wr.set_weight(w)
    return net
    

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    it = 0
    res, out = forward_prop(net, input_values, sigmoid)
    while accuracy(res,desired_output)<minimum_accuracy:
        net = update_weights(net, input_values, desired_output, out, r)
        res, out = forward_prop(net, input_values, sigmoid)
        it+=1
    return net, it


#### Part 5: Training a Neural Net #############################################




ANSWER_1 = 20
ANSWER_2 = 20
ANSWER_3 = 10
ANSWER_4 = 150
ANSWER_5 = 10

ANSWER_6 = 1
ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A', 'C']
ANSWER_12 = ['A', 'E']


#### SURVEY ####################################################################

NAME = 'Aleksandre Khokhiashvili'
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
