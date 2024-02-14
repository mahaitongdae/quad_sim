import argparse
import numpy as np
import os
import torch

from code_blocks import (
    headers_network_evaluate,
    linear_activation,
    sigmoid_activation,
    relu_activation,
)


def generate(actor, output_path=None):
    """
	Generate mlp model source code given a policy object
	Args:
		actor: the DiagnalGaussianActor used in training.
	"""
    # # TODO: check if the policy is really a mlp policy
    # trainable_list = policy.get_params()
    #
    # trainable_shapes = []
    # trainable_evals = []
    # for tf_trainable in trainable_list:
    # 	trainable_shapes.append(tf_trainable.shape)
    # 	trainable_evals.append(tf_trainable.eval(session=sess))

    """
	To account for the last matrix which stores the std, 
	the # of layers must be subtracted by 1
	"""
    n_layers = int((len(actor.trunk) + 1) / 2)
    weights_strings = []  # strings
    biases_strings = []  # strings
    outputs_strings = []  # strings

    weights = []
    biases = []
    for i in range(n_layers):
        w = actor.trunk[2 * i].weight.detach().numpy()
        b = actor.trunk[2 * i].weight.detach().numpy()
        weights.append(w)
        biases.append(b)

    structure = """static const int structure[""" + str(int(n_layers / 2)) + """][2] = {"""

    n_weight = 0
    n_bias = 0
    for l in range(n_layers):

        weights_shape = weights[l].shape

        ## it is a weight matrix
        weight = """static const float layer_""" + str(l) + """_weight[""" + str(weights_shape[0]) + """][""" + str(
            weights_shape[1]) + """] = {"""
        for row in weights[l]:
            weight += """{"""
            for num in row:
                weight += str(num) + ""","""
            # get rid of the comma after the last number
            weight = weight[:-1]
            weight += """},"""
        # get rid of the comma after the last curly bracket
        weight = weight[:-1]
        weight += """};\n"""
        weights_strings.append(weight)

        # augment the structure array
        structure += """{""" + str(weights_shape[0]) + """, """ + str(weights_shape[1]) + """},"""

        ## it is a bias vector
        bias_shape = biases[l].shape
        bias = """static const float layer_""" + str(l) + """_bias[""" + str(bias_shape[0]) + """] = {"""
        for num in biases[l]:
            bias += str(num) + ""","""
        # get rid of the comma after the last number
        bias = bias[:-1]
        bias += """};\n"""
        biases_strings.append(bias)

        ## add the output arrays
        output = """static float output_""" + str(n_bias) + """[""" + str(bias_shape[0]) + """];\n"""
        outputs_strings.append(output)

    # complete the structure array
    ## get rid of the comma after the last curly bracket
    structure = structure[:-1]
    structure += """};\n"""

    """
	Multiple for loops to do matrix multiplication
	 - assuming using tanh activation
	"""
    for_loops = []  # strings

    # the first hidden layer
    input_for_loop = """
		for (int i = 0; i < structure[0][1]; i++) {
			output_0[i] = 0;
			for (int j = 0; j < structure[0][0]; j++) {
				output_0[i] += state_array[j] * layer_0_weight[j][i];
			}
			output_0[i] += layer_0_bias[i];
			output_0[i] = tanhf(output_0[i]);
		}
	"""
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, int(n_layers / 2) - 1):
        for_loop = """
		for (int i = 0; i < structure[""" + str(n) + """][1]; i++) {
			output_""" + str(n) + """[i] = 0;
			for (int j = 0; j < structure[""" + str(n) + """][0]; j++) {
				output_""" + str(n) + """[i] += output_""" + str(n - 1) + """[j] * layer_""" + str(n) + """_weight[j][i];
			}
			output_""" + str(n) + """[i] += layer_""" + str(n) + """_bias[i];
			output_""" + str(n) + """[i] = tanhf(output_""" + str(n) + """[i]);
		}
		"""
        for_loops.append(for_loop)

    n = int(n_layers / 2) - 1
    # the last hidden layer which is supposed to have no non-linearity
    output_for_loop = """
		for (int i = 0; i < structure[""" + str(n) + """][1]; i++) {
			output_""" + str(n) + """[i] = 0;
			for (int j = 0; j < structure[""" + str(n) + """][0]; j++) {
				output_""" + str(n) + """[i] += output_""" + str(n - 1) + """[j] * layer_""" + str(n) + """_weight[j][i];
			}
			output_""" + str(n) + """[i] += layer_""" + str(n) + """_bias[i];
		}
		"""
    for_loops.append(output_for_loop)

    ## assign network outputs to control
    assignment = """
		control_n->thrust_0 = output_""" + str(n) + """[0];
		control_n->thrust_1 = output_""" + str(n) + """[1];
		control_n->thrust_2 = output_""" + str(n) + """[2];
		control_n->thrust_3 = output_""" + str(n) + """[3];	
	"""

    ## construct the network evaluation function
    controller_eval = """
	void networkEvaluate(struct control_t_n *control_n, const float *state_array) {
	"""
    for code in for_loops:
        controller_eval += code
    ## assignment to control_n
    controller_eval += assignment

    ## closing bracket
    controller_eval += """
	}
	"""

    ## combine the all the codes
    source = ""
    ## headers
    source += headers_network_evaluate
    ## helper functions
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation
    ## the network evaluation function
    source += structure
    for output in outputs_strings:
        source += output
    for weight in weights_strings:
        source += weight
    for bias in biases_strings:
        source += bias
    source += controller_eval

    ## add log group for logging
    # source += log_group

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)

    return source


if __name__ == '__main__':
    import torch
    from train.agent.sac.actor import DiagGaussianActor

    actor = DiagGaussianActor(obs_dim=18, action_dim=4, hidden_dim=64, hidden_depth=2,
                              log_std_bounds=[-5., 2.])  # hard coded for drone controllers.

    actor.load_state_dict(torch.load('/home/mht/sim_to_real/training/train/log/Quadrotor-v1/sac/reproduce_speder/1/best_actor.pth'))
    generate(actor, '../deploy/network_evaluate.c')
