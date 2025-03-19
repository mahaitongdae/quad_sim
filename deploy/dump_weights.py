import argparse
import numpy as np
import os
import torch

ACTION_DIM = 4 # for final layer trim

from code_blocks import (
    headers_network_evaluate,
    headers_network_evaluate_test,
    linear_activation,
    sigmoid_activation,
    relu_activation,
    elu_activation,
    scale_clip_op,
    test_func_with_ones_input
)


def generate(actor, output_f_name=None, test_file=True):
    """
	Generate mlp model source code given a policy object
	Args:
		actor: the DiagnalGaussianActor used in training.
	"""
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
        b = actor.trunk[2 * i].bias.detach().numpy()
        if i != n_layers - 1:
            weights.append(w)
            biases.append(b)
        else:
            weights.append(w[:ACTION_DIM, :]) # Only keep the mean value part
            biases.append(b[:ACTION_DIM])

    structure = """static const int structure[""" + str(n_layers) + """][2] = {"""

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
        output = """static float output_""" + str(l) + """[""" + str(bias_shape[0]) + """];\n"""
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
    for (int i = 0; i < structure[0][0]; i++) {
        output_0[i] = 0;
        for (int j = 0; j < structure[0][1]; j++) {
            output_0[i] += state_array[j] * layer_0_weight[i][j];
        }
        output_0[i] += layer_0_bias[i];
        output_0[i] = elu(output_0[i]);
    }
	"""
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, n_layers - 1):
        for_loop = """
    for (int i = 0; i < structure[""" + str(n) + """][0]; i++) {
        output_""" + str(n) + """[i] = 0;
        for (int j = 0; j < structure[""" + str(n) + """][1]; j++) {
            output_""" + str(n) + """[i] += output_""" + str(n - 1) + """[j] * layer_""" + str(n) + """_weight[i][j];
        }
        output_""" + str(n) + """[i] += layer_""" + str(n) + """_bias[i];
        output_""" + str(n) + """[i] = elu(output_""" + str(n) + """[i]);
    }
    """
        for_loops.append(for_loop)

    n = n_layers - 1
    # the last hidden layer which is supposed to have no non-linearity
    output_for_loop = """
    for (int i = 0; i < structure[""" + str(n) + """][0]; i++) {
        output_""" + str(n) + """[i] = 0;
        for (int j = 0; j < structure[""" + str(n) + """][1]; j++) {
            output_""" + str(n) + """[i] += output_""" + str(n - 1) + """[j] * layer_""" + str(n) + """_weight[i][j];
        }
        output_""" + str(n) + """[i] += layer_""" + str(n) + """_bias[i];
    }
    """
    for_loops.append(output_for_loop)

    output_activation = """
    for (int i = 0; i < structure[2][0]; i++) {
		output_2[i] = tanh(output_2[i]);
	}
	
    """

    for_loops.append(output_activation)

    print_for_test = r"""
    printf("actor output\n");
	for (int i=0; i<4; i++) {
		printf("%f ", output_2[i]);
	}
	
    """

    ## assign network outputs to control
    assignment = """
		control->normalizedForces[0] = clip(scale(output_2[0]), 0.0, 1.0);
		control->normalizedForces[1] = clip(scale(output_2[1]), 0.0, 1.0);
		control->normalizedForces[2] = clip(scale(output_2[2]), 0.0, 1.0);
		control->normalizedForces[3] = clip(scale(output_2[3]), 0.0, 1.0);
	"""

    ## construct the network evaluation function
    if test_file:
        controller_eval = """
void networkEvaluate(float *state_array) {
"""
    else:
        controller_eval = """
void networkEvaluate(control_t *control, const float *state_array) {
"""
    for code in for_loops:
        controller_eval += code

    if not test_file:
        ## assignment to control_n
        controller_eval += assignment
    else:
        ## print out the actor output
        controller_eval += print_for_test

    ## closing bracket
    controller_eval += """
	}
	"""

    ## combine the all the codes
    source = ""
    ## headers
    source += headers_network_evaluate if not test_file else headers_network_evaluate_test
    ## helper functions
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation
    source += elu_activation
    source += scale_clip_op
    ## the network evaluation function
    source += structure
    for output in outputs_strings:
        source += output
    for weight in weights_strings:
        source += weight
    for bias in biases_strings:
        source += bias
    source += controller_eval

    if test_file:
        source += test_func_with_ones_input

    ## add log group for logging
    # source += log_group

    suffix = '_test' if test_file else ''
    output_path = os.path.join(os.getcwd(), output_f_name + suffix + '.c')

    # if output_path:
    with open(output_path, 'w') as f:
        f.write(source)

    return source


if __name__ == '__main__':
    import torch
    import sys
    print(os.path.dirname(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from train.agent.sac.actor import DiagGaussianActor
    import subprocess
    import argparse
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/naliseas-workstation/Documents/haitong/sim_to_real/quad_sim/log/hover-aviary-v0/sac/sac_increase_4s/1/log/best_actor.pth', type=str)
    parser.add_argument('--test', default=False, type=bool)
    args = parser.parse_args()

    actor = DiagGaussianActor(obs_dim=20, action_dim=4, hidden_dim=64, hidden_depth=2,
                              log_std_bounds=[-5., 2.],lipsnet=False)  # hard coded for drone controllers.

    actor.load_state_dict(torch.load(args.path, map_location=torch.device('cpu')))
    fname = "network_evaluate_{}".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    generate(actor, '../deploy/{}'.format(fname), test_file=args.test)
    input = torch.ones(1, 20)
    print('torch output \n')
    print(torch.tanh(actor.trunk(input))[0][:4])
    subprocess.run(["gcc",
                          f"{fname}.c",
                          "-o",
                          f"{fname}",
                          "-lm"], capture_output=True, text=True)
    if args.test:
        results = subprocess.run([f"./{fname}_test"], capture_output=True, text=True)
        print(results.stdout)

