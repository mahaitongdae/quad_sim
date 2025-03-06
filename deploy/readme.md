# Deploying NN policy on Crazyflie

## Load policy and print output with all-1 input vector
```shell
python dump_weights.py
```

## Generate test file
```shell
gcc network_evaluate_20250305_test.c -o network_evaluate_20250305_test -lm
./network_evaluate_20250305_test
```