import ga
import bo
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
# Initialize the problem space



# Total Trial = (hw_training_size + hw_trail_size) * (sw_training_size + sw_trail_size) * number of unique layers 


# Fixed hardware and software trial size
hw_trial_size = 10
sw_trial_size = 10
sw_test_size = 1000
hw_training_size = 100
sw_training_size = 100


# Try five different hardware software training size ratio

workload = ['resnet50']
for w in workload:
    probs, layers_counts = ga.get_layers(w)
    training_configs, training_val, best_config, best_edp = bo.hw_optimize(probs, layers_counts, hw_training_size, sw_training_size, hw_trial_size, sw_trial_size, sw_test_size)

    table = pd.DataFrame(columns=['hw_config', 'edp'])
    table['hw_config'] = training_configs
    table['edp'] = training_val
    table.to_csv(w + "_result.csv")
    # save the best config
    with open(w + "_best_config.txt", "w") as f:
        f.write(str(best_config))
        f.write("\n")
        f.write(str(best_edp))
        f.write("\n")