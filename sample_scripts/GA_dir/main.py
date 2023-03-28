import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import matplotlib.pyplot as plt
import numpy as np

from network import NN_direct_LN_exp, NN_each_LN_exp, simple_linear_layer, CNN_symmetry
from population import Population

from functions import puyo_env
from functions import UI_dots as ui
if_disp_dots=False

import time

# env = puyo_env.puyo_env(num_next_2dots=2, num_kind=4, max_num_candidate=10, mode_str="NN")
# env = puyo_env.puyo_env(num_next_2dots=1, num_kind=4, max_num_candidate=np.Inf, mode_str="UD_LN")
env = puyo_env.puyo_env(num_next_2dots=2, num_kind=4, max_num_candidate=6, mode_str="UD_LN")
NN = lambda env: CNN_symmetry(env)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preview_ai(env, model):
    # Note that env.reset is called inside of env.play_one_game
    score, dots_transition_only_result, dots_transition_3D_list, title_for_dots_transition_3D_list = \
            env.play_one_game(model, if_disp=True)
    
    ui.scat_dots_by_procedure(dots_transition_3D_list, title_for_dots_transition_3D_list)
    plt.show(block=False)
    
    return score

def eval_network(env, model):
    all_scores = []
    # ここで平均をとる数が多すぎると, 消極的なモノばかりが採用されるようになる.
    for ii in range(4):
        score, _, _, _ = env.play_one_game(model)
        all_scores.append(score)
    # mean_score = np.array(all_scores).mean()
    return all_scores
    

pop_size = 50
population = Population(env=env, NN=NN, size=pop_size)
score = 0
lines = 0
start = time.time()
for i in range(1):
    score = preview_ai(env, population.models[i])
    print("score: {}".format(score))
print("i: {}, time: {}".format(i, time.time()-start))


iteration = 0
while True:
    iteration += 1
    start = time.time()
    print("{} : ".format(iteration), end="")
    
    all_scores = []
    for i in range(pop_size):
        all_score = eval_network(env, population.models[i])
        all_scores.append(all_score)
        all_score = np.array(all_score)
        population.fitnesses[i] = all_score.mean()
        # population.fitnesses[i] = (all_score.max() + all_score.min())/2
        print("{},".format(i), end="")
    print()
    fined = time.time()
    best_model_idx = population.fitnesses.argmax()
    best_model = population.models[best_model_idx]
    score = 0
    lines = 0
    for i in range(1):
        score = preview_ai(env, best_model)
        print("score: {}".format(score))

    all_scores = -np.array(all_scores)
    sorting_index = all_scores.mean(axis=1).argsort()
    all_scores = (-all_scores[sorting_index, :]).tolist()
    print(all_scores)
    print("time: {}".format(fined-start))
    population = Population(env=env, NN=NN, size=pop_size, old_population=population)