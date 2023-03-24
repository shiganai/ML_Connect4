import numpy as np
import torch
from network import NN_direct_LN_exp, NN_each_LN_exp
from config import elitism_pct, mutation_prob, weights_mutate_power, device


class Population:
    """ 遺伝的アルゴリズムの各世代を管理するクラス
    """

    def __init__(self, env, size=50, old_population=None):
        """ イニシャライザ

        :param size: 各世代の個体数
        :param old_population: (次世代を生成する場合には前世代のPopulationを与える)
        """
        self.size = size
        self.input_size = env.num_dots
        self.env = env
        self.model_basic = lambda env: NN_each_LN_exp(env)
        
        if old_population is None:
            # 前世代なしの場合は、個体数全て初期値でモデルを生成する
            self.models = [self.model_basic(self.env) for i in range(size)]
        else:
            # 前世代が与えられた場合は交叉(crossover),突然変異(mutate)を行い次世代を生成する
            self.old_models = old_population.models
            self.old_fitnesses = old_population.fitnesses
            self.models = []
            self.crossover()
            self.mutate()
        # 各個体の評価値を保存する配列を初期化
        self.fitnesses = np.zeros(self.size)
        

    def crossover(self):
        """ 交叉(crossover)
        :return:
        """
        # 全ての個体の評価値の合計および各個体評価値を正規化
        sum_fitnesses = np.sum(self.old_fitnesses)
        probs = [self.old_fitnesses[i] / sum_fitnesses for i in
                 range(self.size)]

        sort_indices = np.argsort(probs)[::-1]
        for i in range(self.size):
            if i < self.size * elitism_pct:
                # 優秀な個体は(上位20%)はそのまま
                model = self.old_models[sort_indices[i]]
            else:
                # それ以外はランダムな2つの個体をかけ合わせる
                a, b = np.random.choice(self.size, size=2, p=probs,
                                        replace=False)
                prob_neuron_from_a = 0.5

                # モデルの各ウエイトを50/50の確率で交叉
                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = self.model_basic(self.env)

                for layer_index in range(len(model.all_layers)):
                    if len(model.all_layers[layer_index].weight.shape) == 2: # Linear
                        # weight
                        prob = torch.rand(\
                                          size=(\
                                                model.all_layers[layer_index].weight.shape[0], \
                                                model.all_layers[layer_index].weight.shape[1] \
                                                ) \
                                          )
                        
                        # detach().clone()しないと model の変更が model_b にも反映される
                        model.all_layers[layer_index].weight.data = model_b.all_layers[layer_index].weight.data.detach().clone()
                        model.all_layers[layer_index].weight.data[prob < prob_neuron_from_a] = \
                            model_a.all_layers[layer_index].weight.data[prob < prob_neuron_from_a]
                            
                        # bias
                        prob = torch.rand(\
                                          size=(model.all_layers[layer_index].bias.shape[0],), \
                                          )
                        
                        model.all_layers[layer_index].bias.data = model_b.all_layers[layer_index].bias.data.detach().clone()
                        model.all_layers[layer_index].bias.data[prob < prob_neuron_from_a] = \
                            model_a.all_layers[layer_index].bias.data[prob < prob_neuron_from_a]
                        
                        None
                    elif len(model.all_layers[layer_index].weight.shape) == 4: # Conv2d
                        # weight
                        prob = torch.rand(\
                                          model.all_layers[layer_index].weight.shape[2], \
                                          model.all_layers[layer_index].weight.shape[3] \
                                          )
                            
                        model.all_layers[layer_index].weight.data = model_b.all_layers[layer_index].weight.data.detach().clone()
                        model.all_layers[layer_index].weight.data[0][0][prob < prob_neuron_from_a] = \
                            model_a.all_layers[layer_index].weight.data[0][0][prob < prob_neuron_from_a]
                        
                        # bias
                        model.all_layers[layer_index].bias.data = model_b.all_layers[layer_index].bias.data.detach().clone()
                        prob = torch.rand(size=(1,))
                        if prob > prob_neuron_from_a:
                            model.all_layers[layer_index].bias.data = model_a.all_layers[layer_index].bias.data
                        
                        None
                None

            self.models.append(model)

    def mutate(self):
        """ 突然変異(mutate)
        :return:
        """
        for model in self.models:
            for layer_index in range(len(model.all_layers)):
                
                if len(model.all_layers[layer_index].weight.shape) == 2: # Linear
                    weight_noise_size = (model.all_layers[layer_index].weight.shape[0], model.all_layers[layer_index].weight.shape[1])
                    bias_noise_size = (model.all_layers[layer_index].bias.shape[0],)
                elif len(model.all_layers[layer_index].weight.shape) == 4: # Conv2d
                    weight_noise_size = (model.all_layers[layer_index].weight.shape[2], model.all_layers[layer_index].weight.shape[3])
                    bias_noise_size = (1,)
                
                # =============================================================================
                # =============================================================================
                # =============================================================================
                # # #  weight
                # =============================================================================
                # =============================================================================
                # =============================================================================
                    
                prob = torch.rand(weight_noise_size)
                
                base_noise = torch.randint(0,2,size=weight_noise_size)
                small_noise = torch.rand(size=weight_noise_size)
                noise = (base_noise*2 - 1) + (small_noise * 2 - 1) * 0.1
                               
                if len(model.all_layers[layer_index].weight.shape) == 2: # Linear
                    model.all_layers[layer_index].weight.data[prob < mutation_prob] = \
                        model.all_layers[layer_index].weight.data[prob < mutation_prob] * noise[prob < mutation_prob]
                elif len(model.all_layers[layer_index].weight.shape) == 4: # Conv2d
                    model.all_layers[layer_index].weight.data[0][0][prob < mutation_prob] = \
                        model.all_layers[layer_index].weight.data[0][0][prob < mutation_prob] * noise[prob < mutation_prob]
                
                # =============================================================================
                # =============================================================================
                # =============================================================================
                # # #  bias
                # =============================================================================
                # =============================================================================
                # =============================================================================
                prob = torch.rand(size=bias_noise_size)
                
                base_noise = torch.randint(0,2,size=bias_noise_size)
                small_noise = torch.rand(size=bias_noise_size)
                noise = (base_noise*2 - 1) + (small_noise * 2 - 1) * 0.1
                
                if len(model.all_layers[layer_index].weight.shape) == 2: # Linear
                    model.all_layers[layer_index].bias.data[prob < mutation_prob] = \
                        model.all_layers[layer_index].bias.data[prob < mutation_prob] * noise[prob < mutation_prob]
                elif len(model.all_layers[layer_index].weight.shape) == 4: # Conv2d
                    if prob > mutation_prob:
                        model.all_layers[layer_index].bias.data = model.all_layers[layer_index].bias.data * noise
                