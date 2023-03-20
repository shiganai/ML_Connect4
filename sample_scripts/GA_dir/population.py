import numpy as np
import torch
from network import Network
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
        
        if old_population is None:
            # 前世代なしの場合は、個体数全て初期値でモデルを生成する
            self.models = [Network(self.env) for i in range(size)]
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
                model_c = self.old_models[sort_indices[i]]
            else:
                # それ以外はランダムな2つの個体をかけ合わせる
                a, b = np.random.choice(self.size, size=2, p=probs,
                                        replace=False)
                prob_neuron_from_a = 0.5

                # モデルの各ウエイトを50/50の確率で交叉
                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = Network(self.env)

                for layer_index in range(len(model_c.all_layers)):
                    for row_index in range(len(model_a.all_layers[layer_index].weight)):
                        prob = np.random.rand(\
                                       model_c.all_layers[layer_index].weight.data.size()[0], \
                                       model_c.all_layers[layer_index].weight.data.size()[1] \
                                       )
                        prob = torch.tensor(prob, dtype=torch.float32, device=device)
                        
                        model_c.all_layers[layer_index].weight.data = model_b.all_layers[layer_index].weight.data
                        model_c.all_layers[layer_index].weight.data[prob > prob_neuron_from_a] = \
                            model_a.all_layers[layer_index].weight.data[prob > prob_neuron_from_a]
                        None
                    None
                None

            self.models.append(model_c)

    def mutate(self):
        """ 突然変異(mutate)
        :return:
        """
        for model in self.models:
            for layer_index in range(len(model.all_layers)):
                
                if len(model.all_layers[layer_index].weight.data.size()) == 2: # Linear
                    prob = np.random.rand(\
                                   model.all_layers[layer_index].weight.data.size()[0], \
                                   model.all_layers[layer_index].weight.data.size()[1] \
                                   )
                    prob = torch.tensor(prob, dtype=torch.float32, device=device)
                    
                    noise = np.random.rand(\
                                   model.all_layers[layer_index].weight.data.size()[0], \
                                   model.all_layers[layer_index].weight.data.size()[1] \
                                   )
                    noise = (2*noise - 1) * 0.3
                    noise = torch.tensor(noise, dtype=torch.float32, device=device)
                    
                    model.all_layers[layer_index].weight.data[prob < mutation_prob] = \
                        noise[prob < mutation_prob]
                
                if len(model.all_layers[layer_index].weight.data.size()) == 4: # Conv2d
                    prob = np.random.rand(\
                                   model.all_layers[layer_index].weight.data.size()[2], \
                                   model.all_layers[layer_index].weight.data.size()[3] \
                                   )
                    prob = torch.tensor(prob, dtype=torch.float32, device=device)
                    
                    base_noise = np.random.randint(0,2,\
                                   size= (model.all_layers[layer_index].weight.data.size()[2], \
                                          model.all_layers[layer_index].weight.data.size()[3]) \
                                   )
                                   
                    small_noise = np.random.rand(\
                                   model.all_layers[layer_index].weight.data.size()[2], \
                                   model.all_layers[layer_index].weight.data.size()[3] \
                                   )
                        
                    noise = (base_noise*2 - 1) + (small_noise * 2 - 1) * 0.1
                    noise = torch.tensor(noise, dtype=torch.float32, device=device)
                    
                    model.all_layers[layer_index].weight.data[0][0][prob < mutation_prob] = \
                        model.all_layers[layer_index].weight.data[0][0][prob < mutation_prob] * noise[prob < mutation_prob]