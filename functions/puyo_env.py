
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import UI_dots as ui
from functions import engine_dots as eg

import torch
device = 'cpu'

class puyo_env:
    
    # def __init__(self):
                
    def __init__(self, \
            dots_kind_matrix=None, \
            num_horizontal = eg.num_horizontal_default, \
            num_vertical = eg.num_vertical_default, \
            num_kind = eg.num_kind_default, \
            num_dummy_kind = eg.num_dummy_kind_default, \
            num_next_2dots = 3,
            max_num_candidate=None):
        # max_num_candidate に np.Inf を設定すれば, 一切切り捨てないことになる.
            
        print("construct of puyo_env is called")

        self.num_horizontal = num_horizontal
        self.num_vertical = num_vertical
        self.num_kind = num_kind
        self.num_dummy_kind = num_dummy_kind
        self.num_next_2dots = num_next_2dots
        
        self.num_single_depth_pattern = self.num_horizontal + (self.num_horizontal-1)
        self.num_candidate = self.num_single_depth_pattern * 2
        self.num_dots = self.num_horizontal * self.num_vertical
        if max_num_candidate is None:
            self.max_num_candidate = self.num_single_depth_pattern
        else:
            self.max_num_candidate = max_num_candidate
        
        self.action_space = self.action_space(self)
        
        self.turn_count_threshold = -1 + 6 * 6
    
# =============================================================================
# =============================================================================
# =============================================================================
# # #     copying openaigym class
# =============================================================================
# =============================================================================
# =============================================================================

    def reset(self, to_use_result_till_max_depth):
        # print("reset of puyo_env is called")
        
        # self.step()の処理を is_play_one_game_called のフラグによって管理しているからこの初期化は重要
        self.is_play_one_game_called = False
        self.to_use_result_till_max_depth = to_use_result_till_max_depth
        
        self.turn_count = 0 # temporary count for stop at some count
        self.termination_determined = False
        if to_use_result_till_max_depth:
            self.result_till_max_depth = None # メモリ食うけど、落とす処理、消す処理が思いから残しておく.
        
        self.reset_kind_matrix()
        self.generate_next_2dots()
        self.update_candidate_single_depth()
        self.candidate_max_depth = None
        self.list_candidate()
        
        state = self.update_state()
        info = None # No idea what info is
        
        return state, info
        
    def step(self, action_single_depth):
        # TODO: 完全に外部から action_single_depth だけを渡されまくった場合, terminated = True になるような入力だとどうなるか整理
        # play_one_game から呼ばれたかどうかで分岐を作る
        # 具体的には loop_num の取得方法と list_candidate() を呼び出すかについて分岐を作る.
        
        # play_one_game から呼ばれた時はすでにデータが用意されている.
        if self.is_play_one_game_called:
            # result_till_max_depth を使ったほうが早いか遅いか比較用分岐
            if self.to_use_result_till_max_depth:
                # デバック用に参照コピー
                result_till_max_depth = self.result_till_max_depth
                
                # 引継ぎデータから連鎖数を取得
                loop_num = self.get_loop_num_from_result_till_max_depth(action_single_depth)
                
                # 選んだ action に対応する落とした結果の盤面を引継ぎデータを参照して更新
                self.dots_kind_matrix = result_till_max_depth[0][0][:,:,action_single_depth]
            else:
                # result_till_max_depth を使わないときは選んだ action に対して落とす演算を行う
                self.dots_kind_matrix = self.candidate_single_depth[:,:,action_single_depth]
                loop_num = self.drop_candidate()
            
            # 引継ぎデータのトリミングを実行
            self.trimming_take_over_info(action_single_depth)
        else:
            self.dots_kind_matrix = self.candidate_single_depth[:,:,action_single_depth]
            loop_num = self.drop_candidate()
        
        # refresh next_2dots and candidate_single_depth
        self.update_next_2dots()
        self.update_candidate_single_depth()
        
        # play_one_game から呼ばれた時は次のデータを用意する
        if self.is_play_one_game_called:
            # ゲームオーバーフラグが立っている場合は次のデータを用意する必要はない
            if not(self.termination_determined):
                self.list_candidate()
        
        ########## define returning value
        state = self.update_state()
        observation = state
        reward = int(loop_num**2)
        
        # if loop_num < 2:
        #     reward = 0
        
        self.turn_count = self.turn_count + 1
        # terminated = self.get_terminated()
        # if terminated:
        #     reward -= 10
            
        terminated = (self.turn_count > self.turn_count_threshold) or (self.get_terminated())
        # terminated = self.get_terminated()
        
        truncated = False
        
        info = None # info is dummpy return value
        
        
        return observation, reward, terminated, truncated, info
        
    class action_space():
        def __init__(self, master):
            print("construct of action_space is called")
            self.master = master
            self.n = self.master.num_candidate
            
        def sample(self):
            return np.random.randint(0,self.n)
        
# =============================================================================
# =============================================================================
# =============================================================================
# # #     original
# =============================================================================
# =============================================================================
# =============================================================================
        
    def reset_kind_matrix(self):
        self.dots_kind_matrix = np.full( (self.num_vertical, self.num_horizontal), 0 )
        
    def generate_next_2dots(self):
        self.next_2dots = np.random.randint(1,self.num_kind+1,(2,self.num_next_2dots))
    
    def update_next_2dots(self):
        adding_2dots = np.random.randint(1,self.num_kind+1,(1,2))
        next_2dots = []
        for ii in range(1,self.num_next_2dots):
            next_2dots.append(self.next_2dots[:,ii])
        next_2dots.append(adding_2dots)
        
        self.next_2dots = np.transpose(np.vstack(next_2dots))
        
    def update_state(self):
        dots_kind_matrix = np.copy(self.dots_kind_matrix)
        
        for horizontal_index in range(self.num_next_2dots):
            dots_kind_matrix[-2,horizontal_index] = self.next_2dots[0,horizontal_index]
            dots_kind_matrix[-1,horizontal_index] = self.next_2dots[1,horizontal_index]
        
        self.dots_kind_matrix_with_candidate = dots_kind_matrix
        
        return dots_kind_matrix.flatten()
        
    def drop_candidate(self):
        dots_kind_matrix_3D_result, loop_num, dots_transition = \
            eg.delete_and_fall_dots_to_the_end(self.dots_kind_matrix, if_return_only_result=True)
        
        self.dots_kind_matrix = dots_kind_matrix_3D_result[:,:,0]
        
        return loop_num[0]
    
    def update_candidate_single_depth(self):
        self.candidate_single_depth = eg.get_candidate_3D(\
                                                    dots_kind_matrix=self.dots_kind_matrix, \
                                                    next_2dots=self.next_2dots[:,0]\
                                                    )
    
    def get_terminated(self):
        terminated = True
        if np.all(self.dots_kind_matrix[-2,:]==0):
            terminated = False
        
        return terminated
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # taken over data definition: list_candidate
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
    
    
    def list_candidate(self):
        
        if self.candidate_max_depth is None:
            # まだ candidate_single_depth が設定されていない,
            # つまり、reset()の直後の場合は,
            # 最初 candidate_single_depth を作るところから始める
                
            have_read_depth = 0
            candidate_max_depth = self.dots_kind_matrix[:,:,np.newaxis]
            if self.to_use_result_till_max_depth:
                result_till_max_depth = []
            
            # 確定連鎖数推移(loop_num_till_max_depth)と
            # どの盤面を選んだか(procedure_till_max_depth)は
            # 後で初期化する.
        else:
            # 引継ぎデータがある時はそれを利用
            # candidate_single_depth の更新が必要
            # 確定連鎖数推移(loop_num_till_max_depth)と
            # どの盤面を選んだか(procedure_till_max_depth),
            # 盤面の結果 result_till_max_depth
            # 一段削る必要があるけど, これは actionを選んだ時にやる
            candidate_max_depth = self.candidate_max_depth
            loop_num_till_max_depth = self.loop_num_till_max_depth
            procedure_till_max_depth = self.procedure_till_max_depth
            if self.to_use_result_till_max_depth:
                result_till_max_depth = self.result_till_max_depth
            
            have_read_depth = self.num_next_2dots - 1
            if not(self.previous_is_NN_value_chosen):
                print("",end="")
        
        for reading_depth_index in range(have_read_depth, self.num_next_2dots):
            # 確定しているドット分, 深読み
            # いずれ NN_value を基準に切り捨てる読みもあるかも
            
            # 前回のデータをトリミングする
            if (self.is_play_one_game_called) and (candidate_max_depth.shape[2] > self.max_num_candidate):
                # もし play_one_game から呼び出されて, かつ candidate_max_depth のサイズが threshold を超えていれば...
                # candidate_max_depth, procedure_till_max_depth, loop_num_till_max_depth を NN_value に基づいて上からトリミングする
                
                NN_values = self.model(candidate_max_depth).to('cpu').detach().numpy().copy().flatten()
                
                if not(self.previous_is_NN_value_chosen):
                    # もし前回確定連鎖数に基づいて手順が選ばれていたら, その手順を取り除かないようにしなきゃいけない
                    # 前回の選択を保存するために procedure_till_max_depth == self.previous_procedure_transition のものは np.Inf にする
                    
                    # 全確定ドットまでの手順を完全に一致させる場合
                    NN_values[\
                              np.all(\
                                     procedure_till_max_depth == self.previous_procedure_transition[1:], \
                                     axis=1, \
                                     ) \
                              ] = np.Inf
                    
                    # # 第一手順だけを一致させる場合. 
                    # # NN_value の最高値を把握して, 手順完全一致のものは +2, それ以外は +1 をする.
                    # # num_nest_2dtos が3以上の場合、最深度の手順以外が確定しちゃう.
                    # max_NN_value = NN_values.max()
                    # NN_values[procedure_till_max_depth[:,0] == self.previous_procedure_transition[1]] = max_NN_value + 1
                    # NN_values[\
                    #           np.all(\
                    #                  procedure_till_max_depth == self.previous_procedure_transition[1:], \
                    #                  axis=1, \
                    #                  ) \
                    #           ] = max_NN_value + 2
                    
                # 昇順にしか並べられないから反転しておく
                NN_values = -NN_values
                sorted_index = NN_values.argsort()
                # 上から threshold 分だけ取り出す
                trimming_index = sorted_index[0:self.max_num_candidate]
                
                # 該当するところだけ抜き出す.
                candidate_max_depth = candidate_max_depth[:,:,trimming_index]
                procedure_till_max_depth = procedure_till_max_depth[trimming_index,:]
                loop_num_till_max_depth = loop_num_till_max_depth[trimming_index,:]
            
            # candidate_max_depth はあるインデックスだけ変更とかしないから、参照コピーで十分
            candidate_max_depth_before = candidate_max_depth
            
            # 増える倍率を取得
            if self.next_2dots[0,reading_depth_index] == self.next_2dots[1,reading_depth_index]:
                # もし次のドットが同じ色なら
                duplicate_num = self.num_single_depth_pattern
            else:
                # もし次のドットが違う色なら
                duplicate_num = self.num_single_depth_pattern * 2
            
            # 一つ前に種類の状態があるか取得
            org_candidate_unit_num = candidate_max_depth_before.shape[2]
            
            # candidate_max_depth の初期化. 
            # ここで新しいオブジェクトを参照するから candidate_max_depth_before に影響はないはず
            # 元々の状態数(org_candidate_unit_num) x 倍率 のサイズになる.
            candidate_max_depth = np.ones(shape=(candidate_max_depth.shape[0],\
                                                    candidate_max_depth.shape[1],\
                                                    org_candidate_unit_num * duplicate_num,\
                                                    ), dtype=int)
                
            # 派生元インデックスは np.repeat で各要素ごとに繰り返すことで得られる
            # 例: [0,1,2] -> [0,0,0,1,1,1,2,2,2] -> [(0,0,0),(0,0,0),(0,0,0), ...] *()は見やすくしただけ. tupleではない
            if reading_depth_index == 0:
                # 一番最初の読みの時には2次元配列の1列目として初期化
                procedure_till_max_depth = \
                    np.repeat(\
                              range(duplicate_num),\
                              org_candidate_unit_num, \
                              axis=0)[:,np.newaxis]
            else:
                # はじめ以外は末尾に加えていく
                procedure_till_max_depth = \
                    np.concatenate([\
                                    np.repeat(\
                                              procedure_till_max_depth, \
                                              duplicate_num,
                                              axis=0), \
                                    np.tile(\
                                              range(duplicate_num), \
                                              org_candidate_unit_num\
                                              )[:,np.newaxis], \
                                    ], axis=1)
            
            # 一手分の候補を列挙するためのループ
            for candidate_max_depth_before_index in range(candidate_max_depth_before.shape[2]):
                # candidate_max_depth_before の種類分だけ繰り返し
                
                # duplicate_num*candidate_max_depth_before_index から
                # duplicate_num*(candidate_max_depth_before_index+1) までの
                # duplicate_num 数分だけ初期値を更新.
                candidate_max_depth[:,:,\
                                      duplicate_num*candidate_max_depth_before_index:\
                                      duplicate_num*(candidate_max_depth_before_index+1)]= \
                    eg.get_candidate_3D(\
                        dots_kind_matrix=candidate_max_depth_before[:,:,candidate_max_depth_before_index], \
                        next_2dots=self.next_2dots[:,reading_depth_index]\
                        )
            
            # 全ての候補を更新し終わったから, いざ落下計算
            candidate_max_depth, loop_num, _ = \
                eg.delete_and_fall_dots_to_the_end(candidate_max_depth, if_return_only_result=True)
                
            # 多次元化しておく
            loop_num = loop_num[:,np.newaxis]
            if reading_depth_index == 0:
                # loop_num_till_max_depth の初期化
                loop_num_till_max_depth = loop_num
            else:
                # 初期化時以外は末尾に加えていく
                loop_num_till_max_depth = \
                    np.concatenate([\
                                    np.repeat(\
                                              loop_num_till_max_depth,\
                                              duplicate_num,
                                              axis=0), \
                                    loop_num,\
                                    ], axis= 1)
                
                
            if self.to_use_result_till_max_depth:
                # トリミングする前に result_till_max_depth に落とした盤面と結果, それまでの手順を保存しておく
                result_till_max_depth.append([candidate_max_depth, loop_num, procedure_till_max_depth])
            
            # もしゲームオーバーの物があるなら消さなきゃいけないデータ: 盤面, それまでの連鎖数経緯, 派生元
            # 基本的に duplicate したやつ
            will_be_terminated_index = np.where(np.any(candidate_max_depth[-2,:,:]!=0, axis=0))[0]
            if will_be_terminated_index.shape[0] == candidate_max_depth.shape[2]:
                # もし全てゲームオーバーだったら...
                # TODO: ゲームオーバーフラグを立てて、そこまで最大連鎖を出せるものを選ぶ
                self.termination_determined = True
                break
            
            elif will_be_terminated_index.shape[0] > 0:
                # もしどれかがゲームオーバーなら, それは排除
                candidate_max_depth = np.delete(candidate_max_depth, obj=will_be_terminated_index, axis=2)
                procedure_till_max_depth = np.delete(procedure_till_max_depth, obj=will_be_terminated_index, axis=0)
                loop_num_till_max_depth = np.delete(loop_num_till_max_depth, obj=will_be_terminated_index, axis=0)
                
        
        
        # 次の先読み用に self を更新
        self.candidate_max_depth = candidate_max_depth
        self.loop_num_till_max_depth = loop_num_till_max_depth
        self.procedure_till_max_depth = procedure_till_max_depth
        self.loop_num_till_max_depth_abst = loop_num_till_max_depth.max(axis=1)[:,np.newaxis]
        # self.loop_num_till_max_depth_abst = \
        #     np.concatenate([\
        #                     loop_num_till_max_depth.max(axis=1)[:,np.newaxis],\
        #                     loop_num_till_max_depth.argmax(axis=1)[:,np.newaxis],
        #                     ], axis=1)
        if self.to_use_result_till_max_depth:
            self.result_till_max_depth = result_till_max_depth
        return
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # taken over data definition: get_loop_num_from_result_till_max_depth, trimming_take_over_info
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
    
    
    
    def get_loop_num_from_result_till_max_depth(self, action_single_depth):
        
        # デバック用に参照コピー
        result_till_max_depth = self.result_till_max_depth
        
        loop_num_single_depth = result_till_max_depth[0][1]
        chosen_loop_num = loop_num_single_depth[action_single_depth]
        
        return chosen_loop_num
    
    def trimming_take_over_info(self, action_single_depth):
            
        # デバック用に参照コピー
        candidate_max_depth = self.candidate_max_depth
        loop_num_till_max_depth = self.loop_num_till_max_depth
        procedure_till_max_depth = self.procedure_till_max_depth
            
        if not(self.previous_is_NN_value_chosen):
            print("",end="")
        
        # 今回の一手と同じ一手を持つ candidate の candidate_max_depth に対するインデックスを取得
        taken_over_candidate_index = np.where(procedure_till_max_depth[:,0] == action_single_depth)[0]
        # candidate_max_depth は同じ一手のモノだけを引き継ぐ
        candidate_max_depth = candidate_max_depth[:,:, taken_over_candidate_index]
        
        if self.to_use_result_till_max_depth:
            result_till_max_depth = self.result_till_max_depth
            # result_till_max_depth も削る. for で各層ごとに処理する必要があるから, 空のリストで初期化しちゃう
            result_till_max_depth_tmp = []
            for reading_depth_index in range(1, len(result_till_max_depth)):
                # トリミングする読み段階の落とした結果とそこまでの手順を抽出
                trimming_dots_result = result_till_max_depth[reading_depth_index][0]
                trimming_loop_num = result_till_max_depth[reading_depth_index][1]
                trimming_procedure = result_till_max_depth[reading_depth_index][2]
                
                
                # トリミングするインデックスを決定
                # 手順書の初めが一致してるものを残す
                trimming_index = np.where( trimming_procedure[:,0] == action_single_depth )[0]
                # 盤面はそのまま, 手順は最初の手順を削る
                trimming_dots_result = trimming_dots_result[:,:,trimming_index]
                trimming_loop_num = trimming_loop_num[trimming_index]
                trimming_procedure = trimming_procedure[trimming_index,1:]
                
                # 新しい変数 result_till_max_depth_tmp に保存しておく
                result_till_max_depth_tmp.append([trimming_dots_result, trimming_loop_num, trimming_procedure])
                
            # ループが終わったら result_till_max_depth を置換
            result_till_max_depth = result_till_max_depth_tmp
            self.result_till_max_depth = result_till_max_depth
            
        # loop_num_till_max_depth, procedure_till_max_depth は一段削って, 同じ一手のモノだけを引き継ぐ
        loop_num_till_max_depth = loop_num_till_max_depth[taken_over_candidate_index, 1:]
        procedure_till_max_depth = procedure_till_max_depth[taken_over_candidate_index, 1:]
        
        # selfを更新しておく
        self.candidate_max_depth = candidate_max_depth
        self.loop_num_till_max_depth = loop_num_till_max_depth
        self.procedure_till_max_depth = procedure_till_max_depth
        
        return
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # #     
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

    def play_one_game(self, model=None, if_disp=False, to_use_result_till_max_depth=False, to_use_ratio=False):
        self.reset(to_use_result_till_max_depth)
        self.is_play_one_game_called = True
        self.model = model
        sum_reward = 0.1
        max_reward = 0
        step_count = 0
        
        dots_transition_only_result = None
        # dots_transition_3D_list[0]でアクセスできるように初期化
        dots_transition_3D_list = [[]]
        # ラベル付けように追加
        title_for_dots_transition_3D_list = []
        
        # 期待値と実際に起きた連鎖数があっているか用ログ
        NN_history = []
        loop_num_history = []
        
        # 確定連鎖数を基に選んだ手順を消さないためのフラグ.
        self.previous_is_NN_value_chosen = True
        self.previous_loop_num_transition = np.zeros(self.num_next_2dots)
        
        actions_and_loop_nums_till_terminated = None # None で初期化しておいて、 初めてかどうか判定に使う.
        while True:
            
            # デバック用に参照コピー
            candidate_single_depth = self.candidate_single_depth
            candidate_max_depth = self.candidate_max_depth
            loop_num_till_max_depth = self.loop_num_till_max_depth
            procedure_till_max_depth = self.procedure_till_max_depth
            loop_num_till_max_depth_abst = self.loop_num_till_max_depth_abst
            
            # action_single_depth は毎ループ必ず選ばれる必要があるから, 
            # None で初期化して, 選ばれなかった場合 Error が起きるようにしておく.
            action_single_depth = None
            choise_str = ""
            
            if if_disp:
                print("At turn: {:>2}".format(self.turn_count), end="")
            
            if self.termination_determined:
                # どうあがいてもゲームオーバーの時は...
                if actions_and_loop_nums_till_terminated is None:
                    # 初めて呼び出されたときに actions_and_loop_nums_till_terminated を初期化
                    # 一番連鎖数が多いもので一番上のものを選出
                    max_loop_num_till_depth = loop_num_till_max_depth_abst.max()
                    max_loop_num_till_depth_index = np.where(loop_num_till_max_depth_abst == max_loop_num_till_depth)[0][0]
                    
                    actions_and_loop_nums_till_terminated = \
                        np.vstack([\
                                        procedure_till_max_depth[max_loop_num_till_depth_index,:],\
                                        loop_num_till_max_depth[max_loop_num_till_depth_index,:]\
                                        ])
                    
                    if if_disp:
                        print("    ###  ALL PATTERN WILL BE TERMINATED  ###  ",end="")
                        print("LN transition will be: ",end="")
                        print(actions_and_loop_nums_till_terminated[1,:],end="")
                
                action_single_depth = actions_and_loop_nums_till_terminated[0,0]
                chosen_loop_num = actions_and_loop_nums_till_terminated[1,0]
                actions_and_loop_nums_till_terminated = actions_and_loop_nums_till_terminated[:,1:]
                
                is_NN_value_chosen = False
                NN_history.append(np.Inf)
                
                if if_disp:
                    # ゲームオーバーするときは出力が寂しいから loop_num を0であっても表示しておく
                    print("\n current LN was {}".format(chosen_loop_num), end="")
                
            else:
                # ゲームオーバーまでの経緯が決定されていなければ...
                
                if if_disp:
                    print()
            
                NN_values = model(candidate_max_depth).to('cpu').detach().numpy().copy().flatten()
                
                NN_values = np.array(NN_values)
                # NNの計算値が最も大きいものを特定
                best_NN_value = NN_values.max()
                best_NN_index = np.where(NN_values == best_NN_value)[0]
                if len(best_NN_index) > 1:
                    best_NN_index = best_NN_index[np.random.randint(0, len(best_NN_index))]
                else:
                    best_NN_index = best_NN_index[0]
                
                # デフォルトで NN_value の判断を採用
                best_procedure_index = best_NN_index 
                is_NN_value_chosen = True
                current_loop_num_transition_by_NN = loop_num_till_max_depth[best_procedure_index, :]
                max_loop_num_till_max_depth_by_NN = loop_num_till_max_depth_abst[best_procedure_index]
                
                # 確定連鎖数が最も大きいものを特定
                max_loop_num_till_depth = loop_num_till_max_depth_abst.max()
                max_loop_num_till_depth_index = np.where(loop_num_till_max_depth_abst == max_loop_num_till_depth)[0]
            
                is_max_loop_num_till_depth_index_chosen_randomly = False
                if len(max_loop_num_till_depth_index) > 1:
                    # たまにNN_valueまで一緒の時がある
                    NN_values_at_bestLN = NN_values[max_loop_num_till_depth_index]
                    NN_values_max_value_at_bestLN = NN_values_at_bestLN.max()
                    # max_loop_num_till_depth_index 内での NN_value が最大値の位置を把握
                    max_loop_num_till_depth_index_best_NN_index = np.where(NN_values_at_bestLN == NN_values_max_value_at_bestLN)[0]
                    if len(max_loop_num_till_depth_index_best_NN_index) > 1: # NN_valueまで一緒の場合
                        # ランダムに選ぶ
                        max_loop_num_till_depth_index_best_NN_index = \
                            max_loop_num_till_depth_index_best_NN_index[np.random.randint(0, len(max_loop_num_till_depth_index_best_NN_index))]
                        
                        is_max_loop_num_till_depth_index_chosen_randomly = True
                    else:
                        max_loop_num_till_depth_index_best_NN_index = max_loop_num_till_depth_index_best_NN_index[0]
                    
                    # 最後に max_loop_num_till_depth_index の中から max_loop_num_till_depth_index_best_NN_index の位置を取得する
                    max_loop_num_till_depth_index = max_loop_num_till_depth_index[max_loop_num_till_depth_index_best_NN_index]
                    
                else:
                    max_loop_num_till_depth_index = max_loop_num_till_depth_index[0]
                
                # 最大確定連鎖がいつ起きるか取得
                max_loop_num_transition = loop_num_till_max_depth[max_loop_num_till_depth_index,:]
                max_loop_num_NN_value = NN_values[max_loop_num_till_depth_index]
                
                    
                if if_disp:
                    print("    ",end="")
                    print("best_NN: {:>5.2f} with LN: "\
                          .format(best_NN_value), \
                          end="")
                    print(current_loop_num_transition_by_NN, end="")
                    print(", ",end="")
                    print("coming_max_LN: {} with NN: {:>5.2f}"\
                          .format(max_loop_num_transition, max_loop_num_NN_value), \
                          end="")
                
                if max_loop_num_till_depth < 1: # 連鎖がない場合
                    is_NN_value_chosen = True
                else: # 連鎖がある場合
                    if max_loop_num_till_depth_index == best_NN_index: # 2つの選択結果が同じだった場合
                        # LN に基づいて選ばれたことにする.
                        # 同じ手順を選んだ場合, それは取り除かれちゃいけないから, 
                        # フラグ: is_NN_value_chosen = self.previous_is_NN_value_chosen = False を立てて,
                        # 関数 list_candidate のループの最初のトリミング分岐を利用する.
                        is_NN_value_chosen = False
                    elif max_loop_num_till_max_depth_by_NN == max_loop_num_till_depth: # 2つの選択結果が異なった場合は NN_value と秤にかける
                        # 起こす連鎖数が同じなのに違う手順が選ばれたとき
                        if is_max_loop_num_till_depth_index_chosen_randomly:
                            # 最高確定連鎖数 と 最高NN_value を持つケースが2つ以上存在して、
                            # 最高確定連鎖数 のインデックスがランダムに選ばれて、
                            # 最高NN_value のインデックスもランダムに選ばれて、
                            # それが一致しない場合
                            # NN_value を選んだことにしておく
                            is_NN_value_chosen = True
                        else:
                            raise Exception('Undefined case')
                    else:
                        if max_loop_num_till_depth > 9: # 10連鎖以上だったら NN_value 関係なく打つ
                            best_procedure_index = max_loop_num_till_depth_index
                            is_NN_value_chosen = False
                        elif max_loop_num_till_depth > best_NN_value: # 確定連鎖数のほうが大きかったらもう打つ
                            best_procedure_index = max_loop_num_till_depth_index
                            is_NN_value_chosen = False
                        else: # NN_valueの期待値が大きいなら次の2ドットに期待. デフォルトで NN_valueを選んでいるから結果の表示以外何もしない
                            is_NN_value_chosen = True
                
                
                # ここまでで candidate_max_depth, loop_num_till_max_depth, procedure_till_max_depth に対する
                # best_procedure_index が決まった
                
                chosen_loop_num = loop_num_till_max_depth[best_procedure_index,0] # この一手の連鎖数を取得
                chosen_loop_num_transition = loop_num_till_max_depth[best_procedure_index,:] # 連鎖数経緯を取得
                chosen_NN_value = NN_values[best_procedure_index]
                action_single_depth = procedure_till_max_depth[best_procedure_index,0] # この一手の procedure を取得
                    
                # デバック用に今回の記録を保存.
                self.previous_is_NN_value_chosen = is_NN_value_chosen
                self.previous_procedure_transition = procedure_till_max_depth[best_procedure_index,:]
                self.previous_loop_num_transition = chosen_loop_num_transition
                
                NN_history.append(chosen_NN_value)
                
                if if_disp:
                    # 後でグラフのタイトルに足すようの string の設定も一緒にする
                    # デフォルトは選んだ手順の NN_value を表示
                    choise_str = "{:>.1f}".format(chosen_NN_value)
                    if (max_loop_num_till_depth_index == best_NN_index) and (max_loop_num_till_depth > 0):
                        # 選んだ手順が同じ場合で1連鎖以上ある場合, 強調する必要なし
                        print(", and same candidate", end="")
                        choise_str = "same {:>.1f}".format(chosen_NN_value)
                    elif (max_loop_num_till_max_depth_by_NN == max_loop_num_till_depth) and (max_loop_num_till_depth > 0):
                        # 選んだ手順が違うが連鎖数が同じで1連鎖以上ある場合, 強調する必要なし
                        print(", LN was same, so chose NN_value", end="")
                        choise_str = "{:>.1f}".format(chosen_NN_value)
                    elif max_loop_num_till_max_depth_by_NN < max_loop_num_till_depth:
                        # NNが選んだ連鎖数が確定連鎖数より小さい場合
                        if is_NN_value_chosen:
                            # NNが選んだ手順が採用された場合, 強調!!!!
                            print(", BUT chose NN", end="")
                            choise_str = "{:>.1f}>{}".format(chosen_NN_value, max_loop_num_transition)
                        else:
                            # NNが選んだ手順が採用されなかった場合, 強調する必要なし
                            print(", so chose loop_num with NN: {:>.2f}".format(chosen_NN_value), end="")
                            choise_str = "LN{}({:>.1f})>{:>.1f}".format(chosen_loop_num_transition, chosen_NN_value, best_NN_value)
                
                # おかしなことが起きてないかチェック用. 消しても問題ない.
                if is_NN_value_chosen:
                    # NNが選んだ手順が採用された場合, その値を記録して, 後の結果と比較する
                    if NN_values[best_procedure_index] != best_NN_value:
                        raise Exception('NN_values[best_procedure_index] != best_NN_value even though is_NN_value_chosen is True')
            
            # =============================================================================
            # ここまでで action_single_depth が NN_value との秤か, actions_and_loop_nums_till_terminated かで決められているはず.
            # =============================================================================
            
            ratio = 1
            if to_use_ratio:
                # NNが出した期待値との比較から倍率を求める
                if chosen_loop_num > 0:
                    # 以前に出した期待値と実際の連鎖数を比較することで、期待値の精度を高める、という目的だったけど,
                    # 確定した次の一手で期待値分の連鎖数を出さないといけないわけじゃないから,
                    # to_use_ratio をデフォルトで False にしておくことで実質的にコメントアウト
                    
                    # 末尾から 確定している2ドットの数+1 分だけ戻る.
                    # self.num_next_2dots = 1 の時は 2だけ戻る. 末尾は今回の予想値, つまり次のやつだから.
                    NN_value_from_past = NN_history[-(self.num_next_2dots+1)]
                    # 今回の連鎖数との差の絶対値をとる. 離れているほど良くない
                    diff_LN = np.abs(chosen_loop_num - NN_value_from_past)
                    # シグモイドで 0-0.5の間に変換する
                    ratio = 1/(1+np.exp(diff_LN))
                    # ベースラインを 0.5 にする.
                    ratio = ratio + 0.5
            
            # =============================================================================
            # プロット用にデータを保存
            # =============================================================================
            if if_disp:
                if chosen_loop_num > 0:
                    if to_use_ratio:
                        print("\ncurrent LN: {}, ratio: {:>.3f}".format(chosen_loop_num, ratio), end="")
                    else:
                        print(", current LN: {}".format(chosen_loop_num), end="")
                print()
            
                _, _, dots_transition_current_turn = \
                    eg.delete_and_fall_dots_to_the_end(\
                                                       candidate_single_depth[:,:,action_single_depth], \
                                                       if_return_only_result=False)
                        
                # 3Dで入力したとき用に dots_transition_current_turn は [?x?x?, ?x?x?] の形で帰ってくるから、
                # 最初の1つ目を取得. というか len(dots_transition_current_turn) = 1 のはず.
                dots_transition_current_turn = dots_transition_current_turn[0]
                
                # 確定している2ドットを右上に追加して, adding_transition として保存
                adding_transition = self.add_next_2dots_multi_depth_to_transition(candidate_single_depth[:,:,action_single_depth])
                if dots_transition_only_result is None:
                    dots_transition_only_result = adding_transition
                else:
                    dots_transition_only_result = np.concatenate([\
                                                      dots_transition_only_result, \
                                                      adding_transition,\
                                                      ], axis=2)
                
                if chosen_loop_num < 1: # 連鎖がない場合はdots_transition_3D_listの最後の要素の最後に追加する
                    # プロット用タイトルにはターン数だけ入力
                    title_for_dots_transition_3D_list.append("{}: ".format(self.turn_count) + choise_str)
                    if dots_transition_3D_list[-1] == []: # まずは初期化
                        dots_transition_3D_list[-1] = adding_transition
                    else:
                        # もし十分連鎖しない盤面が長くなった場合は改行する.
                        if dots_transition_3D_list[-1].shape[2] > 6:
                            dots_transition_3D_list.append(adding_transition)
                        else:
                            dots_transition_3D_list[-1] = \
                                np.concatenate([\
                                                dots_transition_3D_list[-1],\
                                                adding_transition,\
                                                ],axis=2)
                else: # 連鎖がある場合
                    # タイトルは 初めの2つに turn: ?, chosen loop_num: ? を表示させた後、
                    # 空白を繰り返す.
                    title_for_dots_transition_current_turn = [\
                                                              "{}: ".format(self.turn_count) + choise_str, \
                                                              "LN: {}".format(chosen_loop_num),\
                                                              ]
                    title_for_dots_transition_current_turn.extend(["---------"]*(dots_transition_current_turn.shape[2]-2))
                    title_for_dots_transition_3D_list.extend(title_for_dots_transition_current_turn)
                    
                    # 確定している2ドットを右上に追加して, adding_transition として保存
                    adding_transition = self.add_next_2dots_multi_depth_to_transition(dots_transition_current_turn)
                    if dots_transition_3D_list[-1] == []:
                        # すでに初期化されていた場合. 2回連続で連鎖すると起きる
                        dots_transition_3D_list[-1] = adding_transition
                    else:
                        dots_transition_3D_list.append(adding_transition)
                    
                    # 次連鎖しない場合用に空を挿入しておく
                    dots_transition_3D_list.append([])
            
            # =============================================================================
            # 次のターンに self.step(action_single_depth) によって進める
            # =============================================================================
            
            observation, reward, terminated, truncated, info = self.step(action_single_depth)
            
            if to_use_ratio:
                reward = int(reward * ratio * 100)/100
                
            sum_reward += reward
            step_count += 1
            
            if reward > max_reward:
                max_reward = reward
            
            if terminated:
                break
        
        # 連鎖直後に終わると dots_transition_3D_list の末尾に [] が残るから、これを消しておく.
        if dots_transition_3D_list[-1]==[]:
            dots_transition_3D_list = dots_transition_3D_list[0:-1]
            
        # return max_reward + sum_reward, dots_transition_only_result, dots_transition_3D_list, title_for_dots_transition_3D_list
        return max_reward, dots_transition_only_result, dots_transition_3D_list, title_for_dots_transition_3D_list
        # return sum_reward, dots_transition
        
    def add_next_2dots_multi_depth_to_transition(self, dots_kind_matrix_3D):
        dots_kind_matrix_3D = eg.convet_2D_dots_to_3D(dots_kind_matrix_3D)
        
        if self.num_next_2dots > 1:
            # 2つ目以降の2ドットを追加する用の座標を追加
            dots_kind_matrix_3D = np.concatenate([\
                                                  dots_kind_matrix_3D, \
                                                  np.zeros_like(dots_kind_matrix_3D[0:2,:,:]),\
                                                  ], axis=0)
            
            num_layer = dots_kind_matrix_3D.shape[2]
            horizontal_index = 0
            next_2dots = self.next_2dots.transpose()
            for depth_index in range(1, self.num_next_2dots):
                # はじめの2ドットは飛ばして次のから右上に追加
                horizontal_index -= 1
                dots_kind_matrix_3D[-1, [horizontal_index*3 + 1, horizontal_index*3 + 2], :] = \
                    np.repeat(\
                              next_2dots[depth_index,:,np.newaxis],
                              num_layer,
                              axis=1)
        
        return dots_kind_matrix_3D