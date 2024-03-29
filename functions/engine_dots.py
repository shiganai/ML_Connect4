import numpy as np
import warnings
from functions import UI_dots as ui
import math


import os; os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initiate colors
colors_exist = ['blue', 'red', 'green', 'purple', 'yellow']
repeat_num = 2
colors = ['none']
colors.extend(colors_exist*repeat_num)

connected_threshold_default = 4

num_horizontal_default = 6
num_vertical_default = 12 + 2 # 2 はダミー用(次の2ドットを入れる用)
num_kind_default = 4
num_dummy_kind_default = 2

def refresh_connected_threshold_conv2d(connected_threshold):
    # 中心の1マス + int( ( 最大横に並べた場合の中心からの距離 + カーネルサイズは左右どちらにも伸びるから2で割るけど、そのときceil側を取りたいから+1 ) /2 )
    # = 中心の1マス + int( ( connected_threshold-1 + カーネルサイズは左右どちらにも伸びるから2で割るけど、そのときceil側を取りたいから+1 ) /2 )
    # = 中心の1マス + int( 最大横に並べた場合の中心からの距離 / 2 )
    # 備考: 塊の一番真ん中にあるところから数え始めればいい
    #       塊の一番真ん中からは離れて floor(connected_threshold / 2) までの距離しかいかない
    #       カーネルサイズは真ん中からの距離の2倍だから floor(connected_threshold / 2) * 2
    #       パディングはサイズが変わらないようにすればいい
    onehand_kernel_distance = math.floor(connected_threshold / 2)
    connected_threshold_kernel_size = 1 + onehand_kernel_distance * 2 # 必ず奇数
    connected_threshold_conv2d_tmp = nn.Conv2d(in_channels=1, out_channels=1, \
                                           kernel_size=connected_threshold_kernel_size, \
                                           padding=onehand_kernel_distance, \
                                           bias=False)
    # 中心だけ0, 他は1にする
    nn.init.ones_(connected_threshold_conv2d_tmp.weight)
    connected_threshold_conv2d_tmp.weight.data[0,0,onehand_kernel_distance, onehand_kernel_distance] = 0
    
    return connected_threshold_conv2d_tmp







def generate_empty_dots(\
        num_horizontal=num_horizontal_default, \
        num_vertical=num_vertical_default, \
        num_layer = 1):
    dots_kind_matrix_3D = np.full((num_vertical, num_horizontal, num_layer), 0)
    return dots_kind_matrix_3D

def generate_random_dots(\
        num_horizontal=num_horizontal_default, \
        num_vertical=num_vertical_default, \
        num_kind=num_kind_default, \
        num_dummy_kind=num_dummy_kind_default,
        num_layer = 1):
    if num_kind > len(colors):
        warnings.warn('num_kind is supported up to 5. The color will be duplicated')
    dots_kind_matrix_3D = np.random.randint(0, num_kind + 1 + num_dummy_kind, \
                                         size=(num_vertical, num_horizontal, num_layer))
    dots_kind_matrix_3D[dots_kind_matrix_3D>num_kind] = 0
    # dots_kind_matrix_3D = np.vstack([dots_kind_matrix_3D, np.full((3,num_horizontal),0)])
    return dots_kind_matrix_3D

def get_base_dots_info(dots_kind_matrix_3D):
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)
    # Get the shape of box
    num_vertical = dots_kind_matrix_3D.shape[0]
    num_horizontal = dots_kind_matrix_3D.shape[1]
    num_layer = dots_kind_matrix_3D.shape[2]
    return num_vertical, num_horizontal, num_layer

def convet_2D_dots_to_3D(dots_kind_matrix_3D):
    if dots_kind_matrix_3D.ndim < 3:
        # 単層の入力も多層に変換する
        # スライスするからポインタが外れる。つまり、元のデータには反映が施されなくなる
        dots_kind_matrix_3D = dots_kind_matrix_3D[:,:,np.newaxis]
        
    return dots_kind_matrix_3D

def fall_dots_once(dots_kind_matrix_3D):
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)

    empty_matrix = get_empty_matrix(dots_kind_matrix_3D)
    empty_sorted_indecies = np.argsort(empty_matrix,axis=0)
    
    dots_kind_matrix_falled = np.take_along_axis(dots_kind_matrix_3D, empty_sorted_indecies, axis=0)
    return dots_kind_matrix_falled

def connect_dots(dots_kind_matrix_3D, if_only_toBeDeleted=True, connected_threshold=connected_threshold_default,\
                 start_vh_index=None):
    # start_vh_index は初めに調べるよう指定する用の引数
    # if not(start_vh_index is None) and (start_vh_index.shape[0]!=2):
    #     raise Exception("not(start_vh_index is None) and (start_vh_index.shape[1]==2)")
    
    # Use while true loop for checking dots connection
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)
    
    connected_threshold_conv2d = refresh_connected_threshold_conv2d(connected_threshold)
    
    # Get the shape of box
    num_vertical, num_horizontal, num_layer = get_base_dots_info(dots_kind_matrix_3D)
    
    # しらみつぶし用の各層インデックスを作成しておく
    vertical_index_mesh, horizontal_index_mesh = np.meshgrid(range(num_vertical), range(num_horizontal))
    all_index_order = np.array([vertical_index_mesh.flatten(), horizontal_index_mesh.flatten()]).transpose()

    is_checked_matrix_3D = np.full_like(dots_kind_matrix_3D, False, dtype=bool) # when this value is true, it is already checked and no more checking is needed.
    if if_only_toBeDeleted:
        checking_priority_3D = np.zeros_like(dots_kind_matrix_3D)
        for color_index in range(dots_kind_matrix_3D.max()):
            # ターゲットの色があるところだけを1にする
            each_color_matrix_3D = (dots_kind_matrix_3D == color_index + 1) * 1.0
            # TensorのInputchannelの数を1にするための空次元を挿入
            each_color_matrix_3D = each_color_matrix_3D[:,:,:,np.newaxis]
            # 3次元目を1次元目に、4次元目を2次元目に直す
            each_color_matrix_3D = each_color_matrix_3D.transpose(2,3,0,1)
            # Tensorに変換する. 今回は1次元目を複数にすることで, Conv2d を一気に計算することが目的だから, unsqueeze(0)で1次元目に空次元を追加しない.
            each_color_matrix_3D = torch.tensor(each_color_matrix_3D, dtype=torch.float32, device=device)
            
            # 畳み込みを適用
            each_color_matrix_3D_conved = connected_threshold_conv2d(each_color_matrix_3D)
            each_color_matrix_3D_conved[each_color_matrix_3D==0]=0
            each_color_matrix_3D_conved = each_color_matrix_3D_conved.cpu().detach().numpy().copy()
            
            # 次元を元に戻して最後の空次元を削除
            each_color_matrix_3D_conved = each_color_matrix_3D_conved.transpose(2,3,0,1)[:,:,:,0]
            
            checking_priority_3D = checking_priority_3D + each_color_matrix_3D_conved
            None
        # checking_priorityが低い場合は周りに同じ色が多くないからチェックしなくていい
        # 例えば端っこの場合を考える.
        # 周りに一番ない状況は, 横一直線に並んだ時
        # カーネル内に onehand_kernel_distance 分は少なくともある
        # つまり onehand_kernel_distance を下回った場合は消せるつながりの端点にはなりえない
        # それと、空の場所は checking_priority_3D = 0
        onehand_kernel_distance = math.floor(connected_threshold/2)
        is_checked_matrix_3D[checking_priority_3D < onehand_kernel_distance] = True
        
        # np.argsortが昇順にしか並べられないから負にしておく
        checking_priority_3D = -checking_priority_3D
        None
    else:
        empty_matrix_3D = get_empty_matrix(dots_kind_matrix_3D) # get the empty cells to replace is_checked
        is_checked_matrix_3D[empty_matrix_3D] = True
        checking_index_order = all_index_order

    up_connected_matrix_3D = connect_dots_up(dots_kind_matrix_3D)
    right_connected_matrix_3D = connect_dots_right(dots_kind_matrix_3D)

    connected_dots_list_3D = []
    max_connected_num = np.zeros(num_layer)
    
    for target_layer_index in range(num_layer):
        is_checked_matrix = is_checked_matrix_3D[:,:,target_layer_index]
        is_stacked_matrix = np.copy(is_checked_matrix)
        up_connected_matrix = up_connected_matrix_3D[:,:,target_layer_index]
        right_connected_matrix = right_connected_matrix_3D[:,:,target_layer_index]

        connected_dots_list = [] # Initiate a list to hold connecting info
        
        if if_only_toBeDeleted:
            # TODO: 各層に応じた checking_index_order の設定
            checking_priority = checking_priority_3D[:,:,target_layer_index]
            checking_index_order = all_index_order[checking_priority.flatten().argsort(),:]
            None
        else:
            # 全て調べる場合は何層目に着目していようと順番に探索する. そのための checking_index_order は設定済み
            None
        
        # 初めに調べるよう指定する用の引数 start_vh_index が入力された場合
        # checking_index_order は start_vh_index で置き換える
        if not(start_vh_index is None):
            if start_vh_index[target_layer_index].shape[1]!=2:
                raise Exception("start_vh_index[{}].shape[1]!=2".format(target_layer_index))
            
            checking_index_order = start_vh_index[target_layer_index]
        
        for checking_index_order_index in range(checking_index_order.shape[0]):
            target_vertical_index = checking_index_order[checking_index_order_index, 0]
            target_horizontal_index = checking_index_order[checking_index_order_index, 1]
            
            # Start while loop until the connection ends up?
            adding_connected_dots = np.array([target_vertical_index, target_horizontal_index, False])
            is_stacked_matrix[target_vertical_index, target_horizontal_index] = True
    
            while True:
                # Search not checked adding_connected_dots from bottom, except for the first loop
                # Refresh checking_dots_hor based on checking_adding_connected_dots_index
                if adding_connected_dots.ndim==1:
                    # Check if the initial dot is already checked. This sometimes happen when the dot is lonly.
                    # TODO: どんな時に起きるか具体的に記入
                    if adding_connected_dots[2] == 1:
                        break
    
                    checking_adding_connected_dots_index = 0
                    checking_dots_ver = adding_connected_dots[0]
                    checking_dots_hor = adding_connected_dots[1]
                    adding_connected_dots[2] = True
                else:
                    # adding_connected_dots = np.unique(adding_connected_dots,axis=0) # Enunique # 毎回するべきかどうか悩む
                    not_yet_checked_index = np.where(adding_connected_dots[:,2]==0)
                    if len(not_yet_checked_index[0]) == 0:
                        if if_only_toBeDeleted:
                            if adding_connected_dots.shape[0] < connected_threshold:
                                # connected_threshold を超えないものは加えない
                                break
                            
                        adding_connected_dots = np.array([adding_connected_dots[:,0], adding_connected_dots[:,1]]) # Remove is_checked_info
                        # adding_connected_dots_uniqued = np.unique(adding_connected_dots,axis=1) # Enunique # axisの値が違うことに注意
                        # if not(np.array_equal(adding_connected_dots.flatten().sort(), adding_connected_dots_uniqued.flatten().sort())):
                        #     print("", end="")
                        connected_dots_list.append(adding_connected_dots)
                        
                        if max_connected_num[target_layer_index] < adding_connected_dots.shape[1]:
                            max_connected_num[target_layer_index] = adding_connected_dots.shape[1]
                        
                        break
                    
                    # まだチェックしてないマスの縦横のインデックスを取り出して、これからチェックするからチェックしたフラグを立てる
                    checking_adding_connected_dots_index = not_yet_checked_index[0][-1]
                    checking_dots_ver = adding_connected_dots[checking_adding_connected_dots_index,0]
                    checking_dots_hor = adding_connected_dots[checking_adding_connected_dots_index,1]
                    adding_connected_dots[checking_adding_connected_dots_index,2] = True
    
                # Check if the target dot has been already checked.
                if ~is_checked_matrix[checking_dots_ver, checking_dots_hor]:
                    # When the target dot is to be checked whether it is conncedted up or right
                    is_checked_matrix[checking_dots_ver, checking_dots_hor] = True
    
                    is_nonstacked_right_connected = \
                        right_connected_matrix[checking_dots_ver, checking_dots_hor] \
                            and ~is_stacked_matrix[checking_dots_ver, checking_dots_hor + 1]
                    is_nonstacked_up_connected = \
                        up_connected_matrix[checking_dots_ver, checking_dots_hor] \
                            and ~is_stacked_matrix[checking_dots_ver + 1, checking_dots_hor]
                    is_nonstacked_down_connected = \
                        up_connected_matrix[checking_dots_ver-1, checking_dots_hor] \
                            and ~is_stacked_matrix[checking_dots_ver-1, checking_dots_hor]
    
                    if is_nonstacked_right_connected:
                        # When the target dot is connected to right, add the right dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver, checking_dots_hor + 1, False]])
                        is_stacked_matrix[checking_dots_ver, checking_dots_hor + 1] = True
    
                    if is_nonstacked_up_connected:
                        # When the target dot is connected to upper, add the upeer dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver + 1, checking_dots_hor, False]])
                        is_stacked_matrix[checking_dots_ver + 1, checking_dots_hor] = True
    
                    if is_nonstacked_down_connected:
                        # When the target dot is connected to upper, add the upeer dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver - 1, checking_dots_hor, False]])
                        is_stacked_matrix[checking_dots_ver-1, checking_dots_hor] = True
                        
                    if (if_only_toBeDeleted) and (checking_dots_hor!=0):
                        # 左下からしらみつぶしに探索していない場合は左に戻ることも考える
                        # ただし一番左端の時は別. だけど -1 で一番右端の情報が来る. そして一番右端はつながっていない認識にしているから (checking_dots_hor!=0) は必要ないかも
                        is_nonstacked_left_connected = \
                            right_connected_matrix[checking_dots_ver, checking_dots_hor-1] \
                                and ~is_stacked_matrix[checking_dots_ver, checking_dots_hor-1]
        
                        if is_nonstacked_left_connected:
                            # When the target dot is connected to right, add the right dots to the list
                            adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver, checking_dots_hor - 1, False]])
                            is_stacked_matrix[checking_dots_ver, checking_dots_hor-1] = True
                        
                        
        connected_dots_list_3D.append(connected_dots_list)
    return connected_dots_list_3D, max_connected_num

def connect_dots_up(dots_kind_matrix_3D):
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)

    empty_matrix = get_empty_matrix(dots_kind_matrix_3D) # Get the empty cells to replace later
    
    num_vertical, num_horizontal, num_layer = get_base_dots_info(dots_kind_matrix_3D)
    diff_up_matrix = np.concatenate([\
                                     dots_kind_matrix_3D[1:,:,:] - dots_kind_matrix_3D[0:-1,:,:], \
                                     np.ones((1,num_horizontal,num_layer))\
                                     ], axis=0)
    # Where diff_up_matrix is 0, the kinds of dot are the same between upper and lower
    # Note that 1 are inserted at the bottom cells
    
    diff_up_matrix[empty_matrix] = 1 # replace empty cells as 1, meanning not connected
    up_connected_matrix = diff_up_matrix == 0 # Get the upper connected cells
    return up_connected_matrix

def connect_dots_right(dots_kind_matrix_3D):
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)

    empty_matrix = get_empty_matrix(dots_kind_matrix_3D) # Get the empty cells to replace later
    
    num_vertical, num_horizontal, num_layer = get_base_dots_info(dots_kind_matrix_3D)
    diff_right_matrix = np.concatenate([\
                                     dots_kind_matrix_3D[:,1:,:] - dots_kind_matrix_3D[:,0:-1,:], \
                                     np.ones((num_vertical,1,num_layer))\
                                     ], axis=1)
    # Where diff_right_matrix is 0, the kinds of dot are the same between left and right
    # Note that 1 are inserted at the most right cells
    
    diff_right_matrix[empty_matrix] = 1 # Replace empty cells as 1, meanning not connected
    right_connected_matrix = diff_right_matrix == 0 # Get the upper connected cells
    return right_connected_matrix

def delete_connected_dots(dots_kind_matrix_3D, connected_dots_list_3D, connected_threshold=connected_threshold_default):
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)
    
    dots_kind_matrix_3D_deleted = dots_kind_matrix_3D
    
    for target_layer_index in range(dots_kind_matrix_3D.shape[2]):
        connected_dots_list = connected_dots_list_3D[target_layer_index]
        for connected_dots in connected_dots_list:
            if connected_dots.shape[1] > connected_threshold-1:
                dots_kind_matrix_3D_deleted[connected_dots[0], connected_dots[1], target_layer_index] = 0
    
    return dots_kind_matrix_3D_deleted

def delete_and_fall_dots(dots_kind_matrix_3D, connected_dots_list_3D, connected_threshold=connected_threshold_default):
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)
    
    dots_kind_matrix_3D_result = dots_kind_matrix_3D
    
    dots_kind_matrix_3D_result = \
        delete_connected_dots(dots_kind_matrix_3D, connected_dots_list_3D)
        
    dots_kind_matrix_3D_result = fall_dots_once(dots_kind_matrix_3D_result)
    
    connected_dots_list_3D, max_connected_num = connect_dots(dots_kind_matrix_3D_result)
    
    is_delete_end = max_connected_num < connected_threshold
    
    return dots_kind_matrix_3D_result, connected_dots_list_3D, is_delete_end

def delete_and_fall_dots_to_the_end(dots_kind_matrix_3D, \
                                    connected_threshold=connected_threshold_default, \
                                    if_return_only_result = True):
    dots_kind_matrix_3D = convet_2D_dots_to_3D(dots_kind_matrix_3D)
    
    if not(if_return_only_result):
        dots_transition = np.split(ary=dots_kind_matrix_3D, indices_or_sections=dots_kind_matrix_3D.shape[2], axis=2)
    else:
        dots_transition = []
    
    # 元の状態は保持しておくために np.copy
    dots_kind_matrix_3D_result = np.copy(dots_kind_matrix_3D)
    
    dots_kind_matrix_3D_result = fall_dots_once(dots_kind_matrix_3D_result) # Make sure that dots have falled.
    
    if not(if_return_only_result):
        for falled_check_index in range(dots_kind_matrix_3D.shape[2]):
            dots_at_checking_index_org = dots_kind_matrix_3D[:,:,falled_check_index, np.newaxis]
            dots_at_checking_index_falled = dots_kind_matrix_3D_result[:,:,falled_check_index, np.newaxis]
            if np.any( dots_at_checking_index_org != dots_at_checking_index_falled ):
                # もしどこかに違いがあれば、変化状態を加えておく
                # 落下未処理のものが与えられた場合, それを落とすから, このコードが実行される
                dots_transition[falled_check_index] = np.concatenate([\
                                                                      dots_transition[falled_check_index],\
                                                                      dots_at_checking_index_falled,\
                                                                      ], axis=2)
    
    loop_num = np.zeros(dots_kind_matrix_3D.shape[2], dtype=int)
    # dots_kind_matrix_3D に対応するインデックスの保存
    remaining_index = np.array(range(dots_kind_matrix_3D.shape[2]))
    dots_kind_matrix_3D_remaining = dots_kind_matrix_3D_result[:,:,remaining_index]
    while True:
    
        connected_dots_list_3D, max_connected_num = connect_dots(dots_kind_matrix_3D_remaining)
        is_delete_end = max_connected_num < connected_threshold
        
        # 最終的に出力する結果の更新
        dots_kind_matrix_3D_result[:,:,remaining_index[is_delete_end]] = \
            dots_kind_matrix_3D_remaining[:,:,is_delete_end]
    
        if np.all(is_delete_end):
            break
        
        # まだ終わっていないものだけをトリミング. スライスするのはメモリ的に効率悪いかも
        is_not_end = np.logical_not(is_delete_end)
        trimming_index = np.where(np.logical_not(is_delete_end))[0]
        remaining_index = remaining_index[is_not_end]
        dots_kind_matrix_3D_remaining = dots_kind_matrix_3D_remaining[:,:,is_not_end]
        
        # 無理やりndarrayにしてスライスしようとするとエラーが起きる.
        # a = np.array([[1,2,3],[4,5,6]])
        # b = np.array([[1,2,3,4],[5,6,7,8]])
        # c = [a,a,b]
        # np.array(c)
        # とかすると
        # could not broadcast input array from shape (2,3) into shape (2,)
        # というエラーが起きる.
        # だからおとなしく for loop をつかう.
        connected_dots_list_3D_tmp = []
        for ii in trimming_index:
            connected_dots_list_3D_tmp.append(connected_dots_list_3D[ii])
        connected_dots_list_3D = connected_dots_list_3D_tmp
        
        # 連鎖数の更新
        loop_num[remaining_index] = loop_num[remaining_index] + 1
        
        # 複製してから delete_connected_dots
        dots_kind_matrix_3D_deleted = delete_connected_dots(np.copy(dots_kind_matrix_3D_remaining), connected_dots_list_3D)
        
        # 複製してから fall_dots_once
        dots_kind_matrix_3D_falled = fall_dots_once(np.copy(dots_kind_matrix_3D_deleted))
        
        if not(if_return_only_result):
            for index_based_on_remaining in range(remaining_index.shape[0]):
                index_based_on_all = remaining_index[index_based_on_remaining]
                
                adding_transition = dots_kind_matrix_3D_deleted[:,:,index_based_on_remaining]
                falled_transition = dots_kind_matrix_3D_falled[:,:,index_based_on_remaining]
                if np.all( adding_transition == falled_transition ):
                    # 何も落ちてないから transition に _falled は追加しない.
                    # np.concatenate ように空次元の追加
                    adding_transition = adding_transition[:,:,np.newaxis]
                else:
                    # どれかのドットが落ちたからTransitionに落ちた後も追加
                    adding_transition = np.stack([adding_transition, falled_transition], axis=2)
                    
                dots_transition[index_based_on_all] \
                    = np.concatenate([\
                                dots_transition[index_based_on_all],\
                                adding_transition,\
                                ], axis=2)
        
        dots_kind_matrix_3D_remaining = np.copy(dots_kind_matrix_3D_falled)
        
    return dots_kind_matrix_3D_result, loop_num, dots_transition




def get_candidate_3D(dots_kind_matrix, next_2dots, if_list_all=False):
    
    # next_2dots lie vertically
    candidate_3D_up = get_candidate_3D_vertical(dots_kind_matrix, next_2dots)
    # next_2dots lie horizontally
    candidate_3D_right = get_candidate_3D_horizontal(dots_kind_matrix, next_2dots)
    
    if (not(next_2dots[0] == next_2dots[1])) or (if_list_all):
        next_2dots = [next_2dots[1], next_2dots[0]]
        candidate_3D_down = get_candidate_3D_vertical(dots_kind_matrix, next_2dots)
        candidate_3D_left = get_candidate_3D_horizontal(dots_kind_matrix, next_2dots)
        
        candidate_3D = np.concatenate([\
                                       candidate_3D_up, \
                                       candidate_3D_right, \
                                       candidate_3D_down, \
                                       candidate_3D_left, \
                                       ], \
                                      axis=2)
    else:
        candidate_3D = np.concatenate([\
                                       candidate_3D_up, \
                                       candidate_3D_right, \
                                       ], \
                                      axis=2)
        
    return candidate_3D

def get_candidate_3D_vertical(dots_kind_matrix, next_2dots):
    num_vertical, num_horizontal, num_layer = get_base_dots_info(dots_kind_matrix)
    
    candidate_3D = np.repeat( dots_kind_matrix[:,:,np.newaxis], num_horizontal, axis=2 )
    for horizontal_index in range(num_horizontal):
        candidate_3D[-2,horizontal_index,horizontal_index] = next_2dots[0]
        candidate_3D[-1,horizontal_index,horizontal_index] = next_2dots[1]
        
    return candidate_3D

def get_candidate_3D_horizontal(dots_kind_matrix, next_2dots):
    num_vertical, num_horizontal, num_layer = get_base_dots_info(dots_kind_matrix)
    
    candidate_3D = np.repeat( dots_kind_matrix[:,:,np.newaxis], num_horizontal-1, axis=2 )
    for horizontal_index in range(num_horizontal-1):
        candidate_3D[-2,horizontal_index,horizontal_index] = next_2dots[0]
        candidate_3D[-2,horizontal_index+1,horizontal_index] = next_2dots[1]
        
    return candidate_3D

def get_empty_matrix(dots_kind_matrix):
    return dots_kind_matrix == 0


def get_UD_candidate_3D(dots_kind_matrix, connected_threshold=connected_threshold_default):
    # UD stands for un-determined
    # 現状, 1連鎖が起きる candidate_3D だけを返す.
    
    if (dots_kind_matrix.ndim>2) and (dots_kind_matrix.shape[2]>1):
        raise Exception("only 2D input is allowed")
    elif dots_kind_matrix.ndim>2:
        # 2次元化
        dots_kind_matrix = dots_kind_matrix[:,:,0]
    
    # 落とすべき場所と色の組み合わせを取得する
    place_and_color = []
    empty = get_empty_matrix(dots_kind_matrix)
    num_emtpy = np.sum(empty,axis=0)
    # 上下につながる組み合わせ
    for horizontal_index in range(dots_kind_matrix.shape[1]):
        if num_emtpy[horizontal_index] == 0:
            continue
        
        # num_emtpy[horizontal_index] が 1 の時は -1, 2 の時は -2 の位置に新しい色を入れる
        # そのあと num_vertical 分だけ足せば, 同じ座標を正で得られる
        adding_vertical_index = -num_emtpy[horizontal_index]
        adding_vertical_index = adding_vertical_index + dots_kind_matrix.shape[0]
        
        # 下につながる組み合わせ
        if adding_vertical_index > 0:
            # 列 horizontal_index の元々の頂点は adding_vertical_index - 1 の位置にある
            adding_color = dots_kind_matrix[adding_vertical_index-1, horizontal_index]
            place_and_color.append([adding_vertical_index, horizontal_index, adding_color])
        
        # 右につながる組み合わせ
        # １個右の列の, 今回の列の空の個数 num_emtpy[horizontal_index] -> num_emtpy[horizontal_index]-1 の１つ上の色
        if (horizontal_index < dots_kind_matrix.shape[1]-1)\
                and (num_emtpy[horizontal_index+1] < num_emtpy[horizontal_index]):
            adding_color = dots_kind_matrix[adding_vertical_index, horizontal_index+1]
            place_and_color.append([adding_vertical_index, horizontal_index, adding_color])
            
        # 左につながる組み合わせ
        # １個左の列の, 今回の列の空の個数 num_emtpy[horizontal_index] -> num_emtpy[horizontal_index]-1 の１つ上の色
        if (horizontal_index > 0)\
                and (num_emtpy[horizontal_index-1] < num_emtpy[horizontal_index]):
            adding_color = dots_kind_matrix[adding_vertical_index, horizontal_index-1]
            place_and_color.append([adding_vertical_index, horizontal_index, adding_color])
    
    # 0次元目: 組み合わせの数, 1次元目: 色と場所
    place_and_color = np.array(place_and_color)
    place_and_color = np.unique(place_and_color,axis=0)
    candidate_3D = np.repeat( dots_kind_matrix[:,:,np.newaxis], place_and_color.shape[0], axis=2 )
    for ii in range(place_and_color.shape[0]):
        inserting_vertical_index = place_and_color[ii,0]
        inserting_horizontal_index = place_and_color[ii,1]
        inserting_color = place_and_color[ii,2]
        
        candidate_3D[inserting_vertical_index, inserting_horizontal_index, ii] = inserting_color
    
    if place_and_color.shape[0] > 0:
        # (2,1)x? のリストに戻す
        start_vh_index = place_and_color[:,0:2]
        start_vh_index = np.vsplit(start_vh_index, start_vh_index.shape[0])
        connected_dots_list_3D, max_connected_num = connect_dots(candidate_3D, \
                                            connected_threshold=connected_threshold, \
                                            start_vh_index=start_vh_index)
        
        # ndarrayの参照を渡して, ndarrayの各要素に変化させるから,
        # 変数 candidate_3D = で受けなくても, 変更は適用されるが、
        # 分かりやすくるために受けておく
        candidate_3D = delete_connected_dots(candidate_3D, connected_dots_list_3D, connected_threshold=connected_threshold)
        
        # 1連鎖も起きないやつものは消す
        candidate_3D = np.delete(candidate_3D, max_connected_num<connected_threshold, axis=2)
        
    return candidate_3D

def get_if_generate_new_connection(dots_kind_matrix, place_and_color):
    # この方法は connected_threshold=4 でしか使えない...
    
    vn, hn, _ = get_base_dots_info(dots_kind_matrix)
    
    vi = place_and_color[0]
    hi = place_and_color[1]
    target_color = place_and_color[2]
    
    target_color_matrix = dots_kind_matrix == target_color
    
    ############################
    # xooo
    if (hi+4 < hn) and (np.all(target_color_matrix[vi, hi+1:hi+4])):
        return True
    
    if vi > 0:
        #  x
        #  ooo
        if (hi-0 >= 0) and (hi+3 < hn) and (np.all(target_color_matrix[vi-1, hi-0:hi+3])):
            return True
        
        #   x
        #  ooo
        if (hi-1 >= 0) and (hi+2 < hn) and (np.all(target_color_matrix[vi-1, hi-1:hi+2])):
            return True
        
        #    x
        #  ooo
        if (hi-2 >= 0) and (hi+1 < hn) and (np.all(target_color_matrix[vi-1, hi-2:hi+1])):
            return True
    
    #  ooox
    if (hi-3 >= 0) and (hi+0 < hn) and (np.all(target_color_matrix[vi, hi-3:hi+0])):
        return True
    
    ############################
    # ox
    #  oo
    if (vi>0) and (hi-1 >= 0) and (hi+0 < hn) and (np.all(target_color_matrix[vi, hi-3:hi+0])):
        return True
    
    #  o
    # xoo
    
    # xo
    #  oo
    
    #  x
    #  o
    #  oo
    
    #  ox
    #  oo
    
    #  o
    #  oox
    
    #   o
    # xoo
    
    #  xo
    #  oo
    
    #   x
    #   o
    #  oo
    
    #   ox
    #  oo
    
    #   o
    #  oox
    
    #   xo
    #  oo
    
    ############################
    # xoo
    # o
    
    #  oo
    # xo
    
    # xoo
    #  o
    
    #  x
    #  oo
    #  o
    
    #   x
    #  oo
    #  o
    
    #  oox
    #  o
    
    # xoo
    #   o
    
    #  x
    #  oo
    #   o
    
    #   x
    #  oo
    #   o
    
    #  oox
    #   o
    
    #  oo
    #   ox
    
    #  oox
    #    o
    
    
    
    ############################
    #  o
    #  o
    # xo
    
    #  o
    # xo
    #  o
    
    # xo
    #  o
    #  o
    
    #  x
    #  o
    #  o
    #  o
    
    #  ox
    #  o
    #  o
    
    #  o
    #  ox
    #  o
    
    #  o
    #  o
    #  ox
    