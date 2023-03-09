import numpy as np
import warnings

# Initiate colors
colors_exist = ['blue', 'red', 'green', 'purple', 'yellow']
repeat_num = 2
# colors = np.hstack([np.repeat(colors_exist,repeat_num), 'none'])
colors = colors_exist*repeat_num
colors.append('none')

connected_threshold_default = 4

def generate_random_dots(num_horizontal=5, num_vertical=10, num_kind=3,num_dummy_kind=2):
    if num_kind > len(colors):
        warnings.warn('num_kind is supported up to 5. The color will be duplicated')
    dots_kind_matrix = np.random.randint(0, num_kind + num_dummy_kind + 1,(num_vertical, num_horizontal))
    dots_kind_matrix[dots_kind_matrix>num_kind] = -1
    return dots_kind_matrix

def get_base_dots_info(dots_kind_matrix):
    # Get the shape of box
    num_vertical = dots_kind_matrix.shape[0]
    num_horizontal = dots_kind_matrix.shape[1]
    return num_vertical, num_horizontal

def fall_dots_once(dots_kind_matrix):

    # dots_kind_matrix_falled = np.copy(dots_kind_matrix)
    dots_kind_matrix_falled = dots_kind_matrix

    is_empty_matrix = dots_kind_matrix_falled < 0
    empty_sorted_indecies = np.argsort(is_empty_matrix,axis=0)

    # Get the shape of box
    num_horizontal = dots_kind_matrix_falled.shape[1]

    for target_horizontal_index in range(num_horizontal):
        target_vertical_vector = dots_kind_matrix_falled[:,target_horizontal_index]
        target_vertical_vector = target_vertical_vector[empty_sorted_indecies[:,target_horizontal_index]]
        # print(target_vertical_vector)
        dots_kind_matrix_falled[:,target_horizontal_index] = target_vertical_vector
    return dots_kind_matrix_falled

def connect_dots(dots_kind_matrix):
    # Use while true loop for checking dots connection

    is_checked_matrix = np.full_like(dots_kind_matrix, False, dtype=bool) # when this value is true, it is already checked and no more checking is needed.
    empty_matrix = dots_kind_matrix == -1 # get the empty cells to replace is_checked
    is_checked_matrix[empty_matrix] = True

    up_connected_matrix = connect_dots_up(dots_kind_matrix)
    right_connected_matrix = connect_dots_right(dots_kind_matrix)
    
    # Get the shape of box
    num_vertical = dots_kind_matrix.shape[0]
    num_horizontal = dots_kind_matrix.shape[1]

    connected_dots_list = [] # Initiate a list to hold connecting info
    max_connected_num = 0

    for target_horizontal_index in range(num_horizontal):
        for target_vertical_index in range(num_vertical):
            # Start while loop until the connection ends up?
            adding_connected_dots = np.array([target_vertical_index, target_horizontal_index, False])

            while True:
                # Search not checked adding_connected_dots from bottom, except for the first loop
                # Refresh checking_dots_hor based on checking_adding_connected_dots_index
                if adding_connected_dots.ndim==1:
                    # Check if the initial dot is already checked. This sometimes happen when the dot is lonly.
                    if adding_connected_dots[2] == 1:
                        break

                    checking_adding_connected_dots_index = 0
                    checking_dots_ver = adding_connected_dots[0]
                    checking_dots_hor = adding_connected_dots[1]
                    adding_connected_dots[2] = True
                else:
                    not_yet_checked_index = np.where(adding_connected_dots[:,2]==0)
                    if len(not_yet_checked_index[0]) == 0:
                        adding_connected_dots = np.unique(adding_connected_dots,axis=0) # Enunique
                        adding_connected_dots = np.array([adding_connected_dots[:,0], adding_connected_dots[:,1]]) # Remove is_checked_info
                        connected_dots_list.append(adding_connected_dots)
                        
                        max_connected_num = np.max([max_connected_num, adding_connected_dots[0].size]) # Refresh max_connected_num
                        break
                    
                    checking_adding_connected_dots_index = not_yet_checked_index[0][-1]
                    checking_dots_ver = adding_connected_dots[checking_adding_connected_dots_index,0]
                    checking_dots_hor = adding_connected_dots[checking_adding_connected_dots_index,1]
                    adding_connected_dots[checking_adding_connected_dots_index,2] = True

                # Check if the target dot has been already checked.
                if ~is_checked_matrix[checking_dots_ver, checking_dots_hor]:
                    # When the target dot is to be checked whether it is conncedted up or right
                    is_checked_matrix[checking_dots_ver, checking_dots_hor] = True

                    is_nonchecked_up_connected = \
                        up_connected_matrix[checking_dots_ver, checking_dots_hor] \
                            and ~is_checked_matrix[checking_dots_ver + 1, checking_dots_hor]
                    is_nonchecked_right_connected = \
                        right_connected_matrix[checking_dots_ver, checking_dots_hor] \
                            and ~is_checked_matrix[checking_dots_ver, checking_dots_hor + 1]
                    is_nonchecked_down_connected = \
                        up_connected_matrix[checking_dots_ver-1, checking_dots_hor] \
                            and ~is_checked_matrix[checking_dots_ver-1, checking_dots_hor]

                    if is_nonchecked_right_connected:
                        # When the target dot is connected to right, add the right dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver, checking_dots_hor + 1, False]])

                    if is_nonchecked_up_connected:
                        # When the target dot is connected to upper, add the upeer dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver + 1, checking_dots_hor, False]])

                    if is_nonchecked_down_connected:
                        # When the target dot is connected to upper, add the upeer dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver - 1, checking_dots_hor, False]])

    return connected_dots_list, max_connected_num

def connect_dots_up(dots_kind_matrix):

    empty_matrix = dots_kind_matrix == -1 # Get the empty cells to replace later
    diff_up_matrix = np.vstack([dots_kind_matrix[1:,:] - dots_kind_matrix[0:-1,:], np.ones((1,dots_kind_matrix.shape[1]))]) # if this value is 0, the kinds of dot are the same between upper and lower
    # Note that 1 are inserted at the bottom cells
    
    diff_up_matrix[empty_matrix] = 1 # replace empty cells as 1, meanning not connected
    up_connected_matrix = diff_up_matrix == 0 # Get the upper connected cells
    return up_connected_matrix

def connect_dots_right(dots_kind_matrix):

    empty_matrix = dots_kind_matrix == -1 # Get the empty cells to replace later
    diff_right_matrix = np.hstack([dots_kind_matrix[:,1:] - dots_kind_matrix[:,0:-1], np.ones((dots_kind_matrix.shape[0],1))]) # if this value is 0, the kinds of dot are the same between upper and lower
    # Note that 1 are inserted at the most right cells
    
    diff_right_matrix[empty_matrix] = 1 # Replace empty cells as 1, meanning not connected
    right_connected_matrix = diff_right_matrix == 0 # Get the upper connected cells
    return right_connected_matrix

def delete_connected_dots(dots_kind_matrix, connected_dots_list, connected_threshold=connected_threshold_default):
    
    # dots_kind_matrix_deleted = np.copy(dots_kind_matrix)
    dots_kind_matrix_deleted = dots_kind_matrix
    
    for connected_dots in connected_dots_list:
        if connected_dots[0].size > 3:
            dots_kind_matrix_deleted[connected_dots[0], connected_dots[1]] = -1
    
    return dots_kind_matrix_deleted

def delete_and_fall_dots(dots_kind_matrix, connected_dots_list, connected_threshold=connected_threshold_default):
    # dots_kind_matrix_returned = np.copy(dots_kind_matrix)
    dots_kind_matrix_returned = dots_kind_matrix
    
    dots_kind_matrix_returned = \
        delete_connected_dots(dots_kind_matrix, connected_dots_list)
        
    dots_kind_matrix_returned = fall_dots_once(dots_kind_matrix_returned)
    
    connected_dots_list, max_connected_num = connect_dots(dots_kind_matrix_returned)
    
    is_delete_end = max_connected_num < connected_threshold
    
    return dots_kind_matrix_returned, connected_dots_list, is_delete_end

def delete_and_fall_dots_to_the_end(dots_kind_matrix, connected_threshold=connected_threshold_default):
    connected_dots_list, max_connected_num = connect_dots(dots_kind_matrix)
    is_delete_end = max_connected_num < connected_threshold
    
    dots_transition = [dots_kind_matrix]
    dots_kind_matrix_returned = dots_kind_matrix # Not copy.
    loop_num = 0
    while ~is_delete_end:
        loop_num = loop_num + 1
        
        dots_kind_matrix_returned = np.copy(dots_kind_matrix_returned)
        dots_kind_matrix_returned = \
            delete_connected_dots(dots_kind_matrix_returned, connected_dots_list)
        dots_transition.append(dots_kind_matrix_returned)
        
        dots_kind_matrix_returned = np.copy(dots_kind_matrix_returned)
        dots_kind_matrix_returned = fall_dots_once(dots_kind_matrix_returned)
        dots_transition.append(dots_kind_matrix_returned)
        
# =============================================================================
#         # Do deleting and falling in one function
#         dots_kind_matrix_returned = np.copy(dots_kind_matrix_returned)
#         dots_kind_matrix_returned, _, _ = \
#             delete_and_fall_dots(dots_kind_matrix_returned, connected_dots_list, connected_threshold)
#         dots_transition.append(dots_kind_matrix_returned)
# =============================================================================
            
        connected_dots_list, max_connected_num = connect_dots(dots_kind_matrix_returned)
        is_delete_end = max_connected_num < connected_threshold
        
    return dots_transition, loop_num
    