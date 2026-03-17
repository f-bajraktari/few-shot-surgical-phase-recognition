import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence
import math
from itertools import combinations
from model import CNN_TRX

## Example data
## Video 1 with 10 frames, each frame represented by a 512-dimensional vector
#video1 = torch.randn(10, 3)
## Video 2 with 8 frames, each frame represented by a 512-dimensional vector
#video2 = torch.randn(8, 3)
#
## Padding sequences to the same length
#padded_video1 = pad_sequence(
#    [video1, video2], batch_first=True, padding_value=0.0)
#padded_video2 = pad_sequence(
#    [video2, video1], batch_first=True, padding_value=0.0)
#
## Mask for ignoring padded elements
#src_mask = torch.ones_like(padded_video1)
#src_mask[padded_video1 == 0] = 0
#print(src_mask)

#query_frame = torch.tensor([[[0.00, 0.01], [0.10, 0.11], [0.20, 0.21]], 
#                            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]], 
#                            [[2.00, 2.01], [2.10, 2.11], [2.20, 2.21]], 
#                            [[3.00, 3.01], [3.10, 3.11], [3.20, 3.21]]])
#perm_query_frame = query_frame.permute(0,2,1)
#res_quer_frame = query_frame.reshape(4, -1, 6)
#cat_query_frame = torch.cat((query_frame, query_frame), 0)
#cat_query_frame = torch.cat((query_frame, query_frame), 1)
#cat_query_frame = torch.cat((query_frame, query_frame), 2)
#
#support_frame = query_frame
#
#class_scores = torch.matmul(support_frame, query_frame.transpose(1,0))
#class_scores = torch.softmax(class_scores, dim = 0)
#diff = class_scores - query_frame
#norm_sq = torch.norm(diff, dim=[-2, -1])**2
#distance = torch.div(norm_sq, 3)
#print(distance)

#def create_mask(support_set_length, query_set_length, attention_map, tuples):
#  for id, t in enumerate(tuples):
#    if t[0] > support_set_length or t[1] > support_set_length:
#      attention_map[:, id] = float('-inf')
#    if t[0] > query_set_length or t[1] > query_set_length:
#      attention_map[id, :] = float('-inf')
#
#def create_mask_TRX(support_set_lengths, query_set_lengths, attention_maps, tuples):
#  for id_q, q_length in enumerate(query_set_lengths):
#    for id_s, s_length in enumerate(support_set_lengths):
#      create_mask(s_length, q_length, attention_maps[id_q][id_s], tuples)
#
#
#def delete_tuples(l, n, temporal_set_size):
#    frame_idxs = [i for i in range(1, l+1)]
#    frame_combinations = combinations(frame_idxs, temporal_set_size)
#    cardinality_combs = [comb for comb in frame_combinations]
#    id_to_delete_list = [id for id, t in enumerate(cardinality_combs) if any (x > n for x in t)]
#    return id_to_delete_list
#
##rand_tensor = torch.randn(7, 256, 3)
##rand_tensor = torch.nn.functional.pad(rand_tensor, (0, 0, 0, 0, 0, 3), "constant", 0)
##
##seq_len = 16
##frame_idxs = [i for i in range(seq_len)]
##frame_combinations = combinations(frame_idxs, 2)
##tuples = [torch.tensor(comb) for comb in frame_combinations]
##tuples_len = len(tuples)
##attention_map1 = torch.randn(tuples_len, tuples_len)
##attention_map2 = torch.randn(tuples_len, tuples_len)
##attention_map3 = torch.randn(tuples_len, tuples_len)
##attention_map4 = torch.randn(tuples_len, tuples_len)
##support_set_lengths = [8,15]
##query_set_lengths = [14,10]
##attention_maps = torch.cat([attention_map1, attention_map2, attention_map3, attention_map4])
##attention_maps = attention_maps.reshape(2, 2, tuples_len, -1)
##create_mask_TRX(support_set_lengths, query_set_lengths, attention_maps, tuples)
##print(attention_maps.shape)
#
##seq_len = 100
##test = [delete_tuples(seq_len, i, 2) for i in range(0, seq_len+1)]
##print(f"96: {test[0]}")
##print(f"97: {test[1]}")
##print(f"98: {test[-1]}")
##
##test = delete_tuples(8, 5, 2)
##print(test)
#
#class_softmax = torch.nn.Softmax(dim=0)
#class_scores = torch.randn(5, 66, 10)
#target_n_frames = [10,12,8,6,5]
#tuples_mask = [torch.tensor(delete_tuples(12, n, 3)).cuda() for n in range(13)]
#
##for i_q, query in enumerate(class_scores):
##    n_frames = target_n_frames[i_q]
##    mask = tuples_mask[n_frames-2]
##    for i_t, query in enumerate(query):
##        if i_t in mask:
##            class_scores[i_q][i_t] = 0
##        else:
##            query = class_softmax(query)
##            class_scores[i_q][i_t] = query
#
#tensor = torch.randn(3,6,3,6)
#mask_tensor = torch.zeros_like(tensor)
#tuples_mask = [[0,2,5],[1,2],[4]]
#for video_idx, frames_list in enumerate(tuples_mask):
#  for frame_idx in frames_list:
#    mask_tensor[video_idx, frame_idx, :, :] = 1
#
#tensor = torch.where(mask_tensor.bool(), torch.tensor(float('-inf')), tensor)
#permute_tensor = tensor.permute(2,3,0,1)
#
#print("Test Ende")

def calculate_model_memory(model, input_tensors):
    # Initialize global variable to count activations
    global nb_acts
    nb_acts = 0

    # Define the hook function to count the number of output activations
    def count_output_act(m, input, output):
        global nb_acts
        nb_acts += output.nelement()

    # Register the hook to count activations for each layer of interest
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            module.register_forward_hook(count_output_act)

    # Perform a forward pass to trigger the hooks
    model(*input_tensors)

    # Calculate the number of parameters
    nb_params = sum(p.nelement() for p in model.parameters())

    # Calculate the number of input elements
    nb_inputs = sum(input_tensor.nelement() for input_tensor in input_tensors)

    # Calculate memory usage in bytes
    input_memory = nb_inputs * 4  # Each element is 4 bytes (32 bits)
    param_memory = nb_params * 4
    activation_memory = nb_acts * 4

    # Calculate total memory usage in GB
    total_memory_gb = (input_memory + param_memory + activation_memory) / 1024**3

    return {
        'input_elements': nb_inputs,
        'parameter_elements': nb_params,
        'forward_activations': nb_acts,
        'input_memory_gb': input_memory / 1024**3,
        'parameter_memory_gb': param_memory / 1024**3,
        'activation_memory_gb': activation_memory / 1024**3,
        'total_memory_gb': total_memory_gb
    }

def create_example_model_and_data():
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 1152

            self.way = 5
            self.shot = 5
            self.query_per_class = 1
            self.trans_dropout = 0.1
            self.seq_len = 8
            self.img_size = 224
            self.method = "resnet50"
            self.num_gpus = 1
            self.temp_set = [2]
    args = ArgsObject()
    torch.manual_seed(0)

    device = 'cuda:0'
    model = CNN_TRX(args).to(device)
    
    support_imgs = torch.rand(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]).to(device)
    support_n_frames = torch.full((args.way * args.shot,), args.seq_len).to(device)
    target_n_frames = torch.full((args.way * args.query_per_class,), args.seq_len).to(device)

    # Calculate memory usage
    memory_info = calculate_model_memory(model, (support_imgs, support_labels, target_imgs, support_n_frames, target_n_frames))

    # Print the results
    print('Input elements: {}, Parameter elements: {}, Forward activations: {}'.format(
        memory_info['input_elements'], memory_info['parameter_elements'], memory_info['forward_activations']))
    print('Input memory: {:.4f} GB, Parameter memory: {:.4f} GB, Activation memory: {:.4f} GB, Total memory: {:.4f} GB'.format(
        memory_info['input_memory_gb'], memory_info['parameter_memory_gb'], memory_info['activation_memory_gb'], memory_info['total_memory_gb']))

    return model, support_imgs, support_labels, target_imgs, support_n_frames, target_n_frames

def lookahead_mask(shape):
    # Mask out future entries by marking them with a 1.0
    mask = torch.triu(torch.ones(shape, shape), diagonal=1)
 
    return mask

def padding_mask(input):
    # Create mask which marks the zero padding values in the input by a 1.0
    mask = torch.eq(input, 0).float()

    # The shape of the mask should be broadcastable to the shape
    # of the attention weights that it will be masking later on
    return mask.unsqueeze(0).unsqueeze(1)


# Example usage:
if __name__ == "__main__":
    s_input = torch.tensor([0,1,1,1,1,0,0,0,0])
    q_input = torch.tensor([0,1,1,1,1,0,0,1,1])
    print(f"s_input: {s_input}")
    print(f"q_input: {q_input}")
    
    s_pad = padding_mask(s_input)
    q_pad = padding_mask(q_input)
    s_in_lookahead = lookahead_mask(s_input.shape[0])
    print(f"lookahead: {s_in_lookahead}")
    print(f"q_pad: {q_pad}")
    print(f"s_pad: {s_pad}")
    print(f"Combined: {torch.max(s_pad, s_in_lookahead)}")