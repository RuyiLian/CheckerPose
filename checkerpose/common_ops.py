import numpy as np
import torch


def from_output_to_class_mask(pred_mask_prob, thershold=0.5):
    activation_function = torch.nn.Sigmoid()
    pred_mask_prob = activation_function(pred_mask_prob)
    pred_mask_prob = pred_mask_prob.detach().cpu().numpy()
    pred_mask = np.zeros(pred_mask_prob.shape)
    pred_mask[pred_mask_prob>thershold] = 1.
    return pred_mask


def from_output_to_class_mask_torch(pred_mask_prob, thershold=0.5):
    activation_function = torch.nn.Sigmoid()
    pred_mask_prob = activation_function(pred_mask_prob.detach())
    pred_mask = torch.where(pred_mask_prob > thershold, 1.0, 0.0)
    return pred_mask


def from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, thershold=0.5, divided_num_each_interation=2, binary_code_length=16):
    if BinaryCode_Loss_Type in ["BCE", "L1", "SSIM", "L1_SSIM"]:
        activation_function = torch.nn.Sigmoid()
        pred_code_prob = activation_function(pred_code_prob)
        pred_code_prob = pred_code_prob.detach().cpu().numpy()
        pred_code = np.zeros(pred_code_prob.shape)
        pred_code[pred_code_prob>thershold] = 1.

    elif BinaryCode_Loss_Type == "CE":   
        activation_function = torch.nn.Softmax(dim=1)
        pred_code_prob = pred_code_prob.reshape(-1, divided_num_each_interation, pred_code_prob.shape[2], pred_code_prob.shape[3])
        pred_code_prob = activation_function(pred_code_prob)
        pred_code_prob = pred_code_prob.detach().cpu().numpy()
        pred_code = np.argmax(pred_code_prob, axis=1)
        pred_code = np.expand_dims(pred_code, axis=1)
        pred_code = pred_code.reshape(-1, binary_code_length, pred_code.shape[2], pred_code.shape[3])
        pred_code_prob = pred_code_prob.max(axis=1, keepdims=True)
        pred_code_prob = pred_code_prob.reshape(-1, binary_code_length, pred_code_prob.shape[2], pred_code_prob.shape[3])

    return pred_code


def get_batch_size(second_dataset_ratio, batch_size):
    batch_size_2_dataset = int(batch_size * second_dataset_ratio) 
    batch_size_1_dataset = batch_size - batch_size_2_dataset
    return batch_size_1_dataset, batch_size_2_dataset


# convert str e.g. 1024_256_32 to tuple (1024, 256, 32)
def from_dim_str_to_tuple(src_str):
    if src_str is None:
        return None
    out_str = src_str.split("_")
    out_tup = [int(dim) for dim in out_str]
    out_tup = tuple(out_tup)
    return out_tup
