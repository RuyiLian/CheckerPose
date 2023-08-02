import torch 
import os

def save_checkpoint(path, net, iteration_step, best_score, optimizer, max_to_keep):
    if not os.path.isdir(path):
        os.makedirs(path)
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    saved_ckpt = [int(i) for i in saved_ckpt]
    saved_ckpt.sort()
    
    num_saved_ckpt = len(saved_ckpt)
    if num_saved_ckpt >= max_to_keep:
        os.remove(os.path.join(path, str(saved_ckpt[0])))

    torch.save(
                {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration_step': iteration_step,
                'best_score': best_score
                }, 
                os.path.join(path, str(iteration_step))
            )
        
def get_checkpoint(path):
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    saved_ckpt = [int(i) for i in saved_ckpt]
    saved_ckpt.sort()
    output = os.path.join(path, str(saved_ckpt[-1]))
    print("get_checkpoint: ", output)
    return output

def get_best_checkpoint(path):
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    best_score = -1.0
    best_fn = None
    for ckpt_fn in saved_ckpt:
        cur_best_score = float(ckpt_fn[:6].replace('_', '.'))
        if cur_best_score > best_score:
            best_fn = ckpt_fn
            best_score = cur_best_score
    output = os.path.join(path, str(best_fn))
    print("get_best_checkpoint: ", output)
    return output

def save_best_checkpoint(best_score_path, net, optimizer, best_score, iteration_step):
    if not os.path.isdir(best_score_path):
        os.makedirs(best_score_path)
    saved_ckpt = [f for f in os.listdir(best_score_path) if os.path.isfile(os.path.join(best_score_path, f))]
    if saved_ckpt != []:
        os.remove(os.path.join(best_score_path, saved_ckpt[0]))

    best_score_file_name = '{:.4f}'.format(best_score)
    best_score_file_name = best_score_file_name.replace('.', '_')
    best_score_file_name = best_score_file_name + 'step'
    best_score_file_name = best_score_file_name + str(iteration_step)
    torch.save(
        {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
            'iteration_step': iteration_step
        }, 
        os.path.join(best_score_path, best_score_file_name)
    )
    print("best check point saved in ", os.path.join(best_score_path, best_score_file_name))
