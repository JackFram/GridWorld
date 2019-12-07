import numpy as np
import torch


def vectorize_action(action):
    ret = np.zeros(4)
    if action == 0:
        ret[0] = 1
    elif action == 1:
        ret[1] = 1
    elif action == 2:
        ret[2] = 1
    elif action == 3:
        ret[3] = 1
    return ret


def save_model(path, f_s_a, f_s, epoch, loss, s_a_optimizer, s_optimizer):
    torch.save({
        'epoch': epoch,
        's_a_state_dict': f_s_a.state_dict(),
        's_state_dict': f_s.state_dict(),
        's_a_optimizer_state_dict': s_a_optimizer.state_dict(),
        's_optimizer_state_dict': s_optimizer.state_dict(),
        'loss': loss,
    }, path)

