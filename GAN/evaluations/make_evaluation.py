from .metrics import calculate_fid_score, calculate_ssim_score, calculate_inception_score, calculate_perceptual_loss

def make_evaluation(config):
    eval_name = config['evaluation']['metric'].lower()
    if eval_name == 'fid':
        return calculate_fid_score
    elif eval_name == 'ssim':
        return calculate_ssim_score
    elif eval_name == 'inception_score':
        return calculate_inception_score
    elif eval_name == 'perceptual_loss':
        return calculate_perceptual_loss
    else:
        raise ValueError(f"Evaluation metric {eval_name} not recognized.")
