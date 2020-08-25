

def save_model(state_dict, ckpt_path, is_best_flag, best_model_path):
    torch.save(state_dict, ckpt_path)
    if is_best_flag:
        best_fpath = best_model_path
        shutil.copyfile(ckpt_path, best_model_path)


def load_model(ckpt_path, model, optimizer, resume_training):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    valid_loss = ckpt['valid_loss'].item()
    if resume_training:
        model.train()
    else:
        model.eval()
    return model, optimizer, epoch, valid_loss
