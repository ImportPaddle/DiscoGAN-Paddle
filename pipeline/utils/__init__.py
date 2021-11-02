import numpy as np
import paddle
import torch


def gen_fake_data(seed=100, shape=None):
    if shape is None:
        shape = [64, 3, 32, 32]
    batch_size, channel, input_w, input_H = shape
    np.random.seed(seed)
    data = np.random.randn(batch_size, channel, input_w, input_H).astype(np.float32)
    data_paddle, data_torch = paddle.to_tensor(data), torch.from_numpy(data)
    return data_paddle, data_torch


def data_paddle_2_torch(data_paddle):
    return torch.from_numpy(data_paddle.numpy())


def gen_fake_label(seed=100, shape=None, num_classes=10):
    if shape is None:
        shape = 64
    np.random.seed(seed)
    fake_label = np.random.randint(0, 10, shape)
    label_paddle, label_torch = paddle.to_tensor(fake_label), torch.from_numpy(fake_label)
    return label_paddle, label_torch


def gen_params(model_1):
    model_1_params = model_1.state_dict()
    model_2_params = {}
    for key in model_1_params:
        weight = model_1_params[key].cpu().detach().numpy()
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
        if 'classifier.weight' == key:
            weight = weight.transpose()
        model_2_params[key] = weight
    return model_2_params, model_1_params


def paddle2torch(ckpt_path):
    paddle_model = paddle.load(ckpt_path)
    torch_state_dict = {}
    for key, val in paddle_model.items():
        if '_mean' in key:
            key = key.replace('_mean', 'running_mean')  # old, new
        if '_variance' in key:
            key = key.replace('_variance', 'running_var')
        if 'fc' in key and len(val.shape) == 2:
            val = val.t()
        torch_state_dict[key] = val.detach().numpy()

    # torch.save(torch_state_dict, 'discoGAN.pth')
    return torch_state_dict


def torch2paddle(ckpt_path):
    torch_model = torch.load(ckpt_path)
    paddle_state_dict = {}
    for key, val in torch_model.items():
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')  # old, new
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
        if 'fc' in key and len(val.shape) == 2:
            val = val.t()
        paddle_state_dict[key] = val.data.detach().numpy()

    paddle.save(torch_state_dict, 'discoGAN.pdparams')
    return paddle_state_dict


def torch2paddle_gether(ckpt_path):
    torch_model = torch.load(ckpt_path)
    paddle_state_dict = {}
    tmp_state_dict = {}
    for state_key in torch_model.keys():
        for key, val in torch_model[state_key].items():
            if 'running_mean' in key:
                key = key.replace('running_mean', '_mean')  # old, new
            if 'running_var' in key:
                key = key.replace('running_var', '_variance')
            if 'fc' in key and len(val.shape) == 2:
                val = val.t()
            tmp_state_dict[key] = val.data.detach().numpy()
        paddle_state_dict[state_key] = tmp_state_dict
    # paddle.save(paddle_state_dict, 'discoGAN.pdparams')
    return paddle_state_dict

def gen_npy(seed_list, model_name='resnext'):
    reprod_log_paddle = ReprodLogger()
    reprod_log_torch = ReprodLogger()
    model_paddle, model_torch = gen_model(model_name)
    model_paddle.eval()
    model_torch.eval()
    for seed in seed_list:
        data_paddle, data_torch = gen_fake_data(seed)
        params_paddle, params_torch = gen_params(model_torch)
        model_paddle.set_state_dict(params_paddle)
        model_torch.load_state_dict(params_torch)
        res_paddle, res_torch = model_paddle(data_paddle), model_torch(data_torch)
        reprod_log_paddle.add(f"data_{seed_list.index(seed) + 1}", res_paddle.numpy())
        reprod_log_torch.add(f"data_{seed_list.index(seed) + 1}", res_torch.data.cpu().numpy())
    reprod_log_paddle.save(f"./{model_name}_paddle.npy")
    reprod_log_torch.save(f"./{model_name}_torch.npy")

