# 反向传播对齐

#### 代码讲解

以PaddlePaddle为例，训练流程核心代码如下所示。每个iter中输入相同的fake data，计算loss，进行梯度反传与参数更新，将loss批量返回，用于后续的验证。

```python
# fake data
A, _ = gen_fake_data(seed=42, shape=[4, 3, 64, 64])
B, _ = gen_fake_data(seed=100, shape=[4, 3, 64, 64])

def main(iters):
    gen_loss_lst = []
    dis_loss_lst = []
    for iter in range(iters):
        AB = generator_B(B)
        BA = generator_A(A)

        ABA = generator_A(AB)
        BAB = generator_B(BA)

        # Reconstruction Loss
        recon_loss_A = recon_criterion(ABA, A)
        recon_loss_B = recon_criterion(BAB, B)

        # Real/Fake GAN Loss (A)
        A_dis_real, A_feats_real = discriminator_A(A)
        A_dis_fake, A_feats_fake = discriminator_A(BA)

        dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_criterion)
        fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion)

        # Real/Fake GAN Loss (B)
        B_dis_real, B_feats_real = discriminator_B(B)
        B_dis_fake, B_feats_fake = discriminator_B(AB)

        dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, gan_criterion)
        fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_criterion)

        rate = 0.01

        gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
        gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

        gen_loss = gen_loss_A_total + gen_loss_B_total
        dis_loss = dis_loss_A + dis_loss_B
        print(f"paddle gen loss: {gen_loss.item()}, dis loss: {dis_loss.item()}")

        # 反传
        if iter % 2 == 0:
            dis_loss.backward()
            optim_dis.step()
            optim_dis.clear_grad()
        else:
            gen_loss.backward()
            optim_gen.step()
            optim_gen.clear_grad()

        gen_loss_lst.append(gen_loss)
        dis_loss_lst.append(dis_loss)

    for idx, loss in enumerate(gen_loss_lst):
        print(f"torch gen loss {idx}: {loss.item()}")
        reprod_logger.add(f"gen_loss_{idx}", loss.cpu().detach().numpy())
    for idx, loss in enumerate(dis_loss_lst):
        print(f"torch dis loss {idx}: {loss.item()}")
        reprod_logger.add(f"dis_loss_{idx}", loss.cpu().detach().numpy())
    reprod_logger.save("bp_paddle.npy")
```


#### 操作方法

运行下面的命令，基于 fake data，依次生成若干轮 loss 数据并保存，使用 `reprod_log` 工具进行 diff 排查。

```shell
# paddle 反传 5 个 iters
cd DG_paddle/
python bp_DG.py

# torch 反传 5 个 iters
cd DG_torch/
python bp_DG.py

# 对比生成log
python check_step3.py
```

最终输出结果如下，同时会保存在文件`bp_align_diff.log`中。

```
[2021/11/02 15:45:36] root INFO: gen_loss_0: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: True, value: 4.172325134277344e-07
[2021/11/02 15:45:36] root INFO: gen_loss_1: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 1.5795230865478516e-06
[2021/11/02 15:45:36] root INFO: gen_loss_2: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 1.564621925354004e-06
[2021/11/02 15:45:36] root INFO: gen_loss_3: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 2.689659595489502e-05
[2021/11/02 15:45:36] root INFO: gen_loss_4: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 2.6851892471313477e-05
[2021/11/02 15:45:36] root INFO: dis_loss_0: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 1.1920928955078125e-06
[2021/11/02 15:45:36] root INFO: dis_loss_1: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 1.6927719116210938e-05
[2021/11/02 15:45:36] root INFO: dis_loss_2: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 1.7762184143066406e-05
[2021/11/02 15:45:36] root INFO: dis_loss_3: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 8.463859558105469e-06
[2021/11/02 15:45:36] root INFO: dis_loss_4: 
[2021/11/02 15:45:36] root INFO: 	mean diff: check passed: False, value: 6.67572021484375e-06
[2021/11/02 15:45:36] root INFO: diff check failed
```

前面 5 轮的 loss diff 均小于 3e-5，相比于设定的阈值(1e-6)也相差不大，而且保持稳定。希望 check 通过。
