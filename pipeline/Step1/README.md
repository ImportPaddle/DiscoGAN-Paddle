# 网络结构对齐

```shell
# 进入文件夹
cd pipeline/Step1/

# 生成paddle的前向数据
cd DG_paddle/ && python forward.py
# 生成torch的前向数据
cd DG_torch && python forward.py
# 对比生成log
cd ..
python check_step1.py
```

具体地，以PaddlePaddle为例，`forward_alexnet.py`的具体代码如下所示。


diff检查的代码可以参考：[check_step1.py](./check_step1.py)

产出日志如下，同时会将check的结果保存在`forward_diff.log`文件中。

```
[2021/11/02 09:32:41] root INFO: dis_out: 
[2021/11/02 09:32:41] root INFO: 	mean diff: check passed: True, value: 1.4901161193847656e-08
[2021/11/02 09:32:41] root INFO: gen_out: 
[2021/11/02 09:32:41] root INFO: 	mean diff: check passed: True, value: 4.762114258483052e-09
[2021/11/02 09:32:41] root INFO: diff check passed
```

测试通过，对齐成功。
