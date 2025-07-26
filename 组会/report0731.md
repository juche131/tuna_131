## 第五章：Pytorch优化模块

$ 吴迪，2025年7月31日 $

在书中给出的Covid-19分类案例中，代码被分成数据、模型、优化、迭代四个模块。其中，优化和迭代的模块，实际上承担了训练模型的任务。源代码如下所示。

###### 代码片段 {#1}
```python
    # step 3/4 : 优化模块
    loss_f = nn.CrossEntropyLoss()   # 损失函数：交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)   # 优化器：SGD（随机梯度下降）优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)   # 学习率调整器：每经过step_size个epoch，学习率乘以gamma

    # step 4/4 : 迭代模块
    for epoch in range(100):
        # 训练集训练
        model.train()
        for data, labels in train_loader:
            # forward & backward
            outputs = model(data)    # 训练组数据的训练结果
            optimizer.zero_grad()    # 手动梯度置零

            # loss 计算
            loss = loss_f(outputs, labels)   # 计算损失函数
            loss.backward()   # 反向传播使得nn.Module的Parameter有了.grad值
            optimizer.step()   # 执行一步更新

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)   # 进行分类，predicted记录所属类别的索引
            correct_num = (predicted == labels).sum()   #分类正确的个数
            acc = correct_num / labels.shape[0]   # 正确率
            print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))
            # print(predicted, labels)报错

        # 验证集验证
        model.eval()
        for data, label in valid_loader:
            # forward
            outputs = model(data)

            # loss 计算
            loss = loss_f(outputs, labels)

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == labels).sum()
            acc_valid = correct_num / labels.shape[0]
            print("Epoch:{} Valid Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc_valid))

        # 添加停止条件
        if acc_valid == 1:
            break

        # 学习率调整
        scheduler.step()

```

以上代码说明了损失函数、优化器和学习率调整器的实现方法。现在介绍这三者的基本概念。

### 损失函数
损失函数（loss function）是用来衡量模型输出与真实标签之间的差异，可以看作“惩罚”，是优化器更新模型的关键指标。针对不同的任务有不同的损失函数，例如回归任务常用均方误差损失函数(Mean Square Error，特点为高误差会获得较大的“惩罚”)，分类任务常用交叉熵损失函数（Cross Entropy，特点为只关注正确“类别”的预测概率），这是根据标签的特征来决定的。

书中提到了两个重要的损失函数：`L1Loss`和`CrossEntropyLoss`。我们将会介绍这两个函数的原理和应用场景。

为便于修改参数及附加自定义功能，Pytorch将损失函数封装成了类对象。使用时应当先[实例化](#代码片段-1)。

#### `L1Loss`损失函数
`L1Loss`（也称为平均绝对误差/MAE）是 PyTorch 中最简单的回归损失函数之一，它计算预测值与目标值之间**绝对差值的平均值**。

对于输入张量 $x$ 和目标张量 $y$（形状均为 $(N, *)$，其中 $N$ 是批大小，`*` 表示任意额外维度）：

$$
\text{L1Loss} = 
\begin{cases} 
\frac{1}{n}\sum_{i=1}^{n}|x_i - y_i| & \text{if reduction='mean'} \\
\sum_{i=1}^{n}|x_i - y_i| & \text{if reduction='sum'} 
\end{cases}
$$

对于`torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')`类：

+ `size_average`和`reduce`都已经被舍弃，因考虑版本兼容性而保留。
+ `reduction`有`None`、`mean`和`sum`三个选项。指定为`None`时，实例化损失函数并传入数据时，返回一个同维度的`Tensor`；一般指定为`mean`和`sum`，返回一个标量，表示对差求平均值或求和。

`L1Loss`通常用于允许离群值存在的稳健回归的场景。

#### `CrossEntropyLoss`损失函数
交叉熵损失函数常用于分类问题。对于`torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)`类：
+ `weight (Tensor, optional)`：类别权重，用于调整各类别的损失重要程度，常用于类别不均衡的情况。
+ `ignore_index (int, optional)`：忽略某些类别索引不进行loss计算。
+ `label_smoothing (float, optional)`：标签平滑系数，用于减少方差，防止过拟合。理论值域为0.0\~1.0，实际应用中通常被设置为0.01\~0.1。
> ##### 对于标签平滑：
> 当 `label_smoothing > 0` 时，真实标签被调整为：
>$$
>y_{\text{smooth}} = (1 - \epsilon) \cdot y + \frac{\epsilon}{C}
>$$
>其中：
>+ $\epsilon$ 是平滑系数
>+ $C$ 是类别总数

除了以上两个损失函数外，还有很多不同的损失函数，具有不同的功能，需要根据任务要求和规模选择适当的函数。

### 优化器
优化器是模型的参数更新引擎，负责根据损失函数的梯度信息调整模型参数，使损失函数最小化。
> #### 优化器是如何工作的？
> 原文中抛出了这个问题，但没有给出答案。事实上，优化器工作分为四步：前向传播、损失计算、反向传播、优化器更新。关系如表：
> | 步骤 | 作用 | 结果 |
> |------|------|------|
> | **前向传播** | 模型做出预测 | 计算当前准确率 |
> | **损失计算** | 量化预测错误程度 | 得到损失函数值 |
> | **反向传播** | 计算每个参数的"错误贡献" | 获得损失函数对每个参数的梯度 |
> | **优化器更新** | 根据梯度调整参数 | 参数向正确方向移动 |
>
> 学习率、动量和权重衰减分别用于调整参数步长、避免陷入局部最优和避免过拟合。
> 书中对优化器工作方式的描述如下：
>>众所周知，优化器是根据权重的梯度作为指导，定义权重更新的力度，对权重进行更新。

`Optimizer`类是所有优化器的基类。其中：
+ 属性：
  - 参数组`param_groups`：参数组是用于管理需要进行优化的那些参数，是一个以字典为元素的列表。示例代码如下：
  ```python
    w1 = torch.randn(2, 2)
    w1.requires_grad = True

    w2 = torch.randn(2, 2)
    w2.requires_grad = True

    w3 = torch.randn(2, 2)
    w3.requires_grad = True

    # 一个参数组
    optimizer_1 = optim.SGD([w1, w3], lr=0.1)
    print('len(optimizer.param_groups): ', len(optimizer_1.param_groups))
    print(optimizer_1.param_groups, '\n')

    # 两个参数组
    optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},
                             {'params': w2, 'lr': 0.001}])
    print('len(optimizer.param_groups): ', len(optimizer_2.param_groups))
    print(optimizer_2.param_groups)
    ``` 
  - `state`是一些缓存值，如使用`momentum`时，需要保存之前的梯度，这些数据保存在`state`中。
  - `defaults`：优化方法默认的超参数；
+ 方法：
  - `zero_grad()`：[手动梯度置零](#代码片段-1)。
  - `step`：执行一次更新。
  - `add_param_group(param_group)`：添加参数组。
  - `state_dict()`：获取当前`state`属性，类似于存档。
  - `load_state_dict(state_dict)`：加载存档。

#### 随机梯度下降（SGD）：
SGD基于一个简单的常识：沿着最陡峭的下坡方向移动，下降最快。基本更新公式如下：
```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)
```
其中：
- $\theta$ ：模型参数
- $\eta$ ：学习率（步长）
- $\nabla_\theta J$ ：损失函数梯度

“随机”的意义是减少内存开销，提高效率。

SGD的使用主要有三步：**实例化、梯度清零、执行一步更新**。示例代码如下（仍以新冠肺炎分类示例为例）：
```python
    # step 3/4 : 优化模块
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) # 实例化
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    # step 4/4 : 迭代模块
    for epoch in range(100):
        # 训练集训练
        model.train()
        for data, labels in train_loader:
            # forward & backward
            outputs = model(data)
            optimizer.zero_grad()   # backward前进行梯度清零

            # loss 计算
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()   # 一步更新

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == labels).sum()
            acc = correct_num / labels.shape[0]
            print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))
```

> 此外，实践中比较常用的还有`Adam`。Adam的核心思想是自适应学习率。其更新参数的基本公式如下所示：
> ```math
> \begin{aligned}
>m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t & \text{（一阶动量）} \\
>v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 & \text{（二阶动量）} \\
>\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} & \text{（偏差校正）} \\
>\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} & \text{（偏差校正）} \\
>\theta_t &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} & \text{（参数更新）}
>\end{aligned}
> ```
> 其中，$ \beta $ 表示
> 与SGD相比，Adam收敛速度更快，但牺牲了部分精度。
### 学习率调整器
学习率是深度学习模型训练中的核心超参数，它定量控制了优化器在每次参数更新时的调整步长大小：具体而言，学习率决定了模型根据损失函数梯度调整参数的幅度。由于过高的学习率容易越过最优值，过低的学习率收敛速度则过慢，在优化模型的过程中，要经常调整学习率。

`lr_scheduler`模块和`optim`相似，是由`_LRScheduler`基类派生出不同的学习率调整器。

核心属性有`optimizer`、`base_lrs`和`last_epoch`。

`optimizer`是调整器所管理的优化器，优化器中所管理的参数组有对应的学习率，调整器要调整的内容就在那里。

`base_lrs`是基础学习率，来自于`optimizer`一开始设定的那个值。

`last_epoch`是记录迭代次数，通常用于计算下一轮学习率。注意，默认初始值是-1，因为`last_epoch`的管理逻辑是执行一次，自加1。

核心方法有`state_dict()`、`load_state_dict()`、`get_last_lr()`、`get_lr()`、`print_lr()`、`step()`。

`state_dict()`和`load_state_dict()`分别是获取调整器的状态数据与加载状态数据。

`get_last_lr()`、`get_lr()` 分别为获取上一次和当前的学习率。

`print_lr()`是打印学习率。

`step()`为更新学习率的接口函数，使用者调用`scheduler.step()`即完成一次更新。

学习率调整器的使用分为两步：

第一步：实例化。例如示例代码中，
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
```

第二步：合适的位置执行`step()`。

> #### 十四个学习率调整器的归纳总结：
> ##### 1. **LambdaLR（函数式调整）**
>- **原理**：根据自定义函数动态调整学习率：`lr = 初始lr × func(epoch)`
>- **特点**：完全灵活自定义，适合特殊调度需求
>- **适用场景**：需要复杂学习率曲线的研究性实验
>- **示例**：`func = lambda epoch: 0.95 ** epoch`
>
>##### 2. **MultiplicativeLR（乘法调整）**
>- **原理**：每轮学习率乘以给定函数返回值：`lr = lr × func(epoch)`
>- **与Lambda区别**：乘法因子直接作用于当前学习率而非初始值
>- **适用场景**：需要连续衰减的场景（如指数衰减变体）
>
>##### 3. **StepLR（阶梯衰减）**
>- **原理**：每`step_size`轮将学习率乘以`gamma`：`lr = lr × gamma`
>- **特点**：简单分段式下降，训练稳定
>- **适用场景**：基础训练任务（如ResNet标准配置）
>- **示例**：每30轮学习率减半（gamma=0.5）
>
>##### 4. **MultiStepLR（多阶段衰减）**
>- **原理**：当达到预设里程碑（milestones）时学习率乘以`gamma`
>- **与Step区别**：支持非均匀间隔调整
>- **适用场景**：复杂训练计划（如Transformer训练）
>- **示例**：在第[30,60,120]轮时学习率衰减
>
>##### 5. **ConstantLR（恒定衰减）**
>- **原理**：在达到`total_iters`轮前，学习率乘以恒定因子
>- **特点**：临时性小幅衰减
>- **适用场景**：训练初期的温和调整
>- **示例**：前10轮学习率降为80%，之后恢复
>
>##### 6. **LinearLR（线性衰减）**
>- **原理**：学习率从初始值线性衰减到目标值
>- **计算**：`lr = start_factor + (end_factor-start_factor)*(1-epoch/total_iters)`
>- **适用场景**：学习率预热（warm-up）阶段
>
>##### 7. **ExponentialLR（指数衰减）**
>- **原理**：每轮学习率乘以`gamma`：`lr = lr × gamma`
>- **与Step区别**：连续指数衰减而非阶梯式
>- **适用场景**：需要平滑衰减的任务（如RNN语言模型）
>
>##### 8. **CosineAnnealingLR（余弦退火）**
>- **原理**：按余弦函数周期性调整：`η = η_min + 0.5*(η_max-η_min)*(1+cos(T_cur/T_max))`
>- **特点**：模拟"加热-冷却"过程，避免局部最优
>- **适用场景**：图像分类等高精度任务
>
>##### 9. **ChainedScheduler（链式调度）**
>- **原理**：将多个调度器串联应用（前一输出作为后一输入）
>- **特点**：创建复杂调度策略
>- **示例**：`scheduler1 → scheduler2 → scheduler3`
>
>##### 10. **SequentialLR（顺序调度）**
>- **原理**：按里程碑顺序切换不同调度器
>- **与链式区别**：独立调度器序列而非嵌套
>- **适用场景**：多阶段训练（如预训练+微调）
>- **示例**：前100轮用余弦退火，100-200轮用阶梯衰减
>
>##### 11. **ReduceLROnPlateau（动态调整）**
>- **原理**：监控指标（如验证损失）停滞时自动降低学习率
>- **触发条件**：`factor`（衰减因子）、`patience`（等待轮次）
>- **适用场景**：不确定最佳衰减时机的任务
>- **优势**：自适应性强，减少人工干预
>
>##### 12. **CyclicLR（循环学习率）**
>- **原理**：在基值(base_lr)和最大值(max_lr)间三角循环
>- **特点**：周期性变化帮助逃离局部最优
>- **适用场景**：损失曲面复杂的任务（如GAN训练）
>
>##### 13. **OneCycleLR（单周期策略）**
>- **原理**：学习率先升后降的单周期变化
>- **过程**：前45%轮次线性上升，后55%余弦下降
>- **适用场景**：快速收敛需求（如Kaggle竞赛）
>- **优势**：Smith提出的高效策略
>
>##### 14. **CosineAnnealingWarmRestarts（带重启余弦退火）**
>- **原理**：余弦退火+周期性重启（SGDR算法）
>- **重启机制**：每`T_i`轮重置学习率为初始值
>- **作用**：跳出局部最优，寻找更优解
>- **适用场景**：提升模型最终精度（如分类任务）
>
> 运行示例代码，观察图像，不难直观感受到到不同学习率调整策略的适用场景。

### 小结
PyTorch优化模块通过损失函数量化误差、优化器驱动更新、学习率调整器动态调控三位一体的协作机制，使模型参数沿梯度方向持续优化，最终实现模型精度从随机猜测到接近完美的渐进式提升。掌握这些核心组件的原理与协同机制，是构建高效训练流程、释放模型性能潜力的关键所在。