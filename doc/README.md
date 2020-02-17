# CHROMOSOME分类
对于医生来说，有无血细胞影响不太大，重要的是其中的染色体是否能够看清楚，是否每个染色体都分开。
这个问题实际上是一个细粒度分类，而且粒度非常细。
医生人工查找准确度在90%左右。
暂时的目标是好类别的准确率在90%左右，召回率高于50%。

项目主要需要解决两个问题
1. 数据集中验证集和测试集分布有很大的差异
2. 负样本噪音很多，混杂了很多介于负样本和正正本之间的数据，甚至部分可以评价为正样本的数据也存在于负样本数据集中，不能单纯视作负样本。

## 线上模型更新流程
前置条件：新的模型已经训练好
过程：
1. 在46上将模型代码从/home/voyager/jpz/chromosome/models复制到/home/voyager/jpz/chromosome-server/core文件夹下
2. 更改46上的/home/voyager/jpz/chromosome-server/gpu-worker.py中的use_core变量的值，在CORE_ARGS中增加对应core的各项参数。
3. 停止tmux（jpz）中的celery worker，然后执行sh start-worker.sh重新启动
4. 在48上停止tmux（jpz）中的celery worker，然后执行sh start-worker.sh重新启动

## TODO List
0. 在预处理中添加将背景进行平衡和去除大的血细胞 ✔
1. 可视化数据 jpz/chromosome-server/visualize.ipynb ✔
2. 进行类别正确率分析，经过观察，发现验证集上采样或者下采样对准确率的影响不大，这个时候把混淆矩阵画出来看一下。 ✔
3. 添加细粒度分类的网络，进行细粒度的实验 ✔
4. 将新的数据添加进已有的数据集中 ✔
5. 将之前的模型在新的验证集上测试 ✔
    测完发现0类别的准确率和召回率都是0，基本上全部分类为了1这个类别
6. 将输出改成只有一个值 ✔
    1. 如果使用一个值，那么就没办法在训练的过程中查看perc，因为一个值的话，阈值是在不断变化。暂时使用0.5作为阈值查看效果
7. 自动eval一个文件夹下的所有pth文件 ✔
    1. logfile进行统一 ✔
    2. shell脚本 `exp/example/autoeval.sh` ✔
8. 如果得到的结果中两个类别是线性可分的话，那么通过卡阈值就可以得到结果。✔
    1. 使用regress模型，val的时候自动检测阈值 ✔ 
        使用regress模型自动卡阈值，最好的效果是acc 60左右，两个类别的的准确率和召回率都在60左右。
    2. 继续训练。继续训练发现效果并不好，acc反而会一直下降，是学习率的问题么？lr一开始都是0.0001后来降到了0.00001。✔
        从实验记录中很明显可以看出，是因为训练集优化的方向和验证集想要达到的效果不一样，所以导致了这个问题。
        所以目前的做的东西实际上是寄希望于训练的中间某个时刻学习到的权重能够区分验证集上的特征。
        从这个角度出发，大学习率和小学习率都要试一下才行。大模型和小模型都要试一下才行。或者换用adam来训练。
        1. res34 0.001作为初始lr进行训练 20epoch下降学习率，共30epoch ✔
            超过20个epoch的时候就0类别完全没有召回和准确率了
        2. res34 0.001作为初始lr进行训练 10epoch下降学习率，共30epoch
        3. res34 0.001作为初始lr进行训练 5epoch下降学习率，共20epoch
        4. res34 0.00001作为初始学习率，共训练30epoch查看效果，lr低于1e-5之后没什么意义了，所以再下降的实验就不做了
        5. res18 0.0001作为初始lr进行训练 20epoch下降学习率，共30epoch ✔
            整个训练的过程都不太行，整体准确率没有超过55%的，学习率下降之后0类别的召回率普遍在10%以下。
        6. res18 0.0001作为初始lr进行训练 10epoch下降学习率，共30epoch
        7. res18 0.0001作为初始lr进行训练 5epoch下降学习率，共20epoch
        8. res18 0.00001作为初始学习率，共训练30epoch查看效果
        9. res34 0.0001作为初始学习率，20epoch下降学习率，使用adam作为optimizer
    3. 在数据上下功夫，能不能通过一些办法把两个类型的数据给归一化了。 ✔
        automask将染色体筛选出来，autolevel会改变对比度，貌似没有进一步的处理方法了。
9. 给bilinear模型也弄一个阈值，两个都卡阈值来得到结果
10. 查看bilinear模型的效果 ✔
    1. 效果同样不好，和regress的表现一样，当训练epoch变长，尤其是学习率进一步下降之后，在val的时候基本上全部预测为1了，没有0.
        很明显是对训练集过拟合了。解决过拟合的办法：
        1. 提前终止训练 
        2. 减小学习率 
        3. 换用小模型 其中之前使用bi18训练了一个模型，发现这个模型训练到后期也会出现这样的情况，所以还是需要训练的epoch少一点，然后可以换用optimizer来学习。
            个人觉得res34就可以完成这个任务了，或者使用更小的res18.
11. 添加adam作为optimizer
12. 尝试PU Learning
    1. 关于相关知识
        1. 知乎原答案
            [只有正样本和未标签数据的机器学习怎么做？](https://www.zhihu.com/question/286851129/answer/454902118)
            使用了1000个弱分类器作为基地，每个分类器采样全部的正样本和一小部分未标注样本（数量和负样本相同），训练完再对未采样到的未标注样本进行打标签，这1000个弱分类器得到的结果进行平均，效果非常好。
        2. PU Learning综述，知乎答案上提到的
            [A Survey on Postive and Unlabelled Learning](https://www.eecis.udel.edu/~vijay/fall13/snlp/lit-survey/PositiveLearning.pdf)
        3. ICLR2019的一篇PU Learning文章
            考虑到了得到的正样本和未标注样本之间是分布具有差异的，比如人脸识别，正样本效果都是很好的，而未标注样本中可能有各式各样奇怪的数据。
            [Learning from Positive and Unlabeled Data with a Selection Bias ](https://openreview.net/forum?id=rJzLciCqKm)
        4. oracle的一篇文章，写的是具体的解决PULearning问题的方法 值得看
            [4 Approaches to Overcoming Label Bias in Positive and Unlabeled Learning](https://blogs.oracle.com/datascience/4-approaches-to-overcoming-label-bias-in-positive-and-unlabeled-learning)
        5. stackoverflow，关于PULearning背景的解释
            [Why positive-unlabeled learning?](https://datascience.stackexchange.com/questions/26751/why-positive-unlabeled-learning)
        6. 由ICLR19那篇文章引申出来的文章,ICLR19那篇在处理神经网络的时候实际上就是使用了这篇定义的loss函数
            [nnPULearning](https://github.com/kiryor/nnPUlearning)
    2. 尝试 这个方法大概不太行，训练集都不能够拟合
        1. 简单尝试降低未标注样本中的训练时的梯度。原理实际上是强化以标注正样本的梯度，削弱未标注样本中正样本梯度，让关于正样本的数据的优化方向朝着正常前进，于此同时负样本的梯度的强度也需要调整至和正样本差不多。
        2. 使用puloss进行训练试一下
            1. 先写出来numpy版本 ✔
            2. 迁移到pytorch上，在未标注的样本中大概正样本有2/5的样子。✔
            3. 进行训练 ✔
                1. dataset改成label为1 -1的样子 ✔
                2. 将main中训练eval那里看是不是会有问题 ✔
                3. 在训练的过程中，positive_part_loss会急速下降，然后negative_part_loss一直保持在0.6, 因为所有数据输出的值都小于0很多。
                    pred_positive基本上等于0，pred_unlabeled基本上为64导致最终的loss基本上都为negative_part_loss数值为0.6左右。
                    这个问题是positive部分的loss迅速被优化降低到0，然后因为negative_part_loss那个部分写错了没有乘pred_negative所以negative部分的数据没办法得到优化。已经解决 ✔
                4. 3问题解决之后发现，negative_part_loss迅速下降到0，然后positive_part_loss的数值为prior。 ✔
                    但是同时也出现了positive_part_loss迅速下降为0，然后negative_part_loss持续为prior的情况。
                    看来模型初始的输出值要么是特别大，大到positive loss为0，要么特别小，小到negative loss为0，那么这就是初始化的问题了。
                    初始化使用的是Kaiming normal，将网络中的参数初始化在(0, 2/d)也就是均值为0，std为sqrt(2/d)。这应该是没什么问题的.
                    也有可能是学习率太小了，陷到局部最优了，输出的值一直很大。
                    当positive_part_loss一直为0.5的时候 output的均值一般在-200左右徘徊，没有明显的趋势。 5次执行大概有1次这样的情况。
                    当negative_part_loss一直为0.5的时候 output从800-300都有波动，但是一直降不下来。     5次执行大概有4次出现这样的情况。
                    应该就是学习率太小了，上述两种情况都为局部最优，初始化到了那个地方，从此可以看出这个loss函数的landscape大概是什么样子的。
                    换用0.001的学习率试一下，output为正数的时候反而数值越来越大，范围直接到了1800的水平，output为负数的时候数值越来越小，到了-300的水平
                    换用0.00001的学习率试一下，output为正数的时候范围降低到了80-120，但是仍然不收敛，感觉这个loss有问题...
                    怀疑是网络结构的问题，在网络的最后直接连了两个conv，没有激活和归一化，加上bn relu试一试
                        output均值变为了70左右，实际上这个问题是梯度弥散，就是因为使用了sigmoid。
                    loss_func的本意是：
                        假如全部是正样本那么预测为正样本的损失
                        假如全部是负样本那么预测为负样本的损失
                        所以改一下，使用l1loss或者其他的改一下就行。
        3. 使用regress34效果还不如直接进行回归，预测的时候正类的召回和准确率基本上在0左右。 ✔
        4. pul使用res18进行训练。
        5. 尝试res34，SmoothL1Loss
13. 部署之前的那个60%的模型 ✔
14. 将目前已经有的实验都记录下来
15. 验证集清理
    训练一个autoencoder，然后对验证集进行清理试一下.
    1. 需要把数据集的形式改为读取list文件，然后进行迭代
16. 可视化验证集上的数字结果 jpz/exp/regress34_bs64_automask_pul_lr0.0001/visualize_val_result.ipynb ✔
17. 再标注2000张图片，然后训练一个模型来从训练集中筛选出来一批数据让医生标注进行迭代。 不搞了，目前先把一个模型交了再说
    数据到了，先统计数据 ✔
    生成list，实现对应的Dataset类 ✔
    训练查看效果 训练了50个epoch ACC有 64%
    一遍直接查看epoch的效果，另一边使用更大的模型、更长的学习率周期来搞
    1. 使用resnet101 分类 30epoch 50epoch下降 已经在训练了
    2. 分析之前模型的效果，先查看输出的val txt中的分布
        训练出来的这个模型是为了挑出来最差的染色体图片，所以只关心差那个类别的准确率和召回率，在util.py中实现了 ✔
        在之前的训练集上过一遍，查看有多少会被筛选为最差的类。 
        在训练集70375条数据中，有49547条评价为很差，有17874条评价为普通差。
    3. 新训练的结果出来了 ✔
        整体最好的效果是75%，训练到epoch44
        对于最坏的类别，效果好的，准确率有92% 召回有50%左右，即epoch0023那个pth
        对于最好的类别，效果好的，准确率有81% 召回70%左右，及epoch0037那个pth
    4. 可以把两个模型embed起来，直接在验证集上试一下，策略改一下 ×
        效果不行要么全部预测为坏类，要么好类的准确率太低, 在验证集上面，整体acc=0.8865 1类别的perc和recall有0.9 而0类别的perc只有0.16左右
    5. 将坏类别效果好的那个模型在训练集上过一遍，得到结果然后使用这个结果和之前的训练集中好的部分重新进行训练。
18. 将之前的 效果好的，准确率有81% 召回70%左右效果好的，准确率有81% 召回70%左右 的这个模型调一下阈值，看是不是能够直接得到90% 50%的模型。不行就调中间的模型。
    之前的数据集如果不要的话，可以直接把验证集拿过来使用，验证集也是全部灰色背景的。 ✔
19. 在第三版数据集上进行训练。
    exp文件夹单独搞一个third文件夹，然后在这个文件夹下进行实验文件夹的控制。
    先训练一个强行回归的模型再说。 似乎是真的不行
    直接在测试集下采样即可啊
    看l1 和l2有什么区别
    l2会直接梯度爆炸
20. sigmoid focal loss ✔
    先调整让loss稳定再说
    sigmoid focal loss有希望。直接进行回归是有问题的。
21. 这只是分类，那么检测和分割会不会出现类似的问题？
    这个问题的描述是？
    训练数据标注有缺陷、数据量少（使用gan生成数据）、实际上是个一分类问题（只检测一个类别、只分割一个类别）、并且该类别和其他剩余类别很容易混淆（大量相似目标、分割目标混在一起）

22. 部署那个准确率为75%的模型。✔
    /home/voyager/jpz/chromosome/exp/third_dataset/res34_bs64_automask_epoch50_lr0.0001/checkpoint/epoch_0045_acc_81.929348.pth

## 数据集组织
训练集 /home/voyager/data/chromosome/train/0 or /1
验证集 /home/voyager/data/chromosome/val/1 or /0

**第一版数据集**
时间：20191115之前
训练集0：4569  好类别 路径：`/home/voyager/data/chromosome/train/0`
训练集1：70375 坏类别 `/home/voyager/data/chromosome/train/1`
验证集0：508  `/home/voyager/data/chromosome/val-v1/0` 6.1% 这样非常不合理,要么将0进行上采样，要么将1进行下采样
验证集1：7819  `/home/voyager/data/chromosome/val-v1/1`
共 83271

**第二版数据**
时间：20191118
备注：验证集更新了数据，变为了全部灰色图片。
训练集0：4569  好类别 `/home/voyager/data/chromosome/train/0`
训练集1：70375 坏类别 `/home/voyager/data/chromosome/train/1`
验证集0：50 `/home/voyager/data/chromosome/val/0` 3.8% 需要进行上采样
验证集1：1192 `/home/voyager/data/chromosome/val/1`

在训练中对训练集0进行了上采样，这样训练集一共有70375*2=140750条数据，batchsize=64的时候，一个epoch大概需要2199个batch。

现在数据有一个很重要的问题，坏类别往往都是灰色背景的，且坏类别中存在一定数量的可以被评价为好类别的数据。数据是偏态分布的且数据集的两个类别不完全可分。

**额外版的数据集**
需求提出时间：20191127
数据收到时间：20191202
路径： /home/voyager/data/chromosome/20191127
训练list：/home/voyager/data/chromosome/20191127/list
对应Dataset：ExtDataset
备注：和之前的验证集进行合并，训练一个小模型
    20191202给出的80分以上100分以下共有293张，100分的共有48张，其余的不能使用的1660张，共2001张。
    合并之后设置三个类别0 1 2，其中0代表好的，1代表一般的，2代表差的，0类共有48张 1类有501张，2类有2452张
训练集0：34
训练集1：351
训练集2：1717
验证集0：14
验证集1：150
验证集2：735

**第三版数据集**
将第二版的验证集（背景全部为灰色）中的0类别拿过来，和额外版的数据集进行合并。
路径：/home/voyager/data/chromosome/20191204
训练list: /home/voyager/data/chromosome/20191204/list
对应Dataset 直接使用SimpleDataset即可
训练集0：69
训练集1：351
训练集2：1717  训练集共 5120
验证集0：29
验证集1：150
验证集2：735

**测试数据集**
时间：20191223
路径：/home/voyager/data/chromosome/20191223
list：/home/voyager/data/chromosome/20191223/list/all.txt
0 1 2类别各有20张图片。

**测试数据集**
时间：20190108
路径：/home/voyager/data/chromosome/20200108
list:/home/voyager/data/chromosome/20200108/list/all.txt
0类别14张
1类别341张

### 阈值策略
见`utils/misc.py`中的`focal_finetune_threshold`函数，使用方式如下
```py
strategy = 'strategy5_focal'
focal_result = '/home/voyager/jpz/chromosome/exp/third_dataset/res34_test/focal_val_result_0046_epoch.txt'
focal_finetune_threshold(focal_result, strategy)
```
其中focal_result的格式为'gt, pred0, pred1, pred2, path\n'

当需要新的策略进行测试的时候需要编写新的strategy函数。strategy函数根据每条数据的0、1、2类别和阈值决定最终预测出什么类别。然后focal_finetune_threshold计算召回率和准确率。

### Embed模型
将两个模型在验证集上的结果给保存下来，然后读取结果选择规则进行

**embed ext中的两个res101模型**
模型一
    描述：对于最坏的类别，效果好的，准确率有92% 召回有50%左右，即epoch0023那个pth
    path路径：/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/checkpoint/epoch_0023_acc_66.722308.pth 
    trainset-eval：/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/trainset_0024.txt
模型二
    描述：对于最好的类别，效果好的，准确率有81% 召回70%左右，及epoch0037那个pth
    path路径：/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/checkpoint/epoch_0037_acc_71.444099.pth 
    main_val_set-eval：/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/main_val_set_0037.txt



## 模型代码
所有模型文件都在`models/`下，一个模型一个文件。在`models/__init__.py`中控制`main.py`可以访问的模型。

**oldresnet.py**
此为项目最一开始使用的resnet，并非标准的resnet

**regress_resnet.py**
将最后的结果输出为1，变成回归的模型。

**bilinear.py**
使用resnet作为base model的双线性模型。

**resnet.py**
标准的resnet模型。

## Loss函数

## 测试数据测试流程
1. 在/home/voyager/jpz/chromosome/exp/ 下新建一个实验文件夹
2. 复制一份test.sh过来，更改其中的list列表
3. `conda activate pytorch_0.4.1` 然后执行`sh test.sh`

## 运行速度
使用prefetcher之后可以加速读取数据的速度。
如果数据已经prefetch了，那么一个batch的耗时如下
```
Data Prepared Time:3.9ms, 	# 转移到cuda上
Net Process Time:0.08ms, 	# 神经网络处理
Backward Time:3.2ms, 	    # 反传
Optim Time:534.1ms, 	    # 优化参数
Prefetch Time:13.4ms, 	    # fetch下一个batch，某些时候fetch时间会特别大有1.7秒，这个时候gpu就完全是空着的
Batch Time:554.8ms          # 该batch总共的耗时
```

**单个训练速度记录**
res34，bs64，一个batch大概要花费5秒的时间，一个epoch大概半个小时。
regress34，bs64，50个batch大概要花费65秒，一个epoch大概48分钟。
bi18，bs64，50个batch大概要60秒，一个epoch大概44分钟。

当两个卡同时运行时会因为IO的问题，导致两个卡上的模型速度都会变慢。

## 代码使用
**实验文件夹**
`./exp/`下为每个实验文件夹，每个实验文件夹下包括
 - `train.sh` 执行`sh train.sh`开始训练
 - `autoeval.sh` 自动对一个checkpoint目录下的所有pth文件进行eval。
 - `eval-time.log` 执行`autoeval.sh`得到的log
 - `checkpoint/` 保存训练好的模型
 - `time.log` 输出的log
 - `tfevent` tensorboard文件

**命令行参数**
项目的入口是`main.py`，项目运行时接受命令行参数和`main.py`内部配置的参数。
命令行参数列表
 - `--lr` 初始学习率
 - `--end_epoch` 总的训练epoch
 - `--batch_size` batchsize
 - `--checkpoint` 读取之前保存的模型参数
 - `--resume` 是否接着之前的epoch继续训练，需要指定`checkpoint`
 - `--root` 数据集的根路径，这个路径下应该有`train` `val`这两个文件夹
 - `--device` 指定GPU或者CPU，但实际上是通过`train.sh`或者`autoeval.sh`中的`CUDA_VISIBLE_DEVICES=1`来指定的
 - `--eval_freq` 每几个train epoch进行eval，默认为1
 - `--exp_path` 实验文件夹路径，保存checkpoint的时候用到
 - `--model` 使用的模型，在代码中通过反射来得到模型的实例，所有模型都必须在`models/__init__.py`中导入
 - `--eval` 是否只对模型在验证集上检验效果
 - `--log_file` log文件的路径，代码使用logger来记录log
 - `--log_freq` 在训练的时候每多少个batch输出log
 - `--optim` 优化器，目前还只支持SGD
 - `--lr_milestones` SGD的学习率下降节点
 - `--preprocess` 预处理方法可以使用automask或者autolevel
 - `--save_val_result` 将验证集上的结果保存下来，格式为`gt, pred\n`
 - `--loss` 使用的loss函数，支持PULoss和pytorch自带的loss
 - `--dataset` 分类和普通的回归使用SimpleDataset，如果PULearning则使用PUDataset

**保存模型的输出**
在参数中指定`--save_val_result`的路径，将结果保存下来格式为`gt, pred\n`。当模型名字中有regress字段时会调用`utils.regress_get_result()`，如果没有则会调用`utils.get_result()`

**focal loss**
使用focal loss的时候在`train.sh`中指定`--loss FocalLoss`即可，在训练的过程中会调用`utils.get_result()`来进行val并输出混淆矩阵，但是同时保存下来的`focal_val_result_{:04d}_epoch.txt`是调用了`utils.focal_get_result()`和`utils.save_val_result()`的，会保存下来`gt, pred0, pred1, pred2, path\n`。

## 实验记录
这种量级的数据使用一个1080应该差不多，先看看能装下多少张图片。
这种二分类使用res34大概就可以了吧，需要对0类别进行上采样。
使用res34 batshzie=64一个gpu刚好占满。
使用res101 bs=16一个gpu差不多占满
进行eval的时候res101，使用bs64只占了一半显存。
模型    上采样  preprocess  epoch  bs    实验文件夹                             学习率    Loss      Acc     0-recall   0-precision    val集   th   optimizer
res101    Y     autolevel    78    -           -                                -      0.1982    93.121      -           -         val-v1   -       SGD
res101    Y     autolevel    78    -           -                                -      1.5086    50.206     0.0         0.0          val    -       SGD
res34     Y     automask      1    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   2.0321    50.411     0.0         0.0          val    -       SGD
res34     Y     automask      5    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   1.7638    50.740     0.0         0.0          val    -       SGD
res34     Y     automask      9    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   2.1895    50.863     0.0         0.0          val    -       SGD
res34     Y     automask     15    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   2.3123    50.822     0.0         0.0          val    -       SGD
res34     Y     automask     20    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   2.1598    50.863     0.0         0.0          val    -       SGD
res34     Y     automask     27    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   1.9785    50.863     0.0         0.0          val    -       SGD
res34     Y     automask     36    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   1.8815    50.699     0.0         0.0          val    -       SGD
res34     Y     automask     41    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   2.0565    50.740     0.0         0.0          val    -       SGD
res34     Y     automask     47    64   res34_bs64_epoch50_automask_lr0.0001  0.0001   2.0540    50.699     0.0         0.0          val    -       SGD
regress34 Y     automask     13    64   regress34_bs64_automask_lr0.0001      0.0001   0.3991    62.668    63.926      61.501        val    0.8     SGD
regress34 Y     automask     14    64   regress34_bs64_automask_lr0.0001      0.0001   0.4902    50.000     0.0         0.0          val    0.1     SGD
regress34 Y     automask     15    64   regress34_bs64_automask_lr0.0001      0.0001   0.4778    52.643     11.997     59.583        val    0.1     SGD
regress34 Y     automask     17    64   regress34_bs64_automask_lr0.0001      0.0001   0.5164    51.342     4.027      75.000        val    0.9     SGD