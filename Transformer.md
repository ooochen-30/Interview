# Transformer篇

**[Attention is All You Need](https://arxiv.org/pdf/1706.03762)**

**1. 什么是 Transformer？介绍一下 Transformer 的结构**

Transformer是Google提出的**完全基于自注意力机制**的序列建模模型，**抛弃了传统的RNN/CNN**，依靠Self-Attention实现**并行计算和长距离依赖**建模。是现在所有大模型的基础架构。

Transformer的**整体结构**分为**Encoder**(编码器)和**Decoder**(解码器)两大部分。两者都由多层堆叠，编码器用于理解输入序列，解码器用于生成输出序列。

整体流程：

1. 将输入进行Tokenizer分词并转为词向量和加上位置编码
2. 交给Encoder层层提取特征，理解输入的含义
3. Encoder处理完的结果给到Decoder，它一边看Encoder理解好的内容，一边预测下一个词生成输出
4. 通过全连接层和Softmax，输出每个位置最可能的词

**2. 具体介绍一下各个组成部分？**

![1772071473951](src/1772071473951.png)

Encoder包含两个子层：

1. 多头自注意力层（双向注意力，看到**整个输入序列**）
2. 前馈网络FFN

两个子层都使用了残差连接与LayerNorm

Decoder包含三个子层：

1. 带掩码的多头自注意力（只能看到**当前及之前的token**）
2. 编码器-解码器注意力（Q来自Decoder，K、V来自Encoder **生成时关注输入序列**）
3. 前馈网络FFN

三个子层都使用了残差连接和LayerNorm

总体来说Transformer的核心组件包括Self-Attention, Positional Encoding, FFN和Normalization

**3. 介绍一下常见的Transformer衍生架构**

+ **Encoder-only**：基于**双向自注意力**，充分捕捉上下文语义，适合需要“理解输入”的任务。代表模型是`BERT, RoBERTa, ALBERT`。典型任务是文本分类、情感分析等
+ **Decoder-only**：基于**单向掩码自注意力**，天然适配“自回归生成”，适合需要“生成输出序列的任务”。代表模型是`GPT-1/2/3, LLaMA, Qwen`。典型任务是文本生成、机器翻译等
+ **Encoder-Decoder**：Encoder做源序列理解，Decoder做目标序列生成，通过**编码器-解码器注意力**建立源-目标序列的关联，跨序列的语义映射能力强。适合源和目标为不同序列的转换任务。代表模型是`T5, BART`等。典型任务是机器翻译、文本摘要、语音识别等

**4. Transformer对比RNN/CNN架构的优缺点是什么？**

Transformer的核心优势是**长距离依赖建模**、**训练效率高，天然并行**、**特征表示更灵活**。（这里也可以问做**为什么用Self-Attention?**）

+ **长距离依赖**：RNN是串行计算，词与词的依赖会随序列变长而衰减，LSTM/GRU只能缓解；CNN靠卷积核建模依赖，核大小有限，长距离依赖需要多层堆叠效果差；**Self-Attention直接计算任意两个词的关联，距离不影响建模能力**
+ **训练效率高、天然并行**：RNN必须按序列顺序计算完全无法并行；CNN能局部并行，但长序列仍需逐层滑动；**Transformer对所有token的计算是同时进行**，无时序依赖，可充分利用GPU算力
+ **特征表示更灵活**：RNN/CNN的特征提取是固定路径；Self-Attention的**注意力权重是动态学习的**，不同句子、不同语境下，词与词的关联权重会变化，能适配更复杂的语义。

Transfomer的核心缺点是**短序列/局部结构建模不够直观**、**长序列计算成本高**、**无固有时序信息，依赖额外位置编码**。

+ **短序列/局部结构建模不够直观**：CNN靠卷积核天然捕捉局部相邻特征；RNN也按时序关注局部；Transformer是全局注意力，对局部特征的捕捉需要靠模型学习，初期训练成本较高
+ **长序列计算成本高**：Self-Attention计算n个token之间两两之间的注意力，计算量和显存占用随序列长度平方级增长$O(n^2)$， RNN是$O(n)$，CNN是$O(n* k)（k为卷积核大小）$

+ **无固有时序信息、依赖额外位置编码**：RNN/CNN结构自带时序（RNN按顺序，CNN按滑动窗口）；Transformer必须手动加位置编码才能区分token顺序。

![1772074622048](src/1772074622048.png)

**5. 介绍一下Attention机制的原理？**

Attention机制就是模拟人类**聚焦关键信息**的能力，核心是先定义Q查询、K键、V值三个向量，以Q为核心计算和所有K的相似度，把相似度归一化得到注意力权重，最后用权重对V加权求和，得到了聚焦于关键信息的输出；他的优势是能动态分配权重，比池化更精准捕捉语义关联。

**6. Self-Attention为什么需要缩放点积（进行开根号）？**

Self-Attention的公式为 $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$，缩放点积是为了防止**点积过大**直接送入softmax结果趋向于0或1，导致**softmax的导数几乎为0**，产生梯度消失现象，模型无法训练。

**7. 为什么Self-Attention的实现中，softmax的维度是dim=-1?**

因为在权重矩阵中，**每一行代表一个Query对所有Key的注意力权重分布**，我们要确保对于每个特定的Query，它对所有输入token的**权重之和为1**

**8.  自注意力机制的计算复杂度怎么算 结合公式讲一下**

自注意力的整体时间复杂度为$O(n^2*d + n* d^2)$ 其中n是序列长度， d是token的特征维度 核心耗时为$O(n^2*d)$，也是自注意力长序列计算成本高的根本原因。Self-Attention的公式是$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$，计算的

**9. 介绍一下Multi-Head Attention? 他为什么有效？**

Multi-Head Attention是对原始Self-Attention的扩展升级，它把QKV拆分成多个头，没个头单独计算自注意力，最后把所有头的结果拼接、线性投影得到最终输出。Multi-Head Attention有效的原因主要有3个：首先多头能学习到不同的注意力模式，特征更全面；第二，通过拆分多头降低单头注意力的维度压力，降低计算复杂度的同时提升效率和精准度；最后单头注意力容易出现偏见，比如过度关注某个位置的token或只捕捉到某一种关联，多头注意力的投票式整合使得模型对不同语境、序列的适配性更强，鲁棒性更高。

**10. 多头注意力的头数为什么不能太多/太少？**

+ 太少：回到单头的问题，无法捕捉多重依赖，失去多头的意义；
+ 太多：每个头的维度过小，模型学习能力不足（比如维度太小，无法承载足够的信息），且计算量会线性增加，性价比降低；

**11. 你觉得多头注意力能提高计算效率吗，结合公式推导一下？如果不能，详细讲讲为什么？**

![1772081619334](src/1772081619334.png)

在模型总维度$dim$固定的前提下，多头注意力和单头注意力的计算复杂度完全一致，并不会直接提升计算效率，但能在不增加计算成本的前提下，大幅提升模型的特征建模能力。

自注意力的计算复杂度主要来自$QK^T$的矩阵乘法，Q/K的维度为[batch,seq,dim] 单头注意力的计算复杂度为$O(n^2 * dim)$

多头注意力的核心是**总维度拆分，分头计算，维度不变** $dim = h * d_k$, 此时的单头注意力复杂度为$O(n^2 * d_k)$ ，多头注意力的总计算复杂度为$O(n^2 * dim)$

从算法复杂度上多头和单头一致，但从**工程实现（GPU/硬件运行效率）**上多头能通过并行计算提升实际运行速度：

+ 单头是**单路高维计算**，GPU的并行算力无法充分利用（高维矩阵乘法的并行度受限）
+ 多头是**多路低维并行计算**，h个头的低维矩阵乘法能同时在GPU的多个计算核心上运行，实际训练/推理的耗时会比单头略低

但头数超过硬件并行算力上限时，多路并行会变成多路串行，且过多的投会带来拼接/线性投影的额外开销，并导致单个头的学习能力不足，模型效果下降。

**12. Transformer的FFN层能不能去掉，为什么**

FFN不能去掉，他是和自注意力层互补的核心组件。

+ FFN的结构是Linear => Activation => Linear，自注意力层是线性层，FFN通过激活函数引入非线性，去掉之后模型无法拟合复杂的自然语言；
+ 自注意力负责建模词之间的全局关联，FFN通过升维降维对单个词的特征进行提取
+ FFN是单token并行计算，与注意力层的高复杂度形成计算平衡，还能通过残差连接提升训练稳定性。

**手写一下SelfAttention**

```python
import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__() # 不要遗漏
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.k_proj = nn.Linear(hidden_dim,hidden_dim)
        self.v_proj = nn.Linear(hidden_dim,hidden_dim)
    
    def forward(self, X):
        # X.shape [batch,seq,dim]
    	Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        attention_weight = torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(self.hidden_dim)
        attention_weight = torch.softmax(
        	attention_weight,
            dim = -1 # 重点
        )
        attention_value = attention_weight @ V
        return attention_value
    
# 进阶实现 padding mask/dropout
class SelfAttention(nn.Module):
    def __init__(self,dim,dropout_rate=0.1):
        super().__init__() # 不要遗漏
        self.dim = dim
        self.dropout = nn.Dropout(dropout_rate)
        
        self.q_proj = nn.Linear(dim,dim)
        self.k_proj = nn.Linear(dim,dim)
        self.v_proj = nn.Linear(dim,dim)
    
    def forward(self,x,mask=None):
        # x.shape [batch,seq,dim]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # attention_weight.shape [batch,seq,seq]
        attention_weight = torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(self.dim)
        if mask is not None: # 先mask再softmax
            attention_weight = attention_weight.masked_fill(
            	mask == 0,
                float('-inf') # softmax之后为0，直接填0那么softmax之后依然有权重分值
            )
        
        attention_weight = torch.softmax(
        	attention_weight,
            dim = -1 # 重点
        )
        
        attention_weight = self.dropout(attention_weight)
        output = attention_weight @ V
        return output
```



**手写一下Multi-Head Attention**

```python

```

