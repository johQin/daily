# Supervised Fine Tuning

监督微调 （Supervised Fine Tuning，SFT）是指在预训练模型的基础上，利用标注数据进一步训练，以增强模型在特定任务上的能力。

分类：

1. 全参数微调：对预训练所有参数进行微调
2. 部分参数微调：仅微调预训练模型的部分参数，eg: LayerNorm Tuning等
3. 并联低秩微调：LoRA，AdaLoRA，QLoRA，PiSSA，OLoRA，LoHa，DoRA等
4. Adapter Tuning：在模型中间层插入瓶颈状的Adapter模块
5. 基于Prompt的微调：Prefix-Tuning，Prompt Tuning，P-Tuning，P-Tuning v2

<img src="./legend/sft-part.jpg" alt="全量与部分" style="zoom: 25%;" />

<img src="./legend/sft-lora.png" alt="lora" style="zoom:30%;" />

<img src="./legend/sft-lora-detail.png" style="zoom:33%;" />

<img src="./legend/sft-adapter.png" style="zoom: 50%;" />

<img src="./legend/sft-prompt.png" style="zoom: 50%;" />
