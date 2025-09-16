from contextlib import nullcontext
import torch
from transformers import AutoTokenizer

from model.model_big_strong import BigStrongConfig, BigStrongForCausalLLM


class TextGenerator:
    def __init__(
        self,
        checkpoint: str,  # 模型检查点路径
        tokenizer_model_path="../model/",  # 分词器模型路径
        seed=42,  # 随机种子，确保可重复性
        device=None,  # 设备，优先使用 CUDA，如果没有可用的 CUDA，则使用 CPU
    ):  
        """
        初始化 TextGenerator 类，加载模型、设置设备和分词器等。
        """
        # 模型加载配置
        self.checkpoint = checkpoint  # 保存的模型检查点路径
        self.tokenizer_model_path = tokenizer_model_path  # 分词器模型文件路径
        self.seed = seed  # 随机数种子，用于生成的可重复性
        self.device = device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # 根据硬件条件选择设备
        self.device_type = (
            "cuda" if "cuda" in self.device else "cpu"
        )  # 判断当前设备是否为 CUDA

        # 设置随机种子，确保生成的可重复性
        torch.manual_seed(seed)  # 设置 CPU 随机种子
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.backends.cuda.matmul.allow_tf32 = (
            True  # 允许 CUDA 使用 TF32 精度进行矩阵乘法运算
        )
        torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32 精度加速

        # 默认使用 bfloat16 精度
        ptdtype = torch.bfloat16
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        )

        # 加载模型检查点文件
        checkpoint_dict = torch.load(
            self.checkpoint, map_location=self.device
        )  # 加载模型参数 # 初始化模型参数
        self.model = BigStrongForCausalLLM(BigStrongConfig(hidden_size=512, num_hidden_layers=8))  # 实例化 Transformer 模型
        sunwanted_prefix = "_orig_mod."
        for k, v in list(checkpoint_dict.items()):
            if k.startswith(sunwanted_prefix):
                checkpoint_dict[k[len(sunwanted_prefix) :]] = checkpoint_dict.pop(k)
        self.model.load_state_dict(checkpoint_dict, strict=False)

        # 计算模型参数量
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.3f} M parameters.")
        # 设置模型为评估模式（evaluation mode），防止训练模式下的 dropout 等操作影响结果
        self.model.eval()
        # 将模型放置到正确的设备上（GPU 或 CPU）
        self.model.to(self.device)
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_model_path
        )  # 根据指定的路径加载分词器

    def chat_template(self, prompt):
        message = [
            {"role": "system", "content": "你是一个AI助手，你的名字叫BigStrongGPT。"},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

    def generate_samples(
        self,
        start="Hello!",  # 生成文本的起始提示词
        num_samples=3,  # 生成样本的数量
        max_new_tokens=None,  # 每个样本生成的最大token数（None时使用模式默认值）
        temperature=None,  # 控制生成随机性（None时使用模式默认值）
        top_k=None,  # 限制生成的token范围（None时使用模式默认值）
        use_chat_template=False  # 是否使用聊天模板处理起始文本
    ):
        """
        根据给定的起始文本生成样本，支持SFT和预训练两种模式。

        :param start: 生成文本的起始提示词
        :param num_samples: 要生成的文本样本数
        :param max_new_tokens: 每个样本生成的最大token数，默认值：SFT模式256，预训练模式100
        :param temperature: 控制生成的随机性，默认值：SFT模式0.7，预训练模式0.8
        :param top_k: 限制生成时选择的token范围，默认值：SFT模式300，预训练模式50
        :param use_chat_template: 是否对起始文本应用chat_template处理（SFT模式常用）
        :return: 生成的文本样本列表
        """
        # 设置不同模式的默认参数
        if use_chat_template:
            # SFT模式默认参数
            default_max_tokens = 256
            default_temp = 0.7
            default_top_k = 300
            # 应用聊天模板处理起始文本
            processed_start = self.chat_template(start)
        else:
            # 预训练模式默认参数
            default_max_tokens = 100
            default_temp = 0.8
            default_top_k = 50
            processed_start = start  # 不处理起始文本

        # 确定最终参数（用户指定值优先，否则使用模式默认值）
        final_max_tokens = max_new_tokens if max_new_tokens is not None else default_max_tokens
        final_temp = temperature if temperature is not None else default_temp
        final_top_k = top_k if top_k is not None else default_top_k

        # 编码起始文本
        start_ids = self.tokenizer(processed_start).data["input_ids"]
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        generated_texts = []
        with torch.no_grad(), self.ctx:
            for _ in range(num_samples):
                # 生成文本
                generate_kwargs = {
                    "input_ids": x,
                    "max_new_tokens": final_max_tokens,
                    "temperature": final_temp,
                    "top_k": final_top_k
                }
                # 如果是SFT模式，添加eos_token_id参数
                if use_chat_template:
                    generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

                y = self.model.generate(**generate_kwargs)
                generated_texts.append(self.tokenizer.decode(y[0].tolist()))

        return generated_texts


if __name__ == "__main__":
    print("------------------- Pretrain Sample ------------------- \n")
    pretrain_prompt_datas = [
        "你好",
        "阿里巴巴",
    ]

    generator = TextGenerator(checkpoint="../output/pretrain_output/pretrain_512.pth")
    for i in range(len(pretrain_prompt_datas)):
        samples = generator.generate_samples(
            start=pretrain_prompt_datas[i],  # 自定义起始文本
            num_samples=5,   # 生成5个样本
            max_new_tokens=300,  # 最大长度300
            temperature=0.9,  # 更高的随机性
            use_chat_template=False
        )
        print(f"\nSample {i + 1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n{'-' * 20}")  # 打印生成的样本并用分隔线分割

    print("\n ------------------- SFT Sample ------------------- \n")
    sft_prompt_datas = [
        '你好',
        "阿里巴巴",
    ]
    generator = TextGenerator(checkpoint='../output/sft_output/full_sft_512.pth')  # 初始化生成器
    for i in range(len(sft_prompt_datas)):
        samples = generator.generate_samples(
            start=sft_prompt_datas[i],  # 自定义起始文本
            num_samples=5,   # 生成5个样本
            max_new_tokens=300,  # 最大长度300
            temperature=0.9,  # 更高的随机性
            use_chat_template=False
        )
        print(f"\nSample {i + 1}:\nQuestion: {sft_prompt_datas[i]} \nAI answer: {samples[0]}\n{'-' * 20}")
    
    print("\n ------------------- LoRA Sample ------------------- \n")
    lora_prompt_datas = [
        '你好',
        "阿里巴巴",
    ]
    generator = TextGenerator(checkpoint='../output/lora_output/lora_512.pth')  # 初始化生成器
    for i in range(len(lora_prompt_datas)):
        samples = generator.generate_samples(
            start=lora_prompt_datas[i],  # 自定义起始文本
            num_samples=5,   # 生成5个样本
            max_new_tokens=300,  # 最大长度300
            temperature=0.9,  # 更高的随机性
            use_chat_template=False
        )
        print(f"\nSample {i + 1}:\nQuestion: {lora_prompt_datas[i]} \nAI answer: {samples[0]}\n{'-' * 20}")
