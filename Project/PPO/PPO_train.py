# ruff: noqa: E731 F401 F841
import random
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.utils.tensorboard import SummaryWriter


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer

        self.final_prompts = []

        for prompt in prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]

                # 表示在末尾加上模型的“回答提示”（如 <|im_start|>assistant）。
                prompt = self.tokenizer.apply_chat_template(
                    content, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = self.tokenizer.bos_token + prompt

            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.final_prompts[index]


class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, num_action):
        # [batch_size, sequence_length, hidden_size]
        hidden_state = self.base_model(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state

        value_output = self.value_head(hidden_state)
        # [batch_size, num_action]
        value_output = value_output.squeeze(-1)[:, -num_action:]

        return value_output


@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor


@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions",
        )
        for key in keys:
            for i, x in enumerate(experiences):
                value = getattr(x, key)
                batch[i][key] = value

        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer) - self.limit :]

    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


def generate_samples(
    prompts,
    model,
    max_length,
    max_new_tokens,
    n_samples_per_prompt,
    micro_rollout_batch_size,
):
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])

    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i : i + micro_rollout_batch_size]
        inputs = actor_tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs_ids = inputs["input_ids"]
        seqs = model.generate(
            **inputs.to(device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        # 原始加生成
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, : max_new_tokens + max_length]
        else:
            seqs = torch.cat(
                [
                    seqs,
                    torch.full(
                        (seqs.size(0), max_new_tokens + max_length - seqs.size(1)),
                        fill_value=pad_token_id,
                        device=seqs.device,
                    ),
                ],
                dim=1,
            )
        # true 变 1 eos有效 后面全是pad  所以attentionmask 比actionmask少马赛克一个
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        ans = seqs[:, inputs_ids.size(1) :]
        # 只在最后一个 <eos> 之前的 token 是 PPO 的“有效行为”。
        action_mask = (ans.ne(pad_token_id) & ans.ne(eos_token_id)).to(dtype=torch.long)
        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)

    return samples_list


def compute_approx_kl(log_probs, ref_log_probs, action_mask=None):
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    kl_divergence_estimate = -kl_ctl * kl
    # [batch,num_actions]
    rewards = kl_divergence_estimate

    # [batch_size]
    ends = action_mask.sum(1) + 1

    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    # 是为了让奖励在 对称区间 [-x, +x] 内截断， 因为 reward model 输出是实值打分，以 0 为中心
    rewad_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)
    batch_size = r.size(0)

    for j in range(batch_size):
        # 该样本最后一个有效动作的位置 末尾加一个奖励 其他全是-kl
        # 这样写不怕越界。。 不能用rewards[j,ends[j]]
        rewards[j, : ends[j]][-1] += rewad_clip[j, 0]

    return rewards


# A(t) = R(t) + gam*V(t+1) - V(t)
# gae:A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
# A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
# returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
def get_advantages_and_returns(values, rewards, action_mask, gamma=0.1, lambd=0.2):
    lastgaelam = 0
    advantage_reverse = []
    response_length = rewards.size(1)

    if action_mask is not None:
        values = values * action_mask
        rewards = rewards * action_mask

    for t in range(response_length - 1, -1, -1):
        next_value = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantage_reverse.append(lastgaelam)

    advantages = torch.stack(advantage_reverse[::-1], dim=1)
    returns = advantages + values

    return advantages.detach(), returns


def generate_experiences(samples_list):
    actor_model.eval()
    critic_model.eval()
    reward_model.eval()
    ref_model.eval()

    experiences = []
    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        with torch.no_grad():
            # 计算策略模型输出token的概率
            output = actor_model(seqs, attention_mask=attention_mask)
            logits = output.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            # [batch,num_actions]
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
            # 参考模型
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(
                dim=-1, index=seqs[:, 1:].unsqueeze(-1)
            )
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]

            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)

            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # 计算奖励模型的奖励值
            reward_model_inputs = reward_tokenizer(
                seq_texts, return_tensors="pt", padding=True
            )
            # 奖励模型的输出，相当于生成最后一个token的奖励（结果奖励模型）
            # r.shape == [batch_size, 1]
            r = reward_model(**reward_model_inputs.to(device)).logits

            kl = compute_approx_kl(action_log_probs, ref_action_log_probs).to(device)

            # [batch,num_actions]
            rewards = compute_rewards(
                kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2
            )

            advantages, returns = get_advantages_and_returns(
                value, rewards, action_mask, gamma=0.1, lambd=0.2
            )

        experiences.append(
            Experience(
                seqs,
                action_log_probs.detach(),
                value.detach(),
                returns.detach(),
                advantages.detach(),
                attention_mask,
                action_mask,
                r.detach(),
                samples.response_length,
                samples.total_length,
                num_actions,
                kl.detach(),
            )
        )

    return experiences


def compute_policy_loss(
    action_log_probs, old_action_log_probs, advantages, action_mask=None, clip_eps=0.2
):
    ratio = (action_log_probs - old_action_log_probs).exp()
    clip_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    # [batch,num_actions]
    actor_loss = -torch.min(ratio * advantages, clip_ratio * advantages)
    if action_mask is None:
        return actor_loss.mean(-1).mean()
    return ((actor_loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps=None):
    # 防止 critic 在一次更新中过度偏离上次的预测
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2

    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


@dataclass
class BufferItem:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]


def collate_fn(batch):
    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []

    for x in batch:
        seqs.append(x["seqs"])
        action_log_probs.append(x["action_log_probs"])
        values.append(x["values"])
        returns.append(x["returns"])
        advantages.append(x["advantages"])
        attention_mask.append(x["attention_mask"])
        action_mask.append(x["action_mask"])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)

    return BufferItem(
        seqs,
        action_log_probs,
        values,
        returns,
        advantages,
        attention_mask,
        action_mask,
        action_mask.size(1),
    )


def train_step(experience, steps):
    actor_model.train()
    optimizer_actor.zero_grad()

    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns

    logits = actor_model(sequences, attention_mask=attention_mask).logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    # [batch,num_actions]
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    policy_loss = compute_policy_loss(
        action_log_probs, old_action_log_probs, advantages, action_mask
    )

    policy_loss.backward()
    optimizer_actor.step()
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    # critic更新
    critic_model.train()
    optimizer_critic.zero_grad()

    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(
        values, old_values, returns, action_mask=action_mask
    )

    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)

    print(
        f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}"
    )


def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            samples = generate_samples(
                rand_prompts,
                actor_model,
                max_length,
                max_new_tokens,
                n_samples_per_prompt,
                micro_rollout_batch_size,
            )
            experiences = generate_experiences(samples)
            buffer.append(experiences)

            dataloader = DataLoader(
                buffer,
                batch_size=micro_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            torch.cuda.empty_cache()

            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1

            buffer.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 100
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2

    writer = SummaryWriter("./runs")
    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to(
        device
    )
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to(
        device
    )
    # 奖励模型  能够对一段文本（如模型生成的回答）输出一个奖励分数（reward）
    # 输出 SequenceClassifierOutput 包含 .logits, .loss, .hidden_states 等
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    ).to(device)

    actor_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )

    # 价值模型
    critic_model = Critic(actor_model.base_model).to(device)

    # 初始化优化器
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)

    # 填充方式为左填充
    actor_tokenizer.padding_side = "left"
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        "请问1+1等于多少？",
        "PowerShell，如何知道BIOS中的虚拟化是否已禁用",
        "为什么人们喜欢在水族馆里游泳，而不是在游泳池里？",
        "你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。",
        "你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。",
        "你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。",
        "为什么所有的镜子都是矩形的？",
        "我们在受感染的植物根部可以找到哪一种，臭氧还是金子？",
    ]
    prompts_dataset = PromptDataset(
        prompt_list, actor_tokenizer, apply_chat_template=True
    )
    prompts_dataloader = DataLoader(
        prompts_dataset, batch_size=rollout_batch_size, shuffle=True
    )
    train()
