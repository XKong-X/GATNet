import torch
import torch.optim
from torch.optim import Optimizer
import math

import torch
from torch.optim import Optimizer
import math

class TBO(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False, 
                 max_steps=10000, lb=-1.0, ub=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"无效的 epsilon 值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta 参数（索引 0）: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta 参数（索引 1）: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        max_steps=max_steps, lb=lb, ub=ub)
        super(TBO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('max_steps', 10000)
            group.setdefault('lb', -1.0)
            group.setdefault('ub', 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            max_steps = group['max_steps']
            lb = group['lb']
            ub = group['ub']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW_AE 不支持稀疏梯度')
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['prev_param'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['Pa'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['Pb'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state['step'] += 1
                state_steps.append(state['step'])

            # 自定义 AdamW 更新
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]

                # 应用权重衰减（AdamW 风格）
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                # 更新移动平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算分母
                denom = exp_avg_sq.sqrt().add_(eps)
                if amsgrad:
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)

                # 更新参数
                step_size = lr
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

            # AE 进化步骤
            for i, p in enumerate(params_with_grad):
                state = self.state[p]
                step = state['step']
                alpha = math.exp(math.log(1 - step / max_steps) - (4 * (step / max_steps))**2)
                cab = step / max_steps

                # 更新 Pa 和 Pb
                if torch.rand(1) < 0.5:
                    A = params_with_grad[torch.randint(len(params_with_grad), (1,))].detach()
                    state['Pa'] = (1 - cab) * state['Pa'] + cab * A
                    Ov = state['Pa']
                else:
                    K = max(1, int(len(params_with_grad) * torch.rand(1)))
                    selected = torch.randperm(len(params_with_grad))[:K]
                    weights = torch.softmax(torch.rand(K), dim=0)
                    Pb_sum = sum(weights[j] * params_with_grad[j].detach() for j in selected)
                    state['Pb'] = (1 - cab) * state['Pb'] + cab * Pb_sum
                    Ov = state['Pb']

                # 生成随机向量
                R1 = torch.rand_like(p)
                R2 = torch.rand_like(p)
                S = torch.randint(0, 2, p.shape, dtype=torch.float32, device=p.device)
                r = (ub - lb) * (2 * R1 * R2 - R2) * S
                ar = alpha * r

                # 进化路径
                sita = torch.rand(1) if torch.rand(1) < 0.5 else 2 * torch.rand(1)
                delta = Ov + ar + sita * (state['prev_param'] + p.detach() - Ov - state['prev_param'])

                # 使用边界约束更新参数
                p.add_(delta)
                p.clamp_(lb, ub)

                # 更新前一参数
                state['prev_param'] = p.detach().clone()

        return loss