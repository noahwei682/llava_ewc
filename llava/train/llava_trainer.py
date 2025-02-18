import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        # [print(name) for name, _ in opt_model.named_parameters() ]

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # if self.args.vision_encoder_lr is not None and self.args.mm_projector_lr is None:
            #     vision_encoder_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
            #     # optimizer_grouped_parameters = [                    
            #     #     {
            #     #         "params": [
            #     #             p for n, p in opt_model.named_parameters() if (n in decay_parameters and vision_encoder_parameters and p.requires_grad)
            #     #         ],
            #     #         "weight_decay": self.args.weight_decay,
            #     #         "lr": self.args.vision_encoder_lr,
            #     #     },
            #     #     {
            #     #         "params": [
            #     #             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and vision_encoder_parameters and p.requires_grad)
            #     #         ],
            #     #         "weight_decay": 0.0,
            #     #         "lr": self.args.vision_encoder_lr,
            #     #     },
            #     #     {
            #     #         "params": [
            #     #             p for n, p in opt_model.named_parameters() if ((n in decay_parameters and p.requires_grad) and (n not in vision_encoder_parameters))
            #     #         ],
            #     #         "weight_decay": self.args.weight_decay,
            #     #     },
            #     #     {
            #     #         "params": [
            #     #             p for n, p in opt_model.named_parameters() if ((n not in decay_parameters and p.requires_grad) and (n not in vision_encoder_parameters))
            #     #         ],
            #     #         "weight_decay": 0.0,
            #     #     },
            #     # ]
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": 0.0,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #             "lr": self.args.vision_encoder_lr,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": 0.0,
            #             "lr": self.args.vision_encoder_lr,
            #         },
            #     ]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            # # ----vision encder opt------------------------------------------------------------------------------------------------------------------------------
            # if self.args.vision_encoder_lr is not None:
            #     vision_encoder_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": 0.0,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #             "lr": self.args.vision_encoder_lr,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_encoder_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": 0.0,
            #             "lr": self.args.vision_encoder_lr,
            #         },
            #     ]
            # else:
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": 0.0,
            #         },
            #     ]
            # # ----vision encder opt------------------------------------------------------------------------------------------------------------------------------


            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        # for name, param in opt_model.named_parameters():
        #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


from copy import deepcopy
# 【新增】EWC实现
class EWC:
    def __init__(self, model: nn.Module):
        """
        初始化EWC
        """
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = None
        
    def update_fisher_matrix(self, data_loader, device):
        """
        优化后的Fisher信息矩阵计算
        """
        precision_matrices = {}
        # 初始化时在CPU上创建
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p.data, device='cpu')

        self.model.eval()
        # 限制使用的batch数量
        max_batches = 10  # 可调整
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            self.model.zero_grad()
            # 使用torch.cuda.amp.autocast()降低显存使用
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # 使用梯度累积
            loss.backward()
            
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        # 即时更新precision_matrices并释放梯度
                        precision_matrices[n].add_((p.grad.data ** 2).cpu() / max_batches)
                        p.grad = None  # 释放梯度
                    
            # 主动清理显存
            torch.cuda.empty_cache()

        # 最后再将结果转移到GPU
        self._precision_matrices = {n: p.to(device) for n, p in precision_matrices.items()}
        # 保存当前参数
        with torch.no_grad():
            self._means = {n: p.data.clone() for n, p in self.params.items()}

    def compute_penalty(self, model: nn.Module):
        """
        优化后的EWC惩罚项计算
        """
        if self._precision_matrices is None:
            return torch.tensor(0., device=next(model.parameters()).device)
        
        loss = 0
        # 分批计算损失以减少峰值显存使用
        for n, p in model.named_parameters():
            if n in self._precision_matrices and p.requires_grad:
                # 使用torch.cuda.amp.autocast()降低显存使用
                with torch.cuda.amp.autocast():
                    _loss = (self._precision_matrices[n] * (p - self._means[n]) ** 2).sum()
                    loss += _loss
                
        return loss

# 【新增】继承LLaVATrainer实现LLaVATrainerWithEWC
class LLaVATrainerWithEWC(LLaVATrainer):
    def __init__(self, model, args, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        self.ewc = None
        # 从model_args获取EWC参数
        self.use_ewc = args.use_ewc
        self.ewc_lambda = args.ewc_lambda
        
        if self.use_ewc:
            self.initialize_ewc()
    
    def initialize_ewc(self):
        """
        优化后的EWC初始化
        """
        if self.use_ewc and self.ewc_lambda > 0:
            print("Initializing EWC...")
            self.ewc = EWC(self.model)
            
            # 只使用一小部分数据来计算Fisher信息矩阵
            small_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                num_workers=self.args.dataloader_num_workers,
                drop_last=True,
                # 限制数据量
                sampler=torch.utils.data.RandomSampler(
                    self.train_dataset,
                    num_samples=min(128, len(self.train_dataset))  # 限制样本数
                )
            )
            
            self.ewc.update_fisher_matrix(small_loader, self.args.device)
            print("EWC initialization completed")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算损失，包括EWC正则化项
        """
        # 获取原始的损失和输出
        # loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # 添加EWC惩罚项
        if self.use_ewc and self.ewc is not None and self.ewc_lambda > 0:
            ewc_loss = self.ewc.compute_penalty(model)
            print("\nEWC loss: ", ewc_loss)
            total_loss = loss + self.ewc_lambda * ewc_loss
            
            # 记录损失值
            if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "task_loss": loss.item(),
                    "ewc_loss": ewc_loss.item(),
                    "total_loss": total_loss.item(),
                    "ewc_lambda": self.ewc_lambda
                })
        else:
            total_loss = loss
            
        return (total_loss, outputs) if return_outputs else total_loss