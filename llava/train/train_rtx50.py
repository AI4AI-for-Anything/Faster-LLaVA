"""LLaVA-1.5 stage-two training on one host of RTX 50 series (or other 16 GB) GPUs.

One process drives every visible GPU through RoundPipe: parameters, gradients and
AdamW state stay in host memory and each decoder stage is streamed to a GPU, so
the 7B model trains without ZeRO-3 collectives (which crawl over PCIe without
peer-to-peer).

The recipe matches ``train.py`` with ``scripts/zero3.json``: the same data,
preprocessing and sampler (each optimizer step holds the same samples as the
reference GPU count would), bf16 compute with fp32 master weights and AdamW
state, cosine schedule with warmup, no gradient clipping, every layer
recomputed in backward.  Differences: samples are packed without padding, and
the loss is the mean over all supervised tokens of the optimizer step (as
``LLAVA_GLOBAL_TOKEN_LOSS=1``).  Optimizer steps are synchronous; RoundPipe's
asynchronous mode would train on one-step-stale weights.
"""
import json
import math
import os
import pathlib
import shutil
import time
from dataclasses import asdict, dataclass, field

import torch
import transformers

from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model import LlavaLlamaForCausalLM
from llava.model.rtx50_sequential import pack_microbatch, split_balanced, to_sequential
from llava.train.llava_trainer import get_modality_length_grouped_indices
from llava.train.train import (DataArguments, ModelArguments, TrainingArguments,
                               make_supervised_data_module)


@dataclass
class RTX50Arguments:
    reference_world_size: int = field(default=8, metadata={
        'help': 'GPU count of the reference DeepSpeed recipe. One optimizer step holds '
                'per_device_train_batch_size x reference_world_size x gradient_accumulation_steps '
                'samples, grouped exactly as that recipe groups them.'})
    num_microbatch: int = field(default=0, metadata={
        'help': 'RoundPipe microbatches per optimizer step (token-balanced); 0 = GPU count + 1.'})


def unsupported(model_args, data_args, training_args):
    checks = {
        'bits': training_args.bits != 16, 'lora_enable': training_args.lora_enable,
        'fp16': training_args.fp16, 'bf16 off': not training_args.bf16,
        'freeze_backbone': model_args.freeze_backbone,
        'tune_mm_mlp_adapter': model_args.tune_mm_mlp_adapter,
        'freeze_mm_mlp_adapter': training_args.freeze_mm_mlp_adapter,
        'mm_projector_lr': training_args.mm_projector_lr is not None,
        'mm_use_im_start_end': model_args.mm_use_im_start_end,
        'no vision_tower': model_args.vision_tower is None,
        'mpt': 'mpt' in model_args.model_name_or_path,
        'version v0': model_args.version == 'v0',
        'group_by_modality_length off': not training_args.group_by_modality_length,
    }
    bad = [name for name, hit in checks.items() if hit]
    if bad:
        raise NotImplementedError(f'train_rtx50.py covers the full-parameter bf16 stage-two '
                                  f'recipe; unsupported here: {", ".join(bad)}')


def build_model(model_args, data_args, training_args, resume=None):
    """The bf16 full-finetune path of ``train.train()``, without DeepSpeed."""
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length, padding_side='right', use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version != 'v0.5':
        conversation_lib.default_conversation = conversation_lib.conv_templates.get(
            model_args.version, conversation_lib.conv_templates['vicuna_v1'])

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    vision_tower = model.get_vision_tower()
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = False
    model.config.freeze_mm_mlp_adapter = False
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = False
    model.config.mm_projector_lr = None
    training_args.use_im_start_end = False
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    if resume is not None:  # Trained weights, including the projector, replace the initial ones.
        keys = transformers.modeling_utils.load_sharded_checkpoint(model, resume, strict=False)
        if keys.missing_keys or keys.unexpected_keys:
            raise RuntimeError(f'{resume} does not match the model: {keys}')
    model.to(torch.bfloat16)  # DeepSpeed bf16 casts the whole module, the projector included.
    check_generation_config(model)
    return model, tokenizer


def check_generation_config(model):
    """Fail at start-up, not at the final save, on a config that ``save_pretrained`` rejects."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.generation_config.validate()
    if caught:
        raise ValueError(f'generation_config.json of the base model cannot be saved: '
                         f'{[str(w.message) for w in caught]} (see docs/training.md)')


def step_schedule(dataset, training_args, rtx50_args):
    """Optimizer-step membership exactly as ``LLaVATrainer``'s sampler builds it."""
    group = rtx50_args.reference_world_size * training_args.gradient_accumulation_steps
    step_size = training_args.per_device_train_batch_size * group
    transformers.set_seed(training_args.seed)
    steps = []
    for _ in range(max(1, math.ceil(training_args.num_train_epochs))):
        # Each epoch the Trainer's DataLoader iterator draws its worker base seed from the
        # global generator before the sampler runs; the sampler's order depends on that draw.
        torch.empty((), dtype=torch.int64).random_()
        order = get_modality_length_grouped_indices(
            dataset.modality_lengths, training_args.per_device_train_batch_size, group, generator=None)
        steps += [order[i:i + step_size] for i in range(0, len(order), step_size)]
    if training_args.max_steps > 0:
        steps = steps[:training_args.max_steps]
    return steps


class PackSteps:
    """DataLoader collate: one optimizer step -> token-balanced packed microbatches."""

    def __init__(self, num_microbatch, image_tokens, max_length, pad_token_id):
        self.num_microbatch, self.image_tokens = num_microbatch, image_tokens
        self.max_length, self.pad_token_id = max_length, pad_token_id

    def __call__(self, items):
        lengths = []
        for item in items:
            ids = item['input_ids'][:self.max_length]
            ids = ids[ids != self.pad_token_id]
            images = int((ids == IMAGE_TOKEN_INDEX).sum())
            lengths.append(min(len(ids) + images * (self.image_tokens - 1), self.max_length))
        groups = split_balanced(lengths, min(self.num_microbatch, len(items)))
        microbatches = [pack_microbatch([items[i] for i in g], self.image_tokens,
                                        self.max_length, self.pad_token_id) for g in groups]
        return dict(microbatches=microbatches, num_samples=len(items), num_rows=sum(lengths),
                    num_labels=sum(int(mb['label_ids'].numel()) for mb in microbatches))


def _by_microbatch(args, kwargs, num_microbatch):
    microbatches = args[0]
    assert len(microbatches) == num_microbatch
    return [(mb,) for mb in microbatches], [{} for _ in microbatches]


def save_checkpoint(path, model, tokenizer, rp, optimizer, scheduler, state, with_optimizer):
    tmp = pathlib.Path(str(path) + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    model.config.use_cache = True
    model.save_pretrained(tmp)
    model.config.use_cache = False
    tokenizer.save_pretrained(tmp)
    if with_optimizer:
        from roundpipe import OptimizerCtx
        with OptimizerCtx():
            master = {name: p.detach() for name, p in rp.named_parameters()}
        torch.save(dict(master=master, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict()), tmp / 'optimizer.pt')
    (tmp / 'trainer_state.json').write_text(json.dumps(state, indent=2))
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, RTX50Arguments))
    model_args, data_args, training_args, rtx50_args = parser.parse_args_into_dataclasses()
    unsupported(model_args, data_args, training_args)
    out = pathlib.Path(training_args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(out.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    resume = ckpts[-1] if ckpts else None
    start_step = json.loads((resume / 'trainer_state.json').read_text())['global_step'] if resume else 0

    model, tokenizer = build_model(model_args, data_args, training_args,
                                   resume=str(resume) if resume else None)
    dataset = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)['train_dataset']
    steps = step_schedule(dataset, training_args, rtx50_args)
    num_microbatch = rtx50_args.num_microbatch or torch.cuda.device_count() + 1
    image_tokens = model.get_vision_tower().num_patches
    collate = PackSteps(num_microbatch, image_tokens, tokenizer.model_max_length, tokenizer.pad_token_id)
    workers = training_args.dataloader_num_workers
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=steps[start_step:], collate_fn=collate, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    batches = iter(loader)  # Fork data workers before RoundPipe starts its threads.

    from roundpipe import OptimizerCtx, RoundPipe, RoundPipeRunConfig
    from roundpipe.optim import Adam
    rp = RoundPipe(to_sequential(model), optim_dtype=torch.float32)
    with OptimizerCtx():
        named = list(rp.named_parameters())
    decay = [p for _, p in named if p.ndim >= 2]  # HF: no decay on biases and norms
    no_decay = [p for _, p in named if p.ndim < 2]
    optimizer = Adam([dict(params=decay, weight_decay=training_args.weight_decay),
                      dict(params=no_decay, weight_decay=0.0)],
                     lr=training_args.learning_rate,
                     betas=(training_args.adam_beta1, training_args.adam_beta2),
                     eps=training_args.adam_epsilon, decoupled_weight_decay=True)
    scheduler = transformers.get_scheduler(
        training_args.lr_scheduler_type, optimizer,
        num_warmup_steps=training_args.get_warmup_steps(len(steps)),
        num_training_steps=len(steps))
    state = dict(global_step=start_step, max_steps=len(steps), log_history=[],
                 args=dict(model=asdict(model_args), rtx50=asdict(rtx50_args)))
    if resume:
        saved = torch.load(resume / 'optimizer.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            for name, p in named:
                p.copy_(saved['master'][name])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state['log_history'] = json.loads((resume / 'trainer_state.json').read_text())['log_history']
        del saved

    clip = training_args.max_grad_norm or 0.0
    params = [p for _, p in named]

    def optimizer_step():
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    print(f'[rtx50] {len(steps)} optimizer steps, starting at {start_step}; '
          f'{num_microbatch} microbatches over {torch.cuda.device_count()} GPUs', flush=True)
    for step in range(start_step, len(steps)):
        begin = time.perf_counter()
        batch = next(batches)
        loaded = time.perf_counter()
        lr = scheduler.get_last_lr()[0]
        num_labels = max(batch['num_labels'], 1)
        loss = rp.forward_backward(
            input_args=(batch['microbatches'],),
            loss_fn=lambda total, _: total / num_labels,
            run_config=RoundPipeRunConfig(num_microbatch=len(batch['microbatches']),
                                          split_input=_by_microbatch))
        computed = time.perf_counter()
        rp.step(optimizer_step, is_async=False)
        end = time.perf_counter()
        state['global_step'] = step + 1
        if (step + 1) % max(1, int(training_args.logging_steps)) == 0 or step + 1 == len(steps):
            entry = dict(step=step + 1, loss=round(float(loss), 6), learning_rate=lr,
                         epoch=round((step + 1) / len(steps) * training_args.num_train_epochs, 4),
                         step_time=round(end - begin, 4), data_wait=round(loaded - begin, 4),
                         optimizer_time=round(end - computed, 4),
                         samples=batch['num_samples'], tokens=batch['num_rows'],
                         labels=batch['num_labels'],
                         peak_gib=round(max(torch.cuda.max_memory_reserved(d) for d in
                                            range(torch.cuda.device_count())) / 2**30, 2))
            state['log_history'].append(entry)
            print(json.dumps(entry), flush=True)
        if (training_args.save_strategy == 'steps' and training_args.save_steps
                and (step + 1) % int(training_args.save_steps) == 0 and step + 1 < len(steps)):
            save_checkpoint(out / f'checkpoint-{step + 1}', model, tokenizer, rp, optimizer,
                            scheduler, state, with_optimizer=not training_args.save_only_model)
            if training_args.save_total_limit:
                old = sorted(out.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
                for path in old[:-int(training_args.save_total_limit)]:
                    shutil.rmtree(path, ignore_errors=True)

    model.config.use_cache = True
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / 'trainer_state.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    train()
