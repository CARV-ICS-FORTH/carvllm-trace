#   Copyright 2025 - 2026 Christos Kozanitis, FORTH, Greece
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler

'''
from transformers import ViTForImageClassification
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.cuda.amp import autocast, GradScaler
'''

from transformers import ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTLayer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # <-- auto-wrap policy
from functools import partial

from torch.amp import autocast, GradScaler



from gpu_activity_exporter import start_exporter, profiler_begin, profiler_end
rank = int(os.environ.get("RANK", "0"))
if rank == 0:
    start_exporter(port=9108, nvml_period_s=0.25)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h: return f"{h:d}h {m:d}m {s:.2f}s"
    elif m: return f"{m:d}m {s:.2f}s"
    else: return f"{s:.2f}s"

def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[GPU {local_rank}] NCCL init: rank {rank}/{world_size}", flush=True)
    return rank, world_size, local_rank

class HuggingFaceViTHuge(nn.Module):
    def __init__(self, num_labels=91):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-huge-patch14-224-in21k",   #  Huge checkpoint
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        return self.model(pixel_values=x).logits



def get_loader(train_dir, rank, world_size, batch_size=14, num_workers=4):
    tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(root=train_dir, transform=tf)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def train(args):
    rank, world_size, local_rank = setup()
    device = f"cuda:{local_rank}"

    wall_start = time.time()

    loader = get_loader(args.imagenet_train_dir, rank, world_size, batch_size=args.batch_size)
    base_model = HuggingFaceViTHuge(num_labels=1000).to(device)



# --- Auto-wrap policy: FSDP
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ViTLayer},
    )

    model = FSDP(
        base_model,
        device_id=torch.device(device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,

        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        #backward_prefetch=None,
        forward_prefetch=True,
        #forward_prefetch=False,
        mixed_precision=None,
    # limit_all_gathers=False,  #  default
    )



    '''
    loader = get_loader(args.coco_image_path, args.annotation_path, rank, world_size)
    base_model = HuggingFaceViTLarge(num_labels=91).to(device)
    model = FSDP(
        base_model,
        device_id=torch.device(device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=True,
        forward_prefetch=True,
        mixed_precision=None
    )
    '''
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    try:
        step = 0
        for epoch in range(args.num_epochs):
            loader.sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0

            for idx, (imgs, lbls) in enumerate(loader):
                if args.max_steps and step >= args.max_steps:
                    break
                step += 1
                os.environ["GPU_EXPORTER_EPOCH"] = str(epoch+1)
                if rank == 0:
                    profiler_begin()

                #imgs, lbls = imgs.to(device), lbls.to(device)
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)

                optimizer.zero_grad()

                try:
                    with autocast("cuda"):
                        out = model(imgs)
                    loss = loss_fn(out, lbls)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                finally:

                    if rank == 0:
                        torch.cuda.synchronize()
                        profiler_end()

                    pass

                total_loss += loss.item()

                if args.verbose and idx % 5 == 0:
                    print(f"[GPU {local_rank}] Epoch {epoch+1} | Step {idx+1} | Loss: {loss.item():.4f}",
                          flush=True)
            '''
            if rank == 0:
                print("[RANK0] about to call profiler_end()", flush=True)
                torch.cuda.synchronize()
                profiler_end()
                print("[RANK0] finished profiler_end()", flush=True)
            '''
            dist.barrier()
            print(f"[GPU {local_rank}] Epoch {epoch+1} complete | Avg loss: {total_loss/(idx+1):.4f}",
                  flush=True)

            if args.max_steps and step >= args.max_steps:
                print(f"[GPU {local_rank}] Reached max_steps={args.max_steps}, exiting.", flush=True)
                break

    finally:
        dist.barrier()
        elapsed = time.time() - wall_start
        print(f"[GPU {local_rank}] Cleanup complete. Total wall time: {fmt_duration(elapsed)} ({elapsed:.3f}s)",
              flush=True)
        dist.destroy_process_group()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_train_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=14)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("[MAIN] NCCL trainer emulating MPI config (ViT + FSDP)", flush=True)
    train(args)

if __name__ == "__main__":
    main()
