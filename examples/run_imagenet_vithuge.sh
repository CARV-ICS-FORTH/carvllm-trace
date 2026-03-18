# On each node, replace NODE_RANK accordingly (0 for master, 1 for second node)
MASTER_ADDR=10.1.1.1  # Master node's IP
#MASTER_ADDR=192.168.1.104
MASTER_PORT=29500	   # Master node's Port

WORLD_SIZE=2  		   # Total number of nodes

echo "ENTER NODE RANK:"
PARAM=$1
echo "You entered: $PARAM"

NODE_RANK=$PARAM  		# Change per machine
GPUS_PER_NODE=2  		# Adjust based on available GPUs. Put the number of GPUs per Node


export NCCL_ALGO=Ring

# --- Logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$PWD/nccl_%h_%p____huge_.log"   # %r = rank, %p = pid, %h = host


export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ibs15f0


export NCCL_IB_GID_INDEX=0


# --- Torch/torchrun logs ---
export TORCH_DISTRIBUTED_DEBUG=INFO
export GPU_EXPORTER_DEBUG_EVENTS=1
export GPU_EXPORTER_DEBUG_TOP=50
export GPU_EXPORTER_DUMP_ALL=1
export GPU_EXPORTER_DUMP_DIR="$PWD/GPU_exporter_logs_Huge_imagenet"

torchrun   	--nnodes=$WORLD_SIZE \
		--node_rank=$NODE_RANK \
        	--nproc_per_node=$GPUS_PER_NODE \
        	--rdzv_id=123 \
		--rdzv_backend=static \
       		--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
		imagenet_vit_huge_FSDP_nccl.py --imagenet_train_dir /mnt/vol1/poldaf/imagenet/ILSVRC2012/extracted/train --num_epochs 2 --max_steps 1 --verbose
