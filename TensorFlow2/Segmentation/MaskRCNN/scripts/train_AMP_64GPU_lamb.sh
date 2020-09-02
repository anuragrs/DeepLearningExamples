#!/usr/bin/env bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

rm -rf /shared/results_8x

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

/opt/amazon/openmpi/bin/mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    --hostfile /shared/hostfiles/hosts_8x \
    -N 8 \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    bash /shared/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/launcher.sh \
    /shared/conda/bin/python ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
        --checkpoint="/shared/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --init_learning_rate=0.032 \
	--optimizer_type="LAMB" \
	--lr_schedule="piecewise" \
        --learning_rate_steps="3733,5133" \
        --model_dir="/shared/results_8x/" \
        --num_steps_per_eval=462 \
        --total_steps=5600 \
        --train_batch_size=4 \
	--l2_weight_decay=1e-4 \
        --eval_batch_size=8 \
        --training_file_pattern="/shared/data/coco/nv_tfrecords/train*.tfrecord" \
        --validation_file_pattern="/shared/data/coco/nv_tfrecords/val*.tfrecord" \
        --val_json_file="/shared/data/coco/nv_tfrecords/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --use_custom_box_proposals_op
#        --nouse_custom_box_proposals_op



#BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#/opt/amazon/openmpi/bin/mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
#    --mca btl_tcp_if_exclude lo,docker0 \
#    --hostfile /shared/hostfiles/hosts.local \
#    -N 8 \
#    -x NCCL_DEBUG=VERSION \
#    -x LD_LIBRARY_PATH \
#    -x PATH \
#    --oversubscribe \
#    bash /shared/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/launcher.sh \
#    /shared/conda/bin/python ${BASEDIR}/../mask_rcnn_main.py \
#        --mode="train_and_eval" \
#        --checkpoint="/shared/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
#        --eval_samples=5000 \
#        --init_learning_rate=0.04 \
#        --learning_rate_steps="30000,40000" \
#        --model_dir="/shared/results/" \
#        --num_steps_per_eval=3696 \
#        --total_steps=45000 \
#        --train_batch_size=4 \
#        --eval_batch_size=8 \
#        --training_file_pattern="/shared/data/coco/nv_tfrecords/train*.tfrecord" \
#        --validation_file_pattern="/shared/data/coco/nv_tfrecords/val*.tfrecord" \
#        --val_json_file="/shared/data/coco/nv_tfrecords/annotations/instances_val2017.json" \
#        --amp \
#        --use_batched_nms \
#        --xla \
#        --use_custom_box_proposals_op
