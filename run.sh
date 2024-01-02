python main.py --model Conv_former \
                --data-set RSNA \
                --scales 1.0 \
                --img-list rsna \
                --data-path /data-8T/hyr/MCTformer/Datasets/rsna \
                --attention-type cls \
                --layer-index 4 \
                --visualize-cls-attn \
                --output_dir /data-8T/hyr/MCTformer/result_rsna/ \
                --finetune /data-8T/
