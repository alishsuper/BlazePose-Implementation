num_joints = 14     # lsp dataset

batch_size = 64
total_epoch = 500

# Train mode: 0-pre-train, 1-finetune
train_mode = 0

if train_mode:
    best_pre_train = 499 # num of epoch where the training loss drops but testing accuracy achieve the optimal

# for test only
epoch_to_test = 199
# for test the heatmap only
vis_img_id = 1797

json_name = "train_record.json" if train_mode else "train_record_pretrain.json"