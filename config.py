num_joints = 14     # lsp dataset

batch_size = 128
total_epoch = 200
dataset = "lsp" # "lspet"

# Train mode: 0-heatmap, 1-regression
train_mode = 0

# Eval mode: 0-output image, 1-pck score
eval_mode = 0

continue_train = 0

best_pre_train = 120 # num of epoch where the training loss drops but testing accuracy achieve the optimal

# for test only
epoch_to_test = 200