num_joints = 14     # lsp dataset

batch_size = 64
total_epoch = 500

# Train mode: 0-heatmap, 1-regression
train_mode = 1

continue_train = 0

best_pre_train = 499 # num of epoch where the training loss drops but testing accuracy achieve the optimal

# for test only
epoch_to_test = 199