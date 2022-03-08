#!~/miniconda3/envs/tf2/bin/python
import tensorflow as tf
from model import BlazePose
from config import total_epoch, train_mode, best_pre_train, continue_train
from analysis import save_record

checkpoint_path_regression = "checkpoints_regression/cp-{epoch:04d}.ckpt"
checkpoint_path_heatmap = "checkpoints_heatmap/cp-{epoch:04d}.ckpt"

if train_mode:
    from data import finetune_train as train_dataset
    from data import finetune_validation as test_dataset
    loss_func = tf.keras.losses.MeanSquaredError()
    checkpoint_path = checkpoint_path_regression
else:
    from data import train_dataset, test_dataset
    loss_func = tf.keras.losses.BinaryCrossentropy()
    checkpoint_path = checkpoint_path_heatmap

model = BlazePose()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_func(y_true=targets, y_pred=model(inputs))
    return loss_value, tape.gradient(loss_value, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

# continue train
if continue_train > 0:
    model.load_weights(checkpoint_path.format(epoch=continue_train))
else:
    if train_mode:
        model.load_weights(checkpoint_path_heatmap.format(epoch=best_pre_train))

if train_mode:
    # start fine-tune
    for layer in model.layers[0:16]:
        layer.trainable = False
else:
    # pre-train
    for layer in model.layers[16:24]:
        layer.trainable = False

train_loss_results = []
train_accuracy_results = []
val_accuracy_results = []

for epoch in range(continue_train, total_epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.MeanSquaredError()
    val_accuracy = tf.keras.metrics.MeanSquaredError()

    # Training loop
    for x, y in train_dataset:
        # create separate outputs for coordinates and visibility
        # if train_mode:
        #     y = (y[:, :, 0:2], tf.concat([tf.expand_dims(y[:, :, 2], axis=2), tf.expand_dims(y[:, :, 2], axis=2)], axis=2))

        # Optimize
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Add current batch loss
        epoch_loss_avg(loss_value)
        # Calculate error from Ground truth
        epoch_accuracy(y, model(x))

    # Record loss and accuracy
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
            
    # Train loss at epoch
    print("Epoch {:03d}: Train Loss: {:.3f}, Accuracy: {:.5%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
    
    if not((epoch + 1) % 5):
        # validata and save weight every 5 epochs
        for x, y in test_dataset:
            # create separate outputs for coordinates and visibility
            # if train_mode:
            #     y = (y[:, :, 0:2], tf.concat([tf.expand_dims(y[:, :, 2], axis=2), tf.expand_dims(y[:, :, 2], axis=2)], axis=2))
            val_accuracy(y, model(x))
        print("Epoch {:03d}, Validation accuracy: {:.5%}".format(epoch, val_accuracy.result()))
        model.save_weights(checkpoint_path.format(epoch=epoch))
        val_accuracy_results.append(val_accuracy.result())

        # save the training record at every validation epoch
        save_record(train_loss_results, train_accuracy_results, val_accuracy_results)

model.summary()
print("Finish training.")