import cv2
import tensorflow as tf
import numpy as np
from model import BlazePose
from config import epoch_to_test
from data import data, label

def Eclidian2(a, b):
# Calculate the square of Eclidian distance
    assert len(a)==len(b)
    summer = 0
    for i in range(len(a)):
        summer += (a[i] - b[i]) ** 2
    return summer

model = BlazePose()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path_regression = "checkpoints_regression/cp-{epoch:04d}.ckpt"
model.load_weights(checkpoint_path_regression.format(epoch=epoch_to_test))

y = np.zeros((2000, 14, 3)).astype(np.uint8)
batch_size = 20
for i in range(0, 2000, batch_size):
    if i + batch_size >= 2000:
        # last batch
        y[i : 2000] = model(data[i : i + batch_size]).numpy()
    else:
        # other batches
        y[i : i + batch_size] = model(data[i : i + batch_size]).numpy()
        print("=", end="")
print(">")

# CALCULATE PCKH SCORE
y = y[:,:,0:2].astype(float)
label = label[:,:,0:2].astype(float)
score_j = np.zeros(14)
pck_metric = 0.5
for i in range(1000, 2000):
    # validation part
    pck_h = Eclidian2(label[i][12], label[i][13])
    for j in range(14):
        pck_j = Eclidian2(y[i][j], label[i][j])
        # pck_j <= pck_h * 0.5 --> True
        if pck_j <= pck_h * pck_metric:
            # True estimation
            score_j[j] += 1
# convert to percentage
score_j = score_j * 0.1
score_avg = sum(score_j) / 14
print(score_j)
print("Average = %f%%" % score_avg)

# GENERATE RESULT IMAGES
for t in range(2000):
    skeleton = y[t]
    img = data[t].astype(np.uint8)
    # draw the joints
    for i in range(14):
        cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
    # draw the lines
    for j in ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), (2, 3), (2, 1), (1, 0), (3, 4), (4, 5)):
        cv2.line(img, tuple(skeleton[j[0]][0:2]), tuple(skeleton[j[1]][0:2]), color=(0, 0, 255), thickness=1)
    # solve the mid point of the hips
    cv2.line(img, tuple(skeleton[12][0:2]), tuple(skeleton[2][0:2] // 2 + skeleton[3][0:2] // 2), color=(0, 0, 255), thickness=1)

    cv2.imwrite("./result/lsp_%d.jpg"%t, img)