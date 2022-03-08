# BlazePose-Implementation in Tensorflow
BlazePose paper is "BlazePose: On-device Real-time Body Pose tracking" by Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann. Available on [arXiv](https://arxiv.org/abs/2006.10204).

## Requirements
```
Anaconda
Python
Tensorflow
```
Please import Anaconda environment file (BlazePoseTensorflow.yml).

## Dataset
Leeds Sports Pose Dataset
Sam Johnson and Mark Everingham
http://sam.johnson.io/research/lsp.html

This dataset contains 2000 images of mostly sports people
gathered from Flickr. The images have been scaled such that the
most prominent person is roughly 150 pixels in length. The file
joints.mat contains 14 joint locations for each image along with
a binary value specifying joint visibility.

The ordering of the joints is as follows:
```
Right ankle
Right knee
Right hip
Left hip
Left knee
Left ankle
Right wrist
Right elbow
Right shoulder
Left shoulder
Left elbow
Left wrist
Neck
Head top
```

## Model
![image](https://user-images.githubusercontent.com/14852495/156509720-2d900f7b-8953-4219-9aa8-dea97dccb93c.png)
![image](https://user-images.githubusercontent.com/14852495/156510922-5d962d87-e021-4a3f-9c67-3afbd168a022.png)

![image](https://user-images.githubusercontent.com/14852495/156573965-3776af14-ffaa-4e65-a5c9-eb4a7ebcf1b5.png)

## Train
1. Pre-train the heatmap branch.
    Edit training settings in `config.py`. Set `train_mode = 0`.
    Then, run `python train.py`.
    
2. Fine-tune for the joint regression branch.
    Set `train_mode = 1` and `best_pre_train` with the num of epoch where the training loss drops but testing accuracy achieve the optimal.
    Then, run `python train.py`.

## Test
1. Set `epoch_to_test` to the epoch you would like to test.

2. Run `python test.py`.

## Performance Comparison
| Model                                                | LSP Dataset <br /> Train - 1000 images <br /> Val - 1000 images              | LSPet Dataset Train - 2500 images <br /> Val - 500 images |
| ---------------------------------------------------- | ---------------------------------------------------------------------------- | ------------- |
| Only Linear (x, y, v)                                | PCK Score – 36.53% <br /> Train MSE Loss - 7.19 <br /> Val MSE Loss – 742.19 | PCK Score – 9.27% <br /> Train MSE Loss - 24.82 <br /> Val MSE Loss – 6244.79 |
| Linear (x, y) + Sigmoid (v) (Concatenate two layers) | PCK Score – 38.6% <br /> Train MSE Loss - 1.05 <br /> Val MSE Loss – 705.67  | |
| Linear (x, y) + Sigmoid (v) (Separate two outputs)   | PCK Score – 37.74% <br /> Train MSE Los - 2.97 <br /> Val MSE Loss - 556.32  | |

## Reference

If the original paper helps your research, you can cite this paper in the LaTex file with:

```tex
@article{Bazarevsky2020BlazePoseOR,
  title={BlazePose: On-device Real-time Body Pose tracking},
  author={Valentin Bazarevsky and I. Grishchenko and K. Raveendran and Tyler Lixuan Zhu and Fangfang Zhang and M. Grundmann},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.10204}
}
```
