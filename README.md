# Apply CLIP on pathological images
Original repository: https://github.com/openai/CLIP
Follow the original instructions to install and run CLIP.

## Requirement (verified environment)
- NVIDIA Quadro RTX 8000
- CUDA 10.2
- pytorch 1.8.1
- clip 1.0
- Pillow 8.2.0
- numpy 1.19.5
- scikit-learn 0.24.2
- tqdm 3.60.0
- ftfy 6.1.1
- regex 2022.3.15
For pytorch_tensorflow_translation.ipynb
- tensorflow   2.3.1

## Dataset preparation
`git clone https://github.com/enigmanx20/CompanionMeeting20220415.git`

`cd CompanionMeeting20220415`

Download tarball:
https://drive.google.com/file/d/1ChmAopO9EfiZJMiDYzw5VNSGTDvLcHn4/view?usp=sharing

`mv ~/downloads/PCam200.tar.gz ./`

`tar -zxvf PCam200.tar.gz`

 
## Zero-shot prediction
Predicting normal and tumor on few samples without any re-training. The performance poor. Perhaps pathological images were scarce in the dataset of 400M images-text pairs (named WebImageText) to train CLIP models.

## Linear probe
Regarding the model as a freezed encoder to extract a feature vector from an image. Then a logistic regression model is trained from the feature vector. This logistic regression model is similar to one layer perceptron with logistic (or softmax) activations. 

## Linear head transfer learning
The original fully connected (or dense) layer for logits is replaced with a fully connected (or dense) layer with intended outputs (2 outputs in PCam200). Then entire network is re-trained. Stochastic  gradient descent (SGD) with small learning rate (typically 1e-4 ~ 3e-3) is often used for this purpose. Usually the performance is better than linear probe and often comparable with scratch training.
