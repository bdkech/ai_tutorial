FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install jupyter RISE matplotlib seaborn scikit-learn tensorboard
