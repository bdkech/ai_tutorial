# AI Tutorial

This is just a suite of notebooks and code to help demonstrate basic workflows and concepts for AI using PyTorch libraries.  The notebook provides documentation and code for tinkering, while the ai_learning library provides the models and additional helpers to gather training results and output them for easier tensorboard visualization.

## Setup

You can either install the necessary packages to your system python or virtual environment via

```
pip install -r requirements.txt
```

or alternatively you can build a docker container using the provided Dockerfile.  As a warning, the image is approximately 5G in size.

```
docker build . -t ai_learning
docker run --rm -it -v ${PWD}:/workspace -p 8888:8888 \
                                         -p 6006:6006 ai_learning:latest \
                                         jupyter notebook --allow-root --ip='*'
```
