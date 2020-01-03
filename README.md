## AdaptIS: Adaptive Instance Selection Network (pytorch With DataParallel Support)
This codebase implements the system described in the paper ["AdaptIS: Adaptive Instance Selection Network"](https://arxiv.org/abs/1909.07829), Konstantin Sofiiuk, Olga Barinova, Anton Konushin. Accepted at ICCV 2019.
The code performs **panoptic segmentation** and can be also used for **instance segmentation**.

<p align="center">
  <img src="./images/adaptis_model_scheme.png" alt="drawing" width="600"/>
</p>


### ToyV2 dataset
![alt text](./images/toy2_wide.jpg)

We generated an even more complex synthetic dataset to show the main advantage of our algorithm over other detection-based instance segmentation algorithms. The new dataset contains 25000 images for training and 1000 images each for validation and testing. Each image has resolution of 128x128 and can contain from 12 to 52 highly overlapping objects.

* You can download the ToyV2 dataset from [here](https://drive.google.com/open?id=1iUMuWZUA4wzBC3ka01jkUM5hNqU3rV_U). 
* You can test and visualize the model trained on this dataset using [this](notebooks/test_toy_v2_model.ipynb) notebook.
* You can download pretrained model from [here](https://drive.google.com/open?id=1fq72ZeVdOHM37Qv648lRVVD0VWjcD_a2).

![alt text](./images/toy_v2_comparison.jpg)


### ToyV1 dataset

We used the ToyV1 dataset for our experiments in the paper. We generated 12k samples for the toy dataset (10k for training and 2k for testing). The dataset has two versions:
* **original** contains generated samples without augmentations;
* **augmented** contains generated samples with fixed augmentations (random noise and blur).

We trained our model on the original/train part with online augmentations and tested it on the augmented/test part. The repository provides an example of testing and metric evalutation for the toy dataset.
* You can download the toy dataset from [here](https://drive.google.com/open?id=161UZrYSE_B3W3hIvs1FaXFvoFaZae4FT). 
* You can test and visualize trained model on the toy dataset using [provided](notebooks/test_toy_model.ipynb) Jupyter Notebook.
* You can download pretrained model from [here](https://drive.google.com/file/d/1n1UzzNN_9H2F71xyhKckJDr8XHDSJ-py).


### Setting up a development environment

AdaptIS is built using Python 3.6 and relies on the most recent version of PyTorch. This code was tested with PyTorch 1.3.0 and TorchVision 0.4.1. The following command installs all necessary packages:

```
pip3 install -r requirements.txt
```

Some of the inference code is written using Cython, you must compile the code before testing:
```
make -C ./adaptis/inference/cython_utils
```


### Training

Currently our implementation supports training only on single gpu, which can be selected through *gpus* flag.

You can train model for the ToyV2 dataset by the following command:
```
python3 train_toy_v2.py --batch-size=14 --workers=2 --gpus=0 --dataset-path=<toy-dataset-path>
```

You can train model for the toy dataset (original from the paper) by the following command:
```
python3 train_toy.py --batch-size=14 --workers=2 --gpus=0 --dataset-path=<toy-dataset-path>
```


### License
The code of AdaptIS is released under the MPL 2.0 License. MPL is a copyleft license that is easy to comply with. You must make the source code for any of your changes available under MPL, but you can combine the MPL software with proprietary code, as long as you keep the MPL code in separate files.


### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/1909.07829).

```
@article{adaptis2019,
  title={AdaptIS: Adaptive Instance Selection Network},
  author={Konstantin Sofiiuk, Olga Barinova, Anton Konushin},
  journal={arXiv preprint arXiv:1909.07829},
  year={2019}
}
```
