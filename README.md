# HMMVED: Micro-video Popularity Prediction via Multimodal Variational Information Bottleneck

This is the implementation of [HMMVED](https://ieeexplore.ieee.org/abstract/document/9576573/) for micro-video popularity prediction. The implementation of our proposed MMVED can be found in [here](https://github.com/yaochenzhu/MMVED).

It includes two parts:

- Micro-video popularity regression on NUS dataset.
- Micro-video popularity sequence prediction on Xigua dataset.

Each part contains everything required to train or test the corresponding HMMVED model. 

For the Xigua datset we collect, we release the data as well.

## Environment

- python == 3.6.5
- numpy == 1.16.1
- tensorflow == 1.13.1
- tensorflow-probability == 0.6.0

## Datasets

### The Xigua dataset

The Xigua micro-video popularity sequence prediction dataset we collect is available at [[google drive]](https://drive.google.com/drive/folders/1Q2iTMKiFSO1uVw4Io4uTwSc2uONO6iFc?usp=sharing). For usage, download, unzip the data folder and put them in the `popularity_sequence_prediction` directory. Descriptions of the files are as follows:

- **`resnet50.npy`**:
   (N×128). Visual features extracted by ResNet50 pre-trained on ImageNet.
 
- **`audiovgg.npy`**:
   (N×128). Aural features extracted by AudioVGG pre-trained on AudioSet.
 
- **`fudannlp.npy`**:
   (N×20). Textual features extracted by the FudanNLP toolkit.

- **`user.npy`**:
   (N×4). Social features crawled from the user attributes, where the first dimension is users' ID.

- **`len_9/target.npy`**: (N×9×2). Popularity groundtruth (0-axis) and absolute time (1-axis) at each timestep.

- **`split/0-4/{train, val, test}.txt`**: Five splits of train, val and test samples used in our paper.

### The NUS dataset

The original NUS dataset can be found [here](https://acmmm2016.wixsite.com/micro-videos), which was released with the TMALL model in this [paper](http://www.nextcenter.org/wp-content/uploads/2017/06/MicroTellsMacro.JournalNExT.pdf). The descriptions of files in the data folder in the NUS directory are as follows:

- **`vid.txt`**:  The ids of the micro-videos that we were able to download successfully at the time of our experiment.

- **`split/0-4/{train, val, test}.txt`**: Five splits of the dataset we used in our paper.

## Examples to run the Codes

An example to run the codes for training and testing HMMVED model can be found in `popularity_sequence_prediction/run_example.py`.

For more advanced arguments, run the `train.py` and `predict.py` with --help argument.

### **If you find our codes and dataset helpful, please kindly cite the following papers. Thanks!**
> @article{hmmved-tmm2021,  
	author={Xie, Jiayi and Zhu, Yaochen and Chen, Zhenzhong},  
	journal={IEEE Transactions on Multimedia},   
	title={Micro-video Popularity Prediction via Multimodal Variational Information Bottleneck},   
	year={2021},  
	volume={},  
	number={},  
	pages={1-1},  
	doi={10.1109/TMM.2021.3120537}}
