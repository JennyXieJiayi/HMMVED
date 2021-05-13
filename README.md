# HMMVED: Micro-video Popularity Prediction via Multimodal Variational Information Bottleneck

This is the implementation of HMMVED for micro-video popularity prediction. The implementation of our proposed MMVED can be found in [here](https://github.com/yaochenzhu/MMVED).

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

The Xigua micro-video popularity sequence prediction dataset we collect is available [[google drive]](https://drive.google.com/open?id=1-q46LeBvi1-z7riJB28tDqk-hM5eu8g_), [[baidu]](https://pan.baidu.com/s/1FA_odoDCwPXX3khdH2GPwQ) (pin: zpwb). Descriptions of the files are as follows:

- **`resnet50.npy`**:
   (N×128). Visual features extracted by ResNet50 pre-trained on ImageNet.
 
- **`audiovgg.npy`**:
   (N×128). Aural features extracted by AudioVGG pre-trained on AudioSet.
 
- **`fudannlp.npy`**:
   (N×20). Textual features extracted by the FudanNLP toolkit.

- **`social.npy`**:
   (N×3). Social features crawled from the user attributes.

- **`len_9/target.npy`**: (N×9×2). Popularity groundtruth (0-axis) and absolute time (1-axis) at each timestep.

- **`split/0-4/{train, val, test}.txt`**: Five splits of train, val and test samples used in our paper.

### The NUS dataset

The original NUS dataset can be found [here](https://acmmm2016.wixsite.com/micro-videos), which was released with the TMALL model in this [paper](http://www.nextcenter.org/wp-content/uploads/2017/06/MicroTellsMacro.JournalNExT.pdf). The descriptions of files in the data folder in the NUS directory are as follows:

- **`vid.txt`**:  The ids of the micro-videos that we were able to download successfully at the time of our experiment.

- **`split/0-4/{train, val, test}.txt`**: Five splits of the dataset we used in our paper.

## Examples to run the Codes

The basic usage of the codes for training and testing HMMVED model on both Xigua and NUS dataset is as follows:

- **For training**: 

	```python train.py --lambd [LAMBDA] --split [SPLIT]```
- **For testing**:

	```python predict.py --model_path [PATH_TO_MODEL] --test_mods [VATS]```

For more advanced arguments, run the code with --help argument.

### **If you find our codes and dataset helpful, please kindly cite the following papers. Thanks!**

    @article{hmmved-2021,
        title={Micro-video Popularity Prediction via Multimodal Variational Information Bottleneck},
        author={Xie, Jiayi and Zhu, Yaochen and Chen, Zhenzhong},
        year={2021}
    }

	@inproceedings{mmved-www2020,
	  title={A Multimodal Variational Encoder-Decoder Framework for Micro-video Popularity Prediction},
	  author={Xie, Jiayi and Zhu, Yaochen and Zhang, Zhibin and Peng, Jian and Yi, Jing and Hu, Yaosi and Liu, Hongyi and Chen, Zhenzhong},
	  booktitle={The Web Conference},
	  year={2020},
	  pages = {2542–2548},
	}
