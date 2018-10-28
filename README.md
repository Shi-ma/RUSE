# RUSE
RUSE: Regression model Using Sentence Embeddings for Automatic Machine Translation Evaluation  
We submitted it to [WMT18 Metrics Shared Task](http://www.statmt.org/wmt18/metrics-task.html).  
Result paper: http://www.statmt.org/wmt18/pdf/WMT078.pdf

## Dependencies
This code is written in python. Dependencies include:
* Python 2.7.9
* [Chainer](https://chainer.org/) 4.2.0
* cupy==4.2.0

## Prepare Sentence Encoders
1. Please git clone our metric and git clone following sentence encoders under a directory named `RUSE/encoder_models/`.
2. lease make the following sentence encoders available refering to each github or TensorFlow hub.
* InferSent \[Conneau et al., 2017\] (https://github.com/facebookresearch/InferSent)
* Quick-Thought \[Logeswaran and Lee, 2018] (https://github.com/lajanugen/S2V)
* Unicersal Sentence Encoder \[Cer et al., 2018\] (https://alpha.tfhub.dev/google/universal-sentence-encoder-large/2)
But, After downloading Quick-Thought's pre-trained model from https://bit.ly/2uttm2j, please set it under a directory named `RUSE/encoder_models/S2V/pretrained_models/`.

## Download our pre-trained models
Please download our pre-trained model from https://drive.google.com/open?id=1AyQMMEdFKmrc3fnPuG2rGrGHePCPSHCa and set a directory named `RUSE/models`.

## Use our metric
Please prepare a file in which the reference and the translated text are entered in a tab-delimited fashion for each line　and set it (.tsv file) under a directory named `RUSE/data`.  
  
To create each sentence encoder's feature (.npz file).
If you want to use all sentence encoder's features, you need to create IS, QT and USE features in under script.
```
cd scripts
bash tsv2npz.sh <SR_MODEL> <TSV_PATH>
```
```
* SR_MODEL  # IS, QT or USE
* TSV_PATH  # Path to your prepared tab-delimited data
```
To make scores with RUSE.
```
cd scripts
bash make_score.sh <SR_MODEL> <NPZ_DIR>
```
```
* SR_MODEL  # IS, QT, USE or IS_QT_USE
* NPZ_DIR   # Path to npz directory
```
