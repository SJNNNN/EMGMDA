# EMGMDA
EMGMDA: enhancing miRNA - disease associations prediction with morphology images and multi-view graph learning

## Requirements
  * python==3.7
  * dgl==0.6.1
  * networkx==2.5
  * numpy==1.16.6
  * scikit-learn==0.20.3
  * pytorch==1.5.0
  * tqdm==4.15.0

## File
### data
  The data files needed to run the model, which contain HMDDv2.0 and HMDDv3.2.
  * disease semantic similarity matrix 1.txt and disease semantic similarity matrix 2.txt: Two kinds of disease semantic similarity
  * miRNA functional similarity matrix.txt: MiRNA functional similarity
  * known disease-miRNA association number.txt:Validated mirNA-disease associations
  * disease number.txt: Disease id and name
  * miRNA number.txt: MiRNA id and name
  * img_fused_features_HMDDv2.0.npy:Pathological image features
  * img_fused_features_HMDDv3.2.npy:Pathological image features
### code
  * eval.py: The startup code of the program
  * mulitmodel_train.py: Train the model
  * multimodal.py: Structure of the model
  * mulitmodel_utils.py: Methods of data processing
 
## Usage
  * download code and data
  * execute ```python eval.py```
