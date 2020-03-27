# RNN-Video-Segmentation
Implementation of analyzing video data. First, CNN-based and RNN-based video features were extracted and then the RNN model was extended to implement temporal action segmentation.

### Results
In order to extract CNN-based video features, VGG16 was used with pretrained weights. To adapt the network to the given task, the last fully connected layer of the pretrained model was removed and replaced by new fully connected layers. 

The results of CNN-based and RNN-based video feature (Trimmed action recognition) are visualized using tSNE. 

In order to realize temporal action segmentation, the RNN-based video feature model was extended. The results are visualized in the following.

# Usage

### Dataset
In order to download the used dataset, a shell script is provided and can be used by the following command.

    bash ./get_dataset.sh
    
The shell script will automatically download the dataset and store the data in a folder called `face`. 

### Packages
The project is done with python3.6. For used packages, please refer to the requirments.txt for more details. All packages can be installed with the following command.

    pip3 install -r requirements.txt
    
### Training
To implement CNN based video features, the data was preprocessed by first forwarding the video frames through the pretrained VGG and then by averaging the feature maps of each video. The preprocessed data is saved and loaded for training the added layers by the following command.

    python3 train_CNN.py
    
Add how to train RNN
    
    python3 train_RNN.py
    
Add how to train s2s

    python3 train_s2s.py
    
### Testing
To test the trained models, the provided script can be run by using the following command. By running the scripts, the predicted labels are saved to a predefined folder.

**CNN-based and RNN-based video features**

    bash ./test_CNN.sh $1 $2 $3
    bash ./test_RNN.sh $1 $2 $3
-   `$1` is the folder containing the ***trimmed*** validation videos (e.g. `TrimmedVideos/video/valid/`).
-   `$2` is the path to the ground truth label file for the videos (e.g. `TrimmedVideos/label/gt_valid.csv`).
-   `$3` is the folder to which the predicted labels are saved (e.g. `./output/`)

**S2S-model**

    bash ./test_s2s.sh $1 $2
-   `$1` is the folder containing the ***full-length*** validation videos.
-   `$2` is the folder to which the predicted labels are saved (e.g. `./output/`)
