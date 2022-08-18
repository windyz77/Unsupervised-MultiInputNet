# Unsupervised-Light-Field-Depth-Estimation

![image](https://github.com/windyz77/Unsupervised-Light-Field-Depth-Estimation/blob/master/net.png)


train Requirements
====================================
    pip install -r requirements.txt

dataset
====================================
We used the HCI 4D LF benchmark for training and evaluation. Please refer to the [benchmark website](https://lightfield-analysis.uni-konstanz.de/) for details.

train 
====================================
    python multidepth_main.py

    --model_name MultiDepth 

    --data_path /your/path/to/full_data

    --total_epoch 500

    --batch_size 2

    --train_txt_path /your/path/train_val_txt/4dlf_7x7star_train.txt

    --output_path=./output/check_code_oldcode

test
====================================
    change train_mode = False

    python multidepth_main.py

    --model_name MultiDepth 

    --data_path /your/path/to/full_data

    --val_txt_path' /your/path/train_val_txt/4dlf_7x7star_val.txt

    --checkpoint_path ./output/check_code1/MultiDepth/your model

Unsupervised learning of light field depth estimation with spatial and angular consistencies
====================================
    @article{article,
        author = {Lin, Lili and Li, Qiujian and Gao, Bin and Yan, Yuxiang and Zhou, Wenhui and Kuruoglu, Ercan},
        year = {2022},
        month = {06},
        pages = {},
        title = {Unsupervised learning of light field depth estimation with spatial and angular consistencies},
        volume = {501},
        journal = {Neurocomputing},
        doi = {10.1016/j.neucom.2022.06.011}
    }
