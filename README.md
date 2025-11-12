# ESCNet:Edge-Semantic Collaborative Network for Camouflaged Object Detection

This repository is the official implementation of ESCNet:Edge-Semantic Collaborative Network for Camouflaged Object Detection



## Requirements
- python == 3.11
- cuda >= 12.4

To install requirements:

```setup
pip install -r requirements.txt
```

[//]: # (>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)

## Dataset
[COD (Camouflaged Object Detection) Dataset](https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod)

## Inference result
For quick evaluation, you can download our test data:
- [Google Drive](https://drive.google.com/uc?id=1QrQ4hGuqmHpHqabPpYvB1FbN7jci1phg&export=download)

## Training

To train the model(s) in the paper, run this command:

```shell
torchrun --nproc_per_node=4 train.py --config config.yaml
```

## Evaluation
To test models, change config.yaml for different datasets:
```shell
# inference preds on different model
python test.py --config config.yaml --pred_root preds
# Then calculate metrics
python eval.py --pred_root preds --save_dir results
```

For ease of use, we create a [eval.sh](scripts%2Feval.sh) script and a use case in the form of a shell script eval.sh.
You can edit the script to change the parameters you want to test.

```shell
bash run.sh
# for eval only
bash run.sh --notrain
```

[//]: # (>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results &#40;section below&#41;.)

[//]: # (>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. )


## Citation
```
@inproceedings{ye2025escnet,
  title={ESCNet: Edge-Semantic Collaborative Network for Camouflaged Object Detection},
  author={Ye, Sheng and Chen, Xin and Zhang, Yan and Lin, Xianming and Cao, Liujuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20053--20063},
  year={2025}
}