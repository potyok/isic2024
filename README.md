# Deep Learning in Practice with Python and LUA - BMEVITMAV45 2024/25/1

The aim of the repository is to present our solution to the [ISIC2024 kaggle challenge](https://www.kaggle.com/competitions/isic-2024-challenge) implemented in the project homework. In the project, we have reviewed some literature and we plan to carry out this exercise based on our experience in other subjects.

## Project homework - ISIC 2024 - Skin Cancer Detection with 3D-TBP

| Group     | Details                                                                      |
| --------- | ---------------------------------------------------------------------------- |
| Team name | IsIcTeam2024                                                                 |
| Members   | Attila Nemes (B6RYIK),</br> Csaba Potyok (OZNVQ4),</br> Peter Arany (U4VQHM) |

## Project description

**Task description**: In this competition, we'll develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). The image quality resembles close-up smartphone photos, which are regularly submitted for telehealth purposes.

**Our approach**: We plan to explore classification by image alone, as well as classification by images and their associated metadata. We plan to use a pre-trained net for image processing and then fine tune it. We will use hyperparameter optimization to achieve the best model and find a solution for the unbalanced dataset. We plan to use [Gradio](https://www.gradio.app/) to build an AI service from it.

> [!NOTE]
> The detailed documentation is available in [PDF](https://github.com/potyok/isic2024/blob/main/docs/ISIC_2024_documentation.pdf) and [docx](https://github.com/potyok/isic2024/blob/main/docs/ISIC_2024_documentation.docx) format in the **docs** folder.

## Related works

During the project, we reviewed several articles that dealt with issues and solutions related to the task.

### Related papers:

- [5 Effective Ways to Handle Imbalanced Data in Machine Learning](https://machinelearningmastery.com/5-effective-ways-to-handle-imbalanced-data-in-machine-learning/)
- [A multiple combined method for rebalancing medical data with class imbalances](https://doi.org/10.1016/j.compbiomed.2021.104527)
- [Skin Cancer Detection Using Deep Learning—A Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10252190/)
- [Detection and Classification of Melanoma Skin Cancer Using Image Processing Technique](https://pmc.ncbi.nlm.nih.gov/articles/PMC10649387/)
- [Dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/articles/nature21056)
- [Skin Cancer Classification With Deep Learning: A Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9327733/)
- [Squeeze-MNet: Precise Skin Cancer Detection Model for Low Computing IoT Devices Using Transfer Learning](https://www.mdpi.com/2072-6694/15/1/12)
- [Free-form deformation of solid geometric models](https://dl.acm.org/doi/10.1145/15886.15903)
- [Analysis of dermoscopy images by using ABCD rule for early detection of skin cancer](https://www.sciencedirect.com/science/article/pii/S2666285X21000017)

## Files

> [!NOTE]
> The Dropbox link can break if too many downloads are made, so it's a good idea to download the ZIP once and use it locally, or upload it to another file share.

- **src/isic_2024_data_prepare.ipynb**: Contains the data prepare solution (data loading, train/val/test splitting and visulazing data)
- **src/FFD.ipynb**: Contains a custom image deformation technique implementation
- **src/isic_2024_baseline.ipynb**: Contains the data loading, train and test pipeline with metrics
- **src/isic_2024_alexnet_cross_validation.ipynb**: Contains the AlexNet-based model and cross-validation process.
- **src/isic_2024_alexnet_hyperopt_relu**: Contains AlexNet-base model with ReLU activations in classifier layers.
- **src/isic_2024_alexnet_hyperopt_prelu**: Contains AlexNet-base model with PReLU activations in classifier layers.
- **src/isic_2024_combined_cross_validation.ipynb**: Contains combined model's (AlexNet + MobileNetv3 with selector net) cross-validation process.
- **src/custom_cross_validation.ipynb**: Contains complex model's (image and tabular data processor) cross-validation process.
- **src/mobilenet_v3_cross_validation.ipynb**: Contains MobileNetv3-based model's cross-validation process.
- **src/mobilenet_v3.ipynb**: Contains MobileNetv3-based model's hyperparameter optimization process.
- **src/pretrained.ipynb**: Contains an example project for using pretrained model.
- **src/isic_2024_m4.ipynb**: Contains a simple CNN model with another model for tabular data and combine them result for classify samples.
- **src/isic_2024_mobilnet_tabular.ipynb**: Contains combined model (MobileNetv3 with with selector net) 
- **src/isic_2024_mobilnet_alexnet_tabular.ipynb**: Contains complex model (MobileNetv3 with AlexNet with selector net) 
- **src/app.py**: Contains config and setup for Gradio interface.
- **src/custom_transform.py**: Contains the hair removal algorithm for Gradio app.
- **src/inference_module.py**: Contains the module that can load trained model.
- **gradio.dockerfile**: Contains the description of Docker image for using Gradio.

In **results** folder you can find the hyperparameter optimization results.
In **images** folder you can find example images to try our Gradio AI service. In **diagram** you can find the source of model architectures' diagrams and a few diagram from wandb.

## Run

Load the isic_2024_data_prepare.ipynb and run all cells, if the original dataset is not available, you can download it from [here](https://drive.google.com/file/d/11RQvjL61Ss2w2R1kKWHa2vdaJLdIGup6/view?usp=drive_link).

**We mostly used Jupyter Notebooks, so running them is very easy on a cell-by-cell basis.**

Load the isic_2024_baseline.ipynb and run all cells, to load data and train/test the baseline model. At the data loading you can set the hair_remove flag on data_module to use hair removal algorithm.

Load the isic_2024_alexnet_cross_validation.ipynb, isic_2024_combined_cross_validation.ipynb, isic_2024_custom_cross_validation.ipynb or isic_2024_mobilenet_v3_cross_validation.ipynb notebooks to train and cross validate models and test with test dataset.

### Gradio

To use Gradio, you need to save a trained model and place it in a location where it can be mounted to the Docker container.

Build container:

```
docker build -f gradio.dockerfile -t [IMAGE_NAME] .
```

Example for running container:

```
docker run --gpus all --rm -p 7860:7860 -v ./model:/app/model [IMAGE_NAME]
```

Due to its size, we managed to create a solution that runs on Huggingface, which is available [here](https://huggingface.co/spaces/Nemes2000/isic_2024).
