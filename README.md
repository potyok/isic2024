# Deep Learning in Practice with Python and LUA - BMEVITMAV45 2024/25/1

The aim of the repository is to present our solution to the [ISIC2024 kaggle challenge](https://www.kaggle.com/competitions/isic-2024-challenge) implemented in the project homework. In the project, we have reviewed some literature and we plan to carry out this exercise based on our experience in other subjects.

## Project homework - ISIC 2024 - Skin Cancer Detection with 3D-TBP

| Group     | Details                                      |
| --------- | -------------------------------------------- |
| Team name | IsIcTeam2024                                 |
| Members   | Attila Nemes (B6RYIK),</br> Csaba Potyok (OZNVQ4),</br> Peter Arany (U4VQHM) |

## Project description

**Task description**: In this competition, we'll develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). The image quality resembles close-up smartphone photos, which are regularly submitted for telehealth purposes.

**Our approach**: We plan to explore classification by image alone, as well as classification by images and their associated metadata. We plan to use a pre-trained net for image processing and then fine tune it. We will use hyperparameter optimization to achieve the best model and find a solution for the unbalanced dataset. We plan to use [Gradio](https://www.gradio.app/) to build an AI service from it.

## Related works

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
- src/isic_2024_data_prepare.ipynb: Contains the data prepare solution (data loading, train/val/test splitting and visulazing data)
- src/FFD.ipynb: Contains a custom image deformation technique implementation

## Run
Load the isic_2024_data_prepare.ipynb and run all cells, if the original dataset is not available, you can download it from [here](https://drive.google.com/file/d/11RQvjL61Ss2w2R1kKWHa2vdaJLdIGup6/view?usp=drive_link).
