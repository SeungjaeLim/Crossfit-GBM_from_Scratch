# CrossFit Performance Predictor Using Gradient Boosting Models from Scratch (Only Numpy)

Welcome to my project! I have developed a CrossFit Performance Predictor using various Gradient Boosting Models, built from scratch using only numpy. I have worked with a dataset from the CrossFit Games, implementing models including Simple Gradient Boost, XGBoost, CatBoost, and LightGBM. Furthermore, I compared these custom implementations with their respective library models to analyze their accuracy and training time.

The project begins with implementing gradient boost models and testing them with diabetes and California housing datasets. I then perform an exploratory data analysis (EDA) and preprocessing on the CrossFit Games dataset and adapt my custom models to this data. Finally, I conduct an evaluation of my models in comparison to the equivalent library models.

For an in-depth understanding and detailed walkthrough of the project, please refer to my Jupyter notebook, `report.ipynb`. This report covers every stage of the project, providing comprehensive insights into the development and evaluation of my models.

This project is a demonstration of the power of Gradient Boosting Models in predictive tasks and the effectiveness of implementing these complex models from scratch. I hope you find it informative and insightful.

## Package versions
![numpy](https://img.shields.io/badge/numpy-1.25.0-blue)
![pandas](https://img.shields.io/badge/pandas-1.5.3-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-blue)
![seaborn](https://img.shields.io/badge/seaborn-0.12.2-blue)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-blue)
![catboost](https://img.shields.io/badge/catboost-1.2-blue)
![xgboost](https://img.shields.io/badge/xgboost-1.7.3-blue)
![lightgbm](https://img.shields.io/badge/lightgbm-3.3.5-blue)

## Installation

To use this repository, first clone it using the following command:

```bash
git clone https://github.com/SeungjaeLim/Crossfit-GBM_from_Scratch.git
```

Then, navigate to the directory of the cloned repository:

```bash
cd Crossfit-GBM_from_Scratch
```

Next, install the necessary dependencies using conda:

```bash
conda env create -f environment.yaml
```

## Usage

To run the training script, use the following command:

```bash
python ./train/train.py
```

After the model has been trained, you can run the inference script with the following command:

```bash
python ./inference/infer.py
```

## Evaluation

| Fran | Helen | Grace | Filthy50 | FGoneBad |
|-------|-----------------------|------|------|------|
|![image](https://github.com/SeungjaeLim/Crossfit-GBM_from_Scratch/assets/74184274/f6cd2961-085a-4cee-93eb-5a23c16156b9)| ![image](https://github.com/SeungjaeLim/Crossfit-GBM_from_Scratch/assets/74184274/ff6c84cf-3cf8-4187-8629-23279f602d80) | ![image](https://github.com/SeungjaeLim/Crossfit-GBM_from_Scratch/assets/74184274/1bbb1804-3435-4014-a326-31788131032d) | ![image](https://github.com/SeungjaeLim/Crossfit-GBM_from_Scratch/assets/74184274/887c8a25-2b6e-4a16-937b-582e24509c91) | ![image](https://github.com/SeungjaeLim/Crossfit-GBM_from_Scratch/assets/74184274/e7647d0a-a704-4734-882a-179a959c67c5) |

![image](https://github.com/SeungjaeLim/Crossfit-GBM_from_Scratch/assets/74184274/fa45b40f-3da3-4c0b-977e-71dde6162b10)



For a more comprehensive evaluation and detailed results of the implemented models, please refer to our Jupyter notebook `report.ipynb`. In this notebook, you'll find in-depth analysis, additional visualizations, and explanations for each step of the project. It serves as a report that covers the entire process of our project, from data exploration and preprocessing to model training, evaluation, and conclusion. 

You can view `report.ipynb` directly on GitHub or download it to run it on your local Jupyter notebook environment. Remember to ensure that you have all the necessary packages installed to avoid any run-time issues.

## Reviews

| Grade | Summary of the Project | Pros | Cons |
|-------|-----------------------|------|------|
| 100.00 | Based on the theoretical background, gradient boost models were actually implemented using numpy. Performed comparisons of each model on various datasets, from the diabetes, california housing dataset to the crossfit dataset, and obtained quite successful accuracy. | 1. There are many visualizations from the introduction to the main code, so it was easy to understand the results and distribution. 2. Data analysis was conducted from various angles using various datasets and models to help understanding. 3. It is impressive that the models were actually implemented using numpy using a theoretical background. | 1. It was unfortunate that the number of data in diabetes or crossfit set was slightly insufficient. 2. As an advantage, but also as a disadvantage, too many distribution visualizations, such as some correlation plots, were not considered for context understanding. 3. What function each code has or what role it plays is written less. If you write down those points, I think it will help me to read more. |
| 100.00 | Introduction on ensemble methods, and their applications on crossfit performance prediction | 1. Describing raw data with graphs, and analyzing its feature via EDA. 2. Introducing and comparing various ML methods. 3. Well-organized structure as a real paper. | I'm not such an expert to figure out its weaknesses, but I hope there would be some more explanations on how each algorithm works and what each of the graph implies for us ML beginners. |
| 100.00 | This notebook goes through crossfit performance predictor with gradient boost. At first, notebook describe about background knowledge to understand how this algorithm works. Then, implement some models based on gradient boost, train, and get result. Finally, comparing those result and conclude the notebook with analysis. | 1. Background knowledge is very high quality with a lot of images that help understanding the concept. 2. A lot of analysis about the predicted result. 3. Topic is very impressive that gets machine learning inspiration from his own crossfit gym. | 1. There are no mathematical notations about introducing the algorithm. Using LaTex in colab to express mathematical notation can make a better notebook. 2. For the diabetes dataset, XGBoost shows the best performance but is worst in the California housing dataset implemented from the library. I want to know the reason why XGBoost's performance is bad in the California housing dataset, but there is nothing about it. 3. In LightGBM classifier, there is background knowledge about how a lot of categorical features makes LightGBM's performance stronger, so I think notating the number of categorical features of each dataset can make a better notebook to understand the feature of LightGBM classifier. |
| 100.00 | I am highly impressed with the student's project on gradient boosting models and their implementation in the context of CrossFit workout outcome prediction. The project showcased a deep understanding of the models, advanced implementation skills using Numpy, and an ability to compare and analyze the performance of different implementations. | With its exemplary performance and insightful conclusions. |  |
