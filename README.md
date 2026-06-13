# California House Price Prediction

This is my Kaggle project for the **California House Prices** competition. The goal of this project is to predict house sale prices using real estate features such as property type, location, size, school information, tax information, and listing history.

## Project Overview

In this project, I built a MLP model using **PyTorch** to predict California house prices. The full workflow includes data loading, preprocessing, feature engineering, model training, K-fold validation, and generating a Kaggle submission file.

## Dataset

The dataset is from the Kaggle competition: [California House Prices](https://www.kaggle.com/competitions/california-house-prices/overview)

The training set contains **47,439 rows and 41 columns**, including the target column `Sold Price`.

The test set contains **31,626 rows and 40 columns**, without the target column.

## Data Preprocessing

The preprocessing steps include:

* Dropping unnecessary columns such as `Id`, `Address`, `Summary`, `Region`, `Heating features`, and `Cooling features`
* Converting date columns such as `Listed On` and `Last Sold On` into useful features
* Creating new features including:

  * `listed_year`
  * `listed_month`
  * `last_sold_year`
  * `years_since_last_sold`
  * `Bedrooms_num`
  * `Bedrooms_missing`
* Handling outliers in columns such as `Year built`, `Garage spaces`, `Total spaces`, and `Listed Price`
* Applying log transformation to skewed numerical columns
* Standardizing numerical features
* Filling missing numerical values with `0`
* One-hot encoding categorical features

After preprocessing, the final feature matrix contains **45,887 features**.

## Model

I used MLP model with 3 layers

```python
nn.Sequential(
    nn.Linear(input_features, 256),
    nn.ReLU(),
    nn.Dropout(0.05),

    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.05),

    nn.Linear(128, 1)
)
```

The model predicts the logarithm of the house price, and the final predictions are converted back to the original price scale using `np.exp()`.

## Training

The model was trained using:

* Loss function: Mean Squared Error
* Optimizer: Adam
* Learning rate: `0.001`
* Weight decay: `0.001`
* Batch size: `128`
* Epochs: `100`
* Validation method: 5-fold validation

## Validation

The 5-fold validation results were:

| Fold | Train Log RMSE | Validation Log RMSE |
| ---- | -------------: | ------------------: |
| 1    |         0.1898 |              0.2492 |
| 2    |         0.1909 |              0.2270 |
| 3    |         0.2297 |              0.2459 |
| 4    |         0.1852 |              0.2238 |
| 5    |         0.1788 |              0.2328 |

Average results:

```text
Average train log RMSE: 0.1949
Average validation log RMSE: 0.2358
```

The final model achieved a training log RMSE of around **0.2113** before generating the submission file.

## Submission Result
<img width="1154" height="140" alt="image" src="https://github.com/user-attachments/assets/f58ced1f-c195-44f3-8595-6785a253bcc9" />


