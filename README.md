<<<<<<< HEAD
# Furniture_Sales_Predictor
This is an Ecommerce project , that basically predicts the furniture sales based on the product_Title, original price, aim of this project is to give a prediction of the furniture sales , this model uses linear regression and random forest to do so.
=======
# Furniture Sales Prediction Model

This project contains a machine learning model that predicts the number of units sold for furniture products based on various features such as price, product title, and shipping information.

## Features Used

The model uses the following features to make predictions:
- Price
- Original Price
- Discount Percentage
- Title Length
- Word Count
- Presence of keywords (patio, outdoor, bedroom, living room, office, modern, wood, metal, fabric)
- Free Shipping Status

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the dataset file `ecommerce_furniture_dataset_2024.csv` in the same directory
2. Run the script:
```bash
python furniture_sales_predictor.py
```

The script will:
- Load and preprocess the data
- Train a Random Forest model
- Display model performance metrics (MSE and R2 score)
- Show feature importance
- Provide an example prediction

## Making Predictions

You can use the `predict_sales()` function to make predictions for new products:

```python
predicted_sales = predict_sales(
    product_title="Modern Patio Furniture Set with Cushions",
    price=299.99,
    original_price=399.99,
    tag_text="Free shipping"
)
print(f"Predicted sales: {predicted_sales:.2f} units")
```

## Model Details

- Algorithm: Random Forest Regressor
- Number of trees: 100
- Features are scaled using StandardScaler
- Train/Test split: 80/20
- Random state: 42 (for reproducibility)

## Performance Metrics

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- R-squared (R2) score

These metrics are displayed when running the script. 
>>>>>>> 8f8f22b (initial commit)
