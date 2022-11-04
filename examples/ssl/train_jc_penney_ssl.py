import logging

import pandas as pd
import yaml

import ludwig
from ludwig import visualize
from ludwig.api import LudwigModel
from ludwig.datasets import jc_penney_products

print("Loading Dataset")
df = jc_penney_products.load()

# Rounds to nearest .5
df["average_product_rating_quantized"] = (df["average_product_rating"] * 2).round() / 2

test_df = df[df.split == 2]

config = """
input_features:
  - name: name_title
    type: text
    encoder:
      type: rnn
  - name: description
    type: text
    encoder:
      type: rnn
  - name: sale_price
    type: number
  - name: brand
    type: category
  - name: total_number_reviews
    type: number
output_features:
   - name: average_product_rating_quantized
     type: category
     decoder:
       type: predictor
trainer:
  epochs: 10
  regularization_lambda: 0.1
  regularization_type: l1
"""

config_dict = yaml.safe_load(config)

print("Constructing Model")
model = LudwigModel(config=config_dict, logging_level=logging.INFO)

print("SSL Training Model")
train_stats, _, _ = model.train(dataset=df, experiment_name="jc_penney_ssl", model_name="example_model")

# print("Train classifier head")
# train_stats, _, _ = model.train(dataset=df, experiment_name="jc_penney_ssl", model_name="example_model")

print("Training Done, Evaluating")

# Generates predictions and performance statistics for the test set.
test_stats, predictions, output_directory = model.evaluate(
    test_df, collect_predictions=True, collect_overall_stats=True
)

visualize.confusion_matrix(
    [test_stats],
    model.training_set_metadata,
    "average_product_rating",
    top_n_classes=[len(df["average_product_rating_quantized"].unique())],
    model_names=[""],
    normalize=True,
    output_directory="./visualizations",
    file_format="png",
)

# Visualizes learning curves, which show how performance metrics changed over time during training.
visualize.learning_curves(
    train_stats, output_feature_name="average_product_rating", output_directory="./visualizations", file_format="png"
)
