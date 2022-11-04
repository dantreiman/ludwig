import logging

import pandas as pd
import yaml

import ludwig
from ludwig.api import LudwigModel
from ludwig.datasets import jc_penney_products

print("Loading Dataset")
df = jc_penney_products.load()

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
   - name: average_product_rating
     type: category
trainer:
  epochs: 10
"""

config_dict = yaml.safe_load(config)

print("Constructing Model")
model = LudwigModel(config=config_dict, logging_level=logging.INFO)

print("Training Model")
train_stats, _, _ = model.train(dataset=df, experiment_name="jc_penney_ssl", model_name="example_model")

print("Training Done, Evaluating")

# Generates predictions and performance statistics for the test set.
test_stats, predictions, output_directory = model.evaluate(
    test_df, collect_predictions=True, collect_overall_stats=True
)

confusion_matrix(
    [test_stats],
    model.training_set_metadata,
    "average_product_rating",
    top_n_classes=[2],
    model_names=[""],
    normalize=True,
    output_directory="./visualizations",
    file_format="png",
)

# Visualizes learning curves, which show how performance metrics changed over time during training.
learning_curves(
    train_stats, output_feature_name="average_product_rating", output_directory="./visualizations", file_format="png"
)
