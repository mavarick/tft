# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""即干预模型v5版本的特征
"""

import data_formatters.base
import data_formatters.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class TFTv6(GenericDataFormatter):
  """
  """

  _column_definition = [
      ('biz_sale_qty', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('time_idx', DataTypes.REAL_VALUED, InputTypes.TIME),
      
      ('poi_sku_id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('city_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('poi_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('base_sku_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      
      ('category1_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('category2_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('category3_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('category4_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

      #('traffic_index', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('traffic_index', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('is_weekend', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('is_holiday', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('icon_day_type', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_icon_id', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_temperature', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('qpf', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('is_seckill', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('promos_discount', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('sell_price', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._time_steps = self.get_fixed_params()['total_time_steps']

  def split_data(self, df, valid_boundary=43, test_boundary=50, his_days=21, pred_days=3):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    index = df['time_idx']
    train = df.loc[index <= valid_boundary+pred_days]
    #valid = df.loc[(index >= valid_boundary - his_days +1) & (index <= valid_boundary+pred_days)]
    valid = df.loc[0:0]
    test = df.loc[index >= test_boundary-his_days +1]

    self.set_scalers(train)

    print("train:{}, valid:{}, test:{}".format(train.shape, valid.shape, test.shape))
    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Initialise scaler caches
    self._real_scalers = {}
    self._target_scaler = {}
    identifiers = []
    for identifier, sliced in df.groupby(id_column):

      if len(sliced) >= self._time_steps:

        data = sliced[real_inputs].values
        targets = sliced[[target_column]].values
        self._real_scalers[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(data)

        self._target_scaler[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(targets)
      identifiers.append(identifier)

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

    # Extract identifiers in case required
    self.identifiers = identifiers

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    print("read_inputs:", real_inputs)
    print("categorical_inputs:", categorical_inputs)
    # Transform real inputs per entity
    df_list = []
    for identifier, sliced in df.groupby(id_col):

      # Filter out any trajectories that are too short
      #print(identifier, len(sliced), self._time_steps)
      if len(sliced) >= self._time_steps:
        sliced_copy = sliced.copy()
        sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
            sliced_copy[real_inputs].values)
        df_list.append(sliced_copy)

    if not df_list:
        return pd.DataFrame()

    output = pd.concat(df_list, axis=0)

    # Format categorical inputs
    for col in categorical_inputs:
      #string_df = df[col].apply(str)
      string_df = output[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """

    if self._target_scaler is None:
      raise ValueError('Scalers have not been set!')

    column_names = predictions.columns

    df_list = []
    for identifier, sliced in predictions.groupby('identifier'):
      sliced_copy = sliced.copy()
      target_scaler = self._target_scaler[identifier]

      for col in column_names:
        if col not in {'forecast_time', 'identifier'}:
          sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        # 表示
        'total_time_steps': 24,
        'num_encoder_steps': 21,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 160,
        'learning_rate': 0.001,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    raise NotImplementedError
    #return 450000, 50000
