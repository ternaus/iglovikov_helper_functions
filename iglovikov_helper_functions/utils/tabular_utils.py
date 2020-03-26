from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from addict import Dict as Addict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class CyclicEncoder:
    """
    Class encodes feature x to (cos(2 pi x / amplitude), sin(2 * pi x / amplitude)
    """

    params: Dict[str, float] = {}

    def __init__(self, amplitude):
        self.params["amplitude"] = amplitude

    def fit(self, x):
        pass

    def fit_transform(self, x: Union[np.array, list]) -> np.array:
        return self.transform(x)

    def get_params(self) -> dict:
        return self.params

    def set_params(self, params: dict) -> None:
        self.params = params

    def inverse_transform(self, x: np.array) -> np.array:
        sin_component = x[:, 1]
        cos_component = x[:, 0]
        angle = np.arctan(sin_component / cos_component) + np.pi / 2 * (1 - np.sign(cos_component))
        return angle * self.params["amplitude"] / (2 * np.pi)

    def transform(self, x: Union[list, np.array]) -> np.array:
        amplitude = self.params["amplitude"]
        argument = 2 * np.pi * x / amplitude
        cos_component = np.cos(argument)
        sin_component = np.sin(argument)

        result = np.vstack([cos_component, sin_component]).T

        return result


class LabelEncoderUnseen(LabelEncoder):
    """Extension of sklearn.preprocessing.LabelEncoder
    that can work with unseen labels.
    All unseen labels are mapped to 'uknown_class'
    """

    set_classes: set

    def __init__(self, unknown_class="unknown"):
        super().__init__()
        self.unknown_class = unknown_class

    def fit(self, x: Union[np.array, list]) -> None:
        super().fit([self.unknown_class] + list(x))
        self.set_classes = set(self.classes_)

    def transform(self, x: Union[np.array, list]) -> np.array:
        if isinstance(x, pd.core.series.Series):
            x = x.values

        for i in range(len(x)):
            if x[i] not in self.set_classes:
                x[i] = self.unknown_class

        return super().transform(x)

    def fit_transform(self, x: Union[np.array, list]) -> np.array:
        self.fit(x)
        return self.transform(x)


class GeneralEncoder:
    """Generate encoders for tabular data transforms.

    numerical: (x - min(x)) / (max(x) - min(x)) i.e. MinMaxScaler
    categorical: OneHotEncoding(x) i.e. OneHotEncoder
    cyclical: x => (cos(2 * pi * x / amplitude), sin(2 * pi * x / amplitude))

    args:
        df: pandas dataframe to transform.
        columns_map: dictionary of the type:
            {
                "numerical": [column1, column2, ...],
                "categorical": [column3, column4, ...],
                "cyclical": [(column5, cyclic_amplitude5), (column6, cyclic_amplitude6), ...]
            }

    return:
        dictionary:
            {
                "numerical": [column1, column2, ...],
                "categorical": [column3, column4, ...],
                "cyclical": [(cos(column_5) sin(column_5), (cos(column_6) sin(column_6), ...]
            }

    """

    encoders: Addict
    column2type: Dict[str, str] = {}
    columns: List = []
    columns_map: Dict[str, list]

    def __init__(self, columns_map):
        self.columns_map = columns_map

        self.type2encoder = {
            "numerical": (MinMaxScaler, {"feature_range": (-1, 1)}),
            "categorical": (LabelEncoderUnseen, {}),
            "cyclical": (CyclicEncoder, {}),
        }

        if set(self.type2encoder.keys()).intersection(columns_map.keys()) != set(columns_map.keys()):
            raise ValueError(f"Wrong column names in columns_map {columns_map}.")

    def fit(self, df: pd.DataFrame) -> None:
        self.columns = list(df.columns)
        self.encoders = Addict()

        for category_type, columns in self.columns_map.items():
            for column_id, column in enumerate(columns):
                encoder_class, parameters = self.type2encoder[category_type]

                if category_type == "cyclical":
                    parameters["amplitude"] = self.columns_map[category_type][column_id][1]

                    column = self.columns_map[category_type][column_id][0]

                encoder = encoder_class(**parameters)

                x = df[column].values

                if category_type == "numerical":
                    x = x.reshape(-1, 1)

                encoder.fit(x)

                self.encoders[category_type][column] = encoder
                self.column2type[column] = category_type

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, list]:
        self.fit(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> Dict[str, list]:
        if not self.encoders:
            raise ValueError(f"Perform fit before calling transform.")

        result: defaultdict = defaultdict(list)

        for column in df.columns:
            if column not in self.column2type:
                continue

            category_type = self.column2type[column]

            encoder = self.encoders[category_type][column]

            x = df[column].values

            if category_type == "numerical":
                encoded = encoder.transform(x.reshape(-1, 1))[:, 0]
            else:
                encoded = encoder.transform(x)

            result[category_type] += [encoded]

        return result

    def get_params(self):
        return self.encoders, self.column2type

    def inverse_transform(self, feature_dict):
        result = {}
        for category_type, encoded_columns in feature_dict.items():
            if category_type not in self.columns_map:
                raise ValueError(f"{category_type}")
            for i, x in enumerate(encoded_columns):
                if category_type == "cyclical":
                    column_name = self.columns_map[category_type][i][0]
                else:
                    column_name = self.columns_map[category_type][i]

                encoder = self.encoders[category_type][column_name]

                if category_type == "numerical":
                    decoded_column = encoder.inverse_transform(x.reshape(-1, 1))[:, 0]
                else:
                    decoded_column = encoder.inverse_transform(x)

                result[column_name] = decoded_column

        return pd.DataFrame(result)[self.columns]
