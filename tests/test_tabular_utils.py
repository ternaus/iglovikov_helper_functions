from collections import defaultdict

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis.extra.numpy import arrays as h_arrays
from hypothesis.strategies import characters as h_char
from hypothesis.strategies import floats as h_float

from iglovikov_helper_functions.utils.tabular_utils import (
    CyclicEncoder,
    LabelEncoderUnseen,
    GeneralEncoder,
    MinMaxScaler,
)

MIN_VALUE = -11
MAX_VALUE = 17
ARRAY_SHAPE = 3


@given(x=h_arrays(dtype=float, shape=ARRAY_SHAPE, elements=h_float(-MIN_VALUE, MAX_VALUE)))
def test_cyclic_day_hours(x):
    amplitude = MAX_VALUE - MIN_VALUE

    encoder = CyclicEncoder(amplitude)

    transformed = encoder.fit_transform(x)

    assert transformed.shape == (len(x), 2)

    encoder2 = CyclicEncoder(amplitude)
    encoder2.fit(x)
    transformed2 = encoder2.transform(x)

    assert transformed.shape == (len(x), 2)

    assert np.array_equal(transformed, transformed2)

    reverse_transformed = encoder.inverse_transform(transformed)

    assert x.shape == reverse_transformed.shape

    assert np.allclose(x, reverse_transformed)


@given(x=h_arrays(dtype=float, shape=ARRAY_SHAPE, elements=h_float(-MIN_VALUE, MAX_VALUE)))
def test_min_max_scaler(x):
    feature_range = (-1, 1)

    encoder = MinMaxScaler(feature_range)

    transformed = encoder.fit_transform(x)

    assert transformed.shape == x.shape

    encoder2 = MinMaxScaler(feature_range)
    encoder2.fit(x)

    transformed2 = encoder2.transform(x.T)

    assert transformed2.shape == x.T.shape

    assert np.array_equal(transformed, transformed2.T)

    reverse_transformed = encoder.inverse_transform(transformed)

    assert x.shape == reverse_transformed.shape

    assert np.allclose(x, reverse_transformed)


@given(
    x=h_arrays(
        dtype="object", shape=ARRAY_SHAPE, elements=h_char(whitelist_categories=["Lu", "Ll", "Lt", "Lm", "Lo", "Nl"])
    )
)
def test_label_encoder_unseen(x):
    e = LabelEncoderUnseen()

    e.fit(x)

    transformed_1 = e.transform(x)

    transformed = e.fit_transform(x)

    assert np.all(transformed == transformed_1), f"{transformed} {transformed_1}"

    assert np.all(x == e.inverse_transform(transformed))

    transformed_2 = e.transform(list(x) + ["qwe"])

    assert np.all(transformed_2 == list(transformed) + list(e.transform([np.nan])))


@given(numerical=h_arrays(dtype=float, shape=(ARRAY_SHAPE, 2), elements=h_float(-MIN_VALUE, MAX_VALUE)))
def test_encoder_numerical(numerical):
    columns_map = defaultdict(list)

    result = {}

    category_type = "numerical"

    joined = {}

    for i in range(numerical.shape[1]):
        column_name = f"{category_type} {i}"
        result[column_name] = numerical[:, i]

        columns_map[category_type] += [column_name]

        joined[column_name] = [column_name + "1", column_name + "2"]

        result[column_name + "1"] = numerical[:, i] / 2
        result[column_name + "2"] = numerical[:, i] + 10

    columns_map["joined_encoders"] = joined

    df = pd.DataFrame(result)

    encoder = GeneralEncoder(columns_map)

    transformed = encoder.fit_transform(df)

    assert set(transformed.keys()) == {category_type}

    assert set(columns_map.keys()) == set(transformed.keys())

    inverse_transform = encoder.inverse_transform(transformed)

    assert list(df.columns) == list(inverse_transform.columns)

    for i in range(numerical.shape[1]):
        column_name = f"{category_type} {i}"
        assert encoder.encoders[column_name + "1"] == encoder.encoders[column_name]
        assert encoder.encoders[column_name + "2"] == encoder.encoders[column_name]

    for column_name in df.columns:
        assert np.allclose(df[column_name].values, inverse_transform[column_name].values)

    assert np.allclose(df.values, inverse_transform.values), f"{df.values - inverse_transform.values}"


@given(cyclical=h_arrays(dtype=float, shape=(ARRAY_SHAPE, 5), elements=h_float(-MIN_VALUE, MAX_VALUE)))
def test_encoder_cyclical(cyclical):
    columns_map = defaultdict(list)

    result = {}

    category_type = "cyclical"

    for i in range(cyclical.shape[1]):
        column_name = f"{category_type} {i}"
        result[column_name] = cyclical[:, i]

        element = (column_name, MAX_VALUE - MIN_VALUE)

        columns_map[category_type] += [element]

    df = pd.DataFrame(result)

    encoder = GeneralEncoder(columns_map)

    transformed = encoder.fit_transform(df)

    assert set(transformed.keys()) == {category_type}
    assert set(columns_map.keys()) == set(transformed.keys())

    inverse_transform = encoder.inverse_transform(transformed)

    assert inverse_transform.shape == df.shape
    assert np.all(inverse_transform.columns == df.columns)
    assert np.all(df.dtypes == inverse_transform.dtypes)

    assert np.allclose(df.values, inverse_transform.values)


@given(
    categorical=h_arrays(
        dtype="object",
        shape=(ARRAY_SHAPE, 7),
        elements=h_char(whitelist_categories=["Lu", "Ll", "Lt", "Lm", "Lo", "Nl"]),
    )
)
def test_encoder_categorical(categorical):
    columns_map = defaultdict(list)

    result = {}

    category_type = "categorical"

    joined_encoders = {}

    for i in range(categorical.shape[1]):
        column_name = f"{category_type} {i}"
        result[column_name] = categorical[:, i]

        element = column_name

        columns_map[category_type] += [element]

        joined_encoders[column_name] = [column_name + "1", column_name + "2"]

        result[column_name + "1"] = np.random.permutation(categorical[:, i])
        result[column_name + "2"] = np.random.permutation(
            np.concatenate([categorical[:5, i], ["Vladimir" + x for x in categorical[5:i]]])
        )

    columns_map["joined_encoders"] = joined_encoders

    df = pd.DataFrame(result)

    encoder = GeneralEncoder(columns_map)

    transformed = encoder.fit_transform(df)

    # We add "unknow category" => number of categories in encoder should be +1 to the ones in df
    for column in columns_map["categorical"]:
        assert df[column].nunique() + 1 == len(encoder.encoders[column].set_classes)

    assert set(transformed.keys()) == {category_type}
    assert set(transformed.keys()).intersection(columns_map.keys()) == set(transformed.keys())

    inverse_transform = encoder.inverse_transform(transformed)

    assert df.equals(inverse_transform)


@given(
    numerical=h_arrays(dtype=float, shape=(ARRAY_SHAPE, 5), elements=h_float(-MIN_VALUE, MAX_VALUE)),
    cyclical=h_arrays(dtype=float, shape=(ARRAY_SHAPE, 2), elements=h_float(-MIN_VALUE, MAX_VALUE)),
    categorical=h_arrays(
        dtype="object",
        shape=(ARRAY_SHAPE, 3),
        elements=h_char(whitelist_categories=["Lu", "Ll", "Lt", "Lm", "Lo", "Nl"]),
    ),
)
def test_encoder(numerical, cyclical, categorical):
    columns_map = defaultdict(list)

    result = {}

    joined_encoders = {}

    for feature, category_type in [(numerical, "numerical"), (cyclical, "cyclical"), (categorical, "categorical")]:
        for i in range(feature.shape[1]):
            column_name = f"{category_type} {i}"
            result[column_name] = feature[:, i]

            if category_type == "numerical":
                joined_encoders[column_name] = [column_name + "1", column_name + "2"]
                result[column_name + "1"] = feature[:, i] / 2
                result[column_name + "2"] = feature[:, i] + 10
                element = column_name
            elif category_type == "categorical":
                joined_encoders[column_name] = [column_name + "1", column_name + "2"]
                result[column_name + "1"] = np.random.permutation(categorical[:, i])
                result[column_name + "2"] = np.random.permutation(
                    np.concatenate([categorical[:5, i], ["Vladimir" + x for x in categorical[5:i]]])
                )
                element = column_name
            else:
                element = (column_name, MAX_VALUE - MIN_VALUE)

            columns_map[category_type] += [element]

    columns_map["joined_encoders"] = joined_encoders

    df = pd.DataFrame(result)

    encoder = GeneralEncoder(columns_map)

    transformed = encoder.fit_transform(df)

    # We add "unknow category" => number of categories in encoder should be +1 to the ones in df
    for column in columns_map["categorical"]:
        assert df[column].nunique() + 1 == len(encoder.encoders[column].set_classes)

    for category_type in encoder.category_types:
        for encoded in transformed[category_type]:
            if category_type == "cyclical":
                assert (df.shape[0], 2) == encoded.shape, f"{category_type} {encoded.shape}"
            else:
                assert (df.shape[0],) == encoded.shape, f"{category_type} {encoded.shape} {df.values.shape}"

    assert set(columns_map.keys()) == set(transformed.keys()), f"{transformed.keys()} {columns_map.keys()}"

    inverse_transform = encoder.inverse_transform(transformed)

    assert inverse_transform.shape == df.shape
    assert np.all(inverse_transform.columns == df.columns)
    assert np.all(df.dtypes == inverse_transform.dtypes)


@given(
    numerical=h_arrays(dtype=float, shape=(ARRAY_SHAPE, 5), elements=h_float(-MIN_VALUE, MAX_VALUE)),
    cyclical=h_arrays(dtype=float, shape=(ARRAY_SHAPE, 2), elements=h_float(-MIN_VALUE, MAX_VALUE)),
    categorical=h_arrays(
        dtype="object",
        shape=(ARRAY_SHAPE, 3),
        elements=h_char(whitelist_categories=["Lu", "Ll", "Lt", "Lm", "Lo", "Nl"]),
    ),
)
def test_encoder_on_nonfull(numerical, cyclical, categorical):
    """The test checks that if the dataframe has more columns than columns map - extra columns are ignored."""
    columns_map = defaultdict(list)

    result = {}

    for feature, category_type in [(numerical, "numerical"), (cyclical, "cyclical"), (categorical, "categorical")]:
        for i in range(feature.shape[1]):
            column_name = f"{category_type} {i}"
            result[column_name] = feature[:, i]

            if category_type == "numerical":
                result[column_name] = feature[:, i]
                element = column_name
            elif category_type == "categorical":
                result[column_name] = feature[:, i]
                element = column_name
            else:
                element = (column_name, MAX_VALUE - MIN_VALUE)

            columns_map[category_type] += [element]

    for category_type in columns_map:
        columns_map[category_type].pop()

    df = pd.DataFrame(result)

    encoder = GeneralEncoder(columns_map)

    transformed = encoder.fit_transform(df)

    # We add "unknow category" => number of categories in encoder should be +1 to the ones in df
    for column in columns_map["categorical"]:
        assert df[column].nunique() + 1 == len(encoder.encoders[column].set_classes)

    for category_type in encoder.category_types:
        for encoded in transformed[category_type]:
            if category_type == "cyclical":
                assert (df.shape[0], 2) == encoded.shape, f"{category_type} {encoded.shape}"
            else:
                assert (df.shape[0],) == encoded.shape, f"{category_type} {encoded.shape} {df.values.shape}"

    assert set(columns_map.keys()) == set(transformed.keys()), f"{transformed.keys()} {columns_map.keys()}"

    inverse_transform = encoder.inverse_transform(transformed)

    assert len(df.columns) == len(inverse_transform.columns) + len(columns_map)
