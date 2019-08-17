from helper_functions.src.config_parsing.from_py import py2dict


def test_py2dict():
    result_dict = py2dict('tests/data/temp_config.py')

    target_dict = dict(a=1, b=3, c=range(10))

    shared_items = {k: target_dict[k] for k in target_dict if k in result_dict and target_dict[k] == result_dict[k]}

    assert len(shared_items) == len(result_dict)
