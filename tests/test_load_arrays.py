from cp_shock_project.data.load_arrays import load_arrays, validate_shapes

from tests.conftest import write_synthetic_data


def test_load_arrays(tmp_path):
    data_dir, *_ = write_synthetic_data(tmp_path)
    arrays = load_arrays(data_dir, mmap=True)
    validate_shapes(arrays.X_train, arrays.Y_train)
    assert arrays.X_train.shape[1] == 9
    assert arrays.Y_train.shape[1] == 4
