import os
import pathlib
import numpy as np
import h5py
from PIL import Image
import warnings
import pydicom
import nibabel
import scipy
import json
import yaml


def get_base_name(file_name):
    base_name = pathlib.Path(file_name)
    for _ in base_name.suffixes:
        base_name = base_name.with_suffix('')

    return base_name.name


def load_json(file_path):
    with open(file_path, 'r') as f:
        temp = f.read()
    temp_json = json.loads(temp)
    return temp_json


def load_yaml(file_path):
    with open(file_path, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file


def get_ext(file_name):
    base_name = pathlib.Path(file_name)
    ext = ''.join(base_name.suffixes)
    return ext


def load_array(input_file, data_key='data', convert2gray=False, sel_slice=None):
    ext = get_ext(input_file)
    base_name = get_base_name(input_file)
    if 'h5' in ext:
        with h5py.File(input_file, 'r') as h5_obj:
            if data_key in h5_obj.keys():
                if sel_slice is None:
                    loaded_array = np.array(h5_obj[data_key])
                elif sel_slice == 'mid':
                    mid_slice = h5_obj[data_key].shape[0] // 2
                    loaded_array = np.array(h5_obj[data_key][mid_slice])
                else:
                    loaded_array = np.array(h5_obj[data_key][sel_slice])
            else:
                loaded_array = None
                warn_str = f"Unknown key {data_key}. Accepting keys {h5_obj.keys()}"
                warnings.warn(warn_str)
    elif 'npy' in ext:
        loaded_array = np.load(input_file)
    # Scary om met IM_ te checken...
    elif ('dicom' in ext) or ('dcm' in ext) or (base_name.startswith('IM_')):
        loaded_array = pydicom.read_file(input_file).pixel_array
    elif 'nii' in ext:
        loaded_array = nibabel.load(input_file).get_fdata()
    elif ('png' in ext) or ('jpg' in ext) or ('jpeg' in ext):
        pillow_obj = Image.open(input_file)
        if convert2gray:
            loaded_array = np.array(pillow_obj.convert('LA'))
        else:
            loaded_array = np.array(pillow_obj.convert("RGB"))
    elif 'mat' in ext:
        mat_obj = scipy.io.loadmat(input_file)
        # Filter out all the protected names
        if data_key in mat_obj.keys():
            loaded_array = mat_obj[data_key]
        else:
            print('Unknown key in mat obj: ', mat_obj.keys())
            print("File name ", input_file)
            print("Returning matlab object")
            loaded_array = mat_obj
    else:
        print('Unknown extension ', input_file, ext)
        loaded_array = np.array(None)

    return loaded_array
