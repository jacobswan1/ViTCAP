import os
import os.path as op
import re


from src.tools.common import load_from_yaml_file, write_to_yaml_file
from src.qd.mask.utils.comm import is_main_process


def parse_yaml_file(yaml_file):
    r = re.compile('.*fea.*lab.*.yaml')
    temp = op.basename(yaml_file).split('.')
    split_name = temp[0]
    if r.match(yaml_file) is not None:
        fea_folder = '.'.join(temp[temp.index('fea') + 1 : temp.index('lab')])
        lab_folder = '.'.join(temp[temp.index('lab') + 1 : -1])
    else:
        fea_folder, lab_folder = None, None
    return split_name, fea_folder, lab_folder


def check_yaml_file(yaml_file):
    # check yaml file, generate if possible
    if not op.isfile(yaml_file):
        try:
            split_name, fea_folder, lab_folder = parse_yaml_file(yaml_file)
            if fea_folder and lab_folder:
                base_yaml_file = op.join(op.dirname(yaml_file), split_name + '.yaml')
                if op.isfile(base_yaml_file):
                    data = load_from_yaml_file(base_yaml_file)
                    data['feature'] = op.join(fea_folder, split_name + '.feature.tsv')
                    data['label'] = op.join(lab_folder, split_name + '.label.tsv')
                    assert op.isfile(op.join(op.dirname(base_yaml_file), data['feature']))
                    assert op.isfile(op.join(op.dirname(base_yaml_file), data['label']))
                    if is_main_process():
                        write_to_yaml_file(data, yaml_file)
                        print("generate yaml file: {}".format(yaml_file))
        except:
            raise ValueError("yaml file: {} does not exist and cannot create it".format(yaml_file))


