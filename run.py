import copy
import logging
import os.path as op
from pprint import pformat
from src.tools.common import execute_func
from src.tools.common import init_logging
from src.tools.common import print_frame_info
from src.tools.common import parse_general_args
from src.tools.common import dict_update_nested_dict
from src.tools.qd_pytorch import load_latest_parameters
from src.tools.common import dict_ensure_path_key_converted


def create_pipeline(kwargs):
    pipeline_type = kwargs.get('pipeline_type')
    info = copy.deepcopy(pipeline_type)
    assert 'param' not in info
    info['param'] = kwargs
    return execute_func(info)


def load_pipeline(**kwargs):

    kwargs = copy.deepcopy(kwargs)
    kwargs_f = load_latest_parameters(op.join('output', kwargs['full_expid']))
    dict_update_nested_dict(kwargs_f, kwargs)
    return create_pipeline(kwargs_f)


# evaluation pipeline
def pipeline_eval_multi(param, all_test_data, **kwargs):
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        dict_ensure_path_key_converted(test_data_info)
        dict_update_nested_dict(curr_param, test_data_info)
        pip = load_pipeline(**curr_param)
        # we should check here instead of before for-loop since we can alter
        # the value of max_iter to just evaluate the intermediate model or take
        # the intermediate model as the final model
        if not pip.is_train_finished():
            logging.info('the model specified by the following is not ready\n{}'.format(
                pformat(param)))
            return
        pip.ensure_predict()
        pip.ensure_evaluate()


# Training and evaluation pipeline
def pipeline_train_eval_multi(all_test_data, param, **kwargs):

    print_frame_info()
    init_logging()
    curr_param = copy.deepcopy(param)
    if len(all_test_data) > 0:
        dict_update_nested_dict(curr_param, all_test_data[0])
    pip = create_pipeline(curr_param)

    # training script
    pip.ensure_train()

    full_expid = pip.full_expid
    param['full_expid'] = full_expid
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        dict_ensure_path_key_converted(test_data_info)
        dict_update_nested_dict(curr_param, test_data_info)
        pip = load_pipeline(**curr_param)
        pip.ensure_predict()
        pip.ensure_evaluate()

    if param.get('monitor_after'):
        for test_data_info in all_test_data:
            curr_param = copy.deepcopy(param)
            dict_ensure_path_key_converted(test_data_info)
            dict_update_nested_dict(curr_param, test_data_info)
            pip = load_pipeline(**curr_param)
            pip.monitor_train()

    return full_expid


if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

