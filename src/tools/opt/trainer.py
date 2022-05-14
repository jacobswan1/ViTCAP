import time
import torch
import datetime
import logging
from src.tools.common import get_mpi_rank, get_mpi_size
from src.tools.logger import MetricLogger
from src.tools.torch_common import to

from src.tools.common import try_once
@try_once
def try_save_intermediate_snapshot(checkpointer, name, arguments):
    checkpointer.save(name, **arguments)


def get_num_image(d):
    if isinstance(d, dict):
        if 'image' in d:
            return len(d['image'])
        elif 'targets' in d:
            # mask-rcnn -> dict
            return len(d['targets'])
        elif 'input_ids' in d:
            # vl
            return len(d['input_ids'])
    elif isinstance(d, tuple) or isinstance(d, list):
        return get_num_image(d[0])
    elif isinstance(d, torch.Tensor):
        return len(d)
    else:
        raise NotImplementedError


def do_train_dict(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    log_step=20,
    data_partition=1,
    explicit_average_grad=False,
    no_update=False,
    ema=None,
    use_amp=False,
    gradient_clip=None,
    model_sub_name_fn=None,):

    if model_sub_name_fn is None:
        model_sub_name_fn = lambda i: "model_iter_{:07d}".format(i)
    logging.info("Start training")
    from src.tools.logger import SmoothedValue
    meters = MetricLogger(delimiter="  ", meter_creator=lambda:
                          SmoothedValue(log_step))
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    from src.tools.common import print_frame_info
    print_frame_info()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    debug = False
    if debug:
        from src.tools.torch_common import init_random_seed
        torch.set_deterministic(False)
        init_random_seed(99)
        from src.qd.layers.forward_pass_feature_cache import ForwardPassFeatureCache
        model = ForwardPassFeatureCache(model)

    # check if checkpoint param is changed
    # init_param = torch.load('./output/vit_base_patch16_384_model_iter_0000000.pt')['model']
    # now_param = model.state_dict()
    # keys = now_param.keys()
    # for key in keys:
    #     if key in init_param:
    #         print(key, (init_param[key] - now_param[key]).sum())
    #     else:
    #         print(key, 'not initialized.')

    # print('Before enumeration.')
    # print(model.state_dict().keys())
    # print(model.state_dict()['module.image_encoder.module.patch_embed.proj.weight'])
    # print(model.state_dict()['module.module.bert.encoder.blocks.0.norm1.bias'])
    # print(model.state_dict()['module.module.bert.encoder.blocks.0.norm1.weight'])

    # save an initial weights for re-check in future
    checkpointer.save(model_sub_name_fn(arguments['iteration']), **arguments)

    for iteration, dict_data in enumerate(data_loader, start_iter):
        iteration += 1
        arguments["iteration"] = iteration

        before_to_device = time.time()
        data_time = before_to_device - end
        dict_data = to(dict_data, device)
        before_gpu = time.time()
        time_to_device = before_gpu - before_to_device

        if not no_update:
            optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(dict_data)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            if gradient_clip:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                meters.update(total_norm=total_norm)
            if not no_update:
                scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(dict_data)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            if gradient_clip:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                meters.update(total_norm=total_norm)
            if not no_update:
                optimizer.step()

        if debug:
            model.sumarize_feature()
            import ipdb;ipdb.set_trace(context=15)

        if not use_amp and losses != losses:
            logging.info('NaN encountered!')
            checkpointer.save("NaN_context_{}".format(get_mpi_rank()), **arguments)
            raise RuntimeError('NaN encountered!')

        meters.update(loss=losses, **loss_dict)

        if not no_update:
            scheduler.step()

        time_gpu = time.time() - before_gpu

        if ema is not None:
            ema.update(model)

        if (iteration % log_step) == 0 or iteration == max_iter:
            speed = get_mpi_size() * log_step * get_num_image(dict_data) / (time.time() - log_start)
            if hasattr(meters, 'time'):
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_string = 'Unknown'

            logging.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        'speed: {speed:.1f} images/sec',
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    speed=speed,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            log_start = time.time()
        if iteration % checkpoint_period == 0:
            # with blobfuse, saving could fail with unknown reason. Instead of
            # saving and crashing, we do a best-effort manner.
            before_save = time.time()
            try_save_intermediate_snapshot(
                checkpointer,
                model_sub_name_fn(iteration),
                arguments)
            meters.update(save_time=(time.time() - before_save))

        batch_time = time.time() - end
        end = time.time()

        if iteration > start_iter + 5:
            # we will skip the first few iterations since the time cost
            # evaluation for those are not good
            meters.update(time=batch_time, data=data_time)
            meters.update(to_device=time_to_device)
            meters.update(time_gpu=time_gpu)

    checkpointer.save("model_final", **arguments)
    if get_mpi_rank() > 0:
        old_value = checkpointer.save_to_disk
        checkpointer.save_to_disk = True
        checkpointer.save("model_final_{}".format(get_mpi_rank()), **arguments)
        checkpointer.save_to_disk = old_value

    checkpointer.save(model_sub_name_fn(arguments['iteration']), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logging.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (1 if max_iter == 0 else
                                                   max_iter)
        )
    )


