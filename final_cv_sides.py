from final_cv_pipeline import single_run, write_result
import numpy as np

def lr_range_test(tenfold_iter=10, valid_fold_num=0):
    start_lr = 1e-8
    gamma = 10**(1/tenfold_iter)
    result = single_run(
        valid_fold_num=valid_fold_num,
        model='esm-t33',
        model_kwargs={
            'freeze_esm': False,
            'freeze_layer_count': 30,
        },
        pipeline_kwargs={
            'optimizer_kwargs': {
                'lr': start_lr,
            },
            'scheduler': 'exponential',
            'scheduler_kwargs': {
                'gamma': gamma,
            }
        },
        patience=10**9,
        max_epoch=int(tenfold_iter * 7.5),
        gpu=1,
    )
    
    for record in result['full_record']:
        result['record'] = record
        write_result(
            model_key=f'esm-t33-lrrange',
            valid_fold=valid_fold_num,
            result=result,
            write_inference=False,
            result_file='result_cv/result_cv_sides.csv',
        )
        

def scheduler_test(base_lr, max_lr, step_size_up, step_size_down, fold=0, gpu=1):
    print(f'base_lr: {base_lr}, max_lr: {max_lr}, step_size_up: {step_size_up}, step_size_down: {step_size_down}')
    result = single_run(
        valid_fold_num=fold,
        model='lm-gearnet',
        model_kwargs={
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30, 
        },
        pipeline_kwargs={
            'scheduler': 'cyclic',
            'scheduler_kwargs': {
                'base_lr': base_lr,
                'max_lr': max_lr,
                'step_size_up': step_size_up,
                'step_size_down': step_size_down,
                'cycle_momentum': False
            }
        },
        patience=step_size_up + step_size_down + 1,
        max_epoch=step_size_up + step_size_down,
        batch_size=8,
        gpu=gpu,
    )
    for record in result['full_record']:
        result['record'] = record
        write_result(
            model_key=f'lm-gearnet-sch',
            valid_fold=fold,
            result=result,
            write_inference=False,
            result_file='result_cv/result_cv_sch.csv',
            additional_record={
                'base_lr': base_lr,
                'max_lr': max_lr,
                'step_size_up': step_size_up,
                'step_size_down': step_size_down,
            }
        )

if __name__ == '__main__':
    # lr_range_test(tenfold_iter=5, valid_fold_num=0)
    # lr_range_test(tenfold_iter=10, valid_fold_num=1)
    lr_range_test(tenfold_iter=20, valid_fold_num=2)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--base_lrs', type=float, nargs='+', default=[1e-3])
    # parser.add_argument('--max_lrs', type=float, nargs='+', default=[1e-2])
    # parser.add_argument('--step_size_ups', type=int, nargs='+', default=[20])
    # parser.add_argument('--step_size_downs', type=int, nargs='+', default=[-1])
    # parser.add_argument('--gpu', type=int, default=0)
    # args = parser.parse_args()
    # base_lrs = args.base_lrs
    # max_lrs = args.max_lrs
    # step_size_ups = args.step_size_ups
    # step_size_downs = args.step_size_downs
    # gpu = args.gpu
    # for base_lr in base_lrs:
    #     for max_lr in max_lrs:
    #         for step_size_up in step_size_ups:
    #             for step_size_down in step_size_downs:
    #                 if step_size_down == -1:
    #                     step_size_down = step_size_up
    #                 scheduler_test(
    #                     base_lr=base_lr,
    #                     max_lr=max_lr, 
    #                     step_size_up=step_size_up, 
    #                     step_size_down=step_size_down,
    #                     fold=0,
    #                     gpu=gpu
    #                 )