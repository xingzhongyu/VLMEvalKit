import json
import warnings

import torch


def measure_flops_once(generate_fn, message, dataset, save_path):
    """Run generate_fn on a single sample under torch.profiler to count TFLOPs.

    Falls back to a plain generate_fn call if profiling fails, so the returned
    response is always valid regardless of whether measurement succeeded.

    Args:
        generate_fn: callable — model.generate(message=..., dataset=...)
        message: the message argument to pass through
        dataset: the dataset argument to pass through
        save_path: path to write the JSON result (skipped on failure)

    Returns:
        (response, tflops_or_None)
    """
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, with_flops=True, record_shapes=True) as prof:
            response = generate_fn(message=message, dataset=dataset)

        total_flops = sum(e.flops for e in prof.key_averages())
        tflops = total_flops / 1e12
        result = {'total_flops': total_flops, 'tflops': round(tflops, 6)}
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'[FLOPs] {tflops:.4f} TFLOPs (note: flash-attn kernels may be excluded) → {save_path}')
        return response, tflops

    except Exception as e:
        warnings.warn(f'[FLOPs] Measurement failed ({type(e).__name__}: {e}), skipping.')
        response = generate_fn(message=message, dataset=dataset)
        return response, None
