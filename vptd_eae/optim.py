"""PyTorch AdamW with bounded update temporaries, including after resume."""
from torch.optim import AdamW


def _single_tensor_updates(optimizer):
    for group in optimizer.param_groups:
        group['foreach'] = False
        group['fused'] = False


class LowMemoryAdamW(AdamW):
    """Use the official single-tensor kernel without resetting Adam moments.

    Foreach creates approximately one extra parameter-sized set of temporaries.
    Loading a checkpoint also restores kernel flags, so reapply them afterward.
    https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **{**kwargs, 'foreach': False, 'fused': False})
        _single_tensor_updates(self)
        self.register_load_state_dict_post_hook(_single_tensor_updates)
