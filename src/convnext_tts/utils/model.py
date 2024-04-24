import torch
import torch.nn.functional as F


def sequence_mask(length):
    max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    device = duration.device
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    path = path * mask
    return path


def slice_segments(x, start_indices, segment_size):
    ret = torch.zeros(x.size(0), x.size(1), segment_size, device=x.device)
    for i in range(x.size(0)):
        start = start_indices[i]
        end = start + segment_size
        ret[i, :, :] = x[i, :, start:end]
    return ret


def rand_slice_segments(x, lengths, segment_size):
    B, _, T = x.size()
    if lengths is None:
        lengths = T
    start_max = lengths - segment_size + 1
    idx_start = (torch.rand([B]).to(device=x.device) * start_max).long()
    idx_start = idx_start.clamp(min=0)
    ret = slice_segments(x, idx_start, segment_size)
    return ret, idx_start


def to_log_scale(x: torch.Tensor):
    x[x != 0] = torch.log(x[x != 0])
    return x
