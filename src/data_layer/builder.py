from torch.utils.data.dataloader import default_collate
import torch

def collate_fn(batch):
    # this function is designed to support any customized type and to be compatible
    # with the default collate function
    ele = batch[0]

    # to handle the rect returning, check the type
    if isinstance(ele, dict):
        re = {}
        for key in ele:
            if type(ele[key]) == torch.Tensor or type(ele[key]) == int:
                re[key] = collate_fn([d[key] for d in batch])
            else:
                re[key] = [d[key] if key in d else [] for d in batch]
        del ele, batch
        return re

    else:
        if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
            if not all(b.shape == batch[0].shape for b in batch[1:]):
                assert all(len(b.shape) == len(batch[0].shape) for b in batch[1:])
                shape = torch.tensor([b.shape for b in batch])
                max_shape = tuple(shape.max(dim=0)[0].tolist())
                batch2 = []
                for b in batch:
                    if any(c < m for c, m in zip(b.shape, max_shape)):
                        b2 = torch.zeros(max_shape, dtype=b.dtype, device=b.device)
                        if b.dim() == 1:
                            b2[:b.shape[0]] = b
                        elif b.dim() == 2:
                            b2[:b.shape[0], :b.shape[1]] = b
                        else:
                            raise NotImplementedError
                        b = b2
                    batch2.append(b)
                batch = batch2
        return default_collate(batch)
