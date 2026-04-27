# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import torch

# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):

        if torch.jit.is_tracing() or torch.compiler.is_compiling():
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            pos = torch.stack([grid_y, grid_x], dim=-1)  # (h, w, 2)
            pos = pos.reshape(1, -1, 2)
            return pos.expand(b, -1, 2).clone()
        else:
            if not (h, w) in self.cache_positions:
                x = torch.arange(w, device=device)
                y = torch.arange(h, device=device)
                self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
            pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
            return pos
