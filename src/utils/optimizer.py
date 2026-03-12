import torch
import torch.optim as optim
from torch.optim import Optimizer


class MuonWithAdamW(Optimizer):
    """
    A composite optimizer that applies torch.optim.Muon to 2D parameters
    and torch.optim.AdamW to all other parameters (1D biases, embeddings, norms, etc.).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        lr_ratio: float = 0.015,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_weight_decay: float = 0.0,
    ):
        # 1. Consolidate params into a list (handling generators)
        # Note: We support param_groups if passed, but typically this is a flat list
        param_list = list(params)

        muon_params = []
        adam_params = []

        # 2. Sort parameters by dimensionality
        for p in param_list:
            # Handle param_groups logic if the user passed a list of dicts
            if isinstance(p, dict):
                # If complex grouping is needed, this logic needs expansion.
                # For this specific interface, we flatten straightforwardly.
                for sub_p in p["params"]:
                    if sub_p.ndim == 2:
                        muon_params.append(sub_p)
                    else:
                        adam_params.append(sub_p)
            else:
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        self.muon_opt = None
        self.adam_opt = None

        # 3. Initialize Internal Optimizers
        # We collect the param_groups from these to pass to the super class later
        all_groups = []

        if muon_params:
            self.muon_opt = optim.Muon(
                muon_params,
                lr=lr,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_steps=ns_steps,
            )
            all_groups.extend(self.muon_opt.param_groups)

        if adam_params:
            self.adam_opt = optim.AdamW(
                adam_params,
                lr=lr * lr_ratio,
                betas=adam_betas,
                weight_decay=adam_weight_decay,
            )
            all_groups.extend(self.adam_opt.param_groups)

        # 4. Initialize Base Optimizer
        # Pass the collected groups so PyTorch (and schedulers) see the parameters.
        # We pass a dummy defaults dict because the groups are already fully populated.
        super().__init__(all_groups, defaults={})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon_opt:
            self.muon_opt.step()

        if self.adam_opt:
            self.adam_opt.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self.muon_opt:
            self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.adam_opt:
            self.adam_opt.zero_grad(set_to_none=set_to_none)

    # We must override state_dict/load_state_dict to manage the split internal states
    def state_dict(self):
        return {
            "muon_opt": self.muon_opt.state_dict() if self.muon_opt else None,
            "adam_opt": self.adam_opt.state_dict() if self.adam_opt else None,
            # We don't save base class state because we don't use it directly
        }

    def load_state_dict(self, state_dict):
        if self.muon_opt and state_dict.get("muon_opt"):
            self.muon_opt.load_state_dict(state_dict["muon_opt"])
        if self.adam_opt and state_dict.get("adam_opt"):
            self.adam_opt.load_state_dict(state_dict["adam_opt"])
