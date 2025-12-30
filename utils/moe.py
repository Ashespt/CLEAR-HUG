import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LEAD_NAME=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
GROUP1 = np.array([0,1,2])
GROUP2 = np.array([3,4,5])
GROUP3 = np.array([6,7,8,9,10,11])

class SimpleExpert(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (B, T, D)
        return self.net(x)

class HierarchicalMoE(nn.Module):
    def __init__(self, input_dim=768,output_dim=512):
        super().__init__()
        # Level 1: 3 experts for 3 groups
        output_dim=input_dim
        self.level1_experts = nn.ModuleList([SimpleExpert(input_dim,output_dim) for _ in range(3)])
        # Level 2: 3 experts for pairwise combination of level1 outputs
        self.level2_experts = nn.ModuleList([SimpleExpert(output_dim,output_dim) for _ in range(3)])
        # Level 3: 1 expert to combine all level2 outputs
        self.level3_expert = SimpleExpert(output_dim,output_dim)

    def forward(self, x):
        # x: (B, 12, 768)
        B, T, D = x.shape
        assert T == 12
        # Level 1 groups: [0-3], [4-7], [8-11]
        # l1_groups = [x[:, i*4:(i+1)*4, :] for i in range(3)]  # List of 3 tensors: each (B, 4, D)
        l1_groups = [x[:,GROUP1,:],x[:,GROUP2,:],x[:,GROUP3,:]]  # List of 3 tensors: each (B, 4, D)
        # Apply Level 1 MoE (simple expert for each group)
        l1_outputs = [self.level1_experts[i](group).mean(dim=1) for i, group in enumerate(l1_groups)]  # list of (B, D)

        # Level 2 groups: (0+1), (1+2), (0+2)
        l2_input_pairs = [
            torch.stack([l1_outputs[0], l1_outputs[1]], dim=1),
            torch.stack([l1_outputs[1], l1_outputs[2]], dim=1),
            torch.stack([l1_outputs[0], l1_outputs[2]], dim=1)
        ]

        l2_outputs = [self.level2_experts[i](pair).mean(dim=1) for i, pair in enumerate(l2_input_pairs)]  # list of (B, D)

        # Level 3 group: combine all level 2 outputs
        l3_input = torch.stack(l2_outputs, dim=1)  # (B, 3, D)
        l3_output = self.level3_expert(l3_input).mean(dim=1)  # (B, D)
        l2_outputs.extend([l3_output])
        l1_outputs.extend(l2_outputs)
        return l1_outputs
        # return {
        #     'group1': l1_outputs,  # List of 3 x (B, D)
        #     'group2': l2_outputs,  # List of 3 x (B, D)
        #     'group3': l3_output     # (B, D)
        # }