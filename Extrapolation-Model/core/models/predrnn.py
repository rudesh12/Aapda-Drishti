__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch, _, channels, height, width = frames.shape

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        x_gen = torch.zeros_like(frames[:, 0])  # Initialize x_gen to prevent undefined variable error

        for t in range(self.configs.total_length - 1):
            # Ensure mask_true_expanded is inside the loop where 't' is defined
            if t >= self.configs.input_length:
                mask_true_expanded = mask_true[:, t - self.configs.input_length].unsqueeze(1)  # Correct dimension

          # Reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    # Extract the correct time step from mask_true
                    mask_true_expanded = mask_true[:, t - self.configs.input_length].unsqueeze(1)  # Shape: [batch, 1, height, width]

                    # Fix: Expand along the channel dimension (192 channels)
                    mask_true_expanded = mask_true_expanded.expand(-1, frames.shape[2], -1, -1.-1)  # Shape: [batch, 192, height, width]

                    # Debugging print
                    print(f"✅ mask_true_expanded shape: {mask_true_expanded.shape}, frames[:, t] shape: {frames[:, t].shape}")

                    # Apply the mask correctly
                    net = mask_true_expanded * frames[:, t] + (1 - mask_true_expanded) * x_gen

            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    mask_true_expanded = mask_true[:, t - self.configs.input_length].unsqueeze(1)  # Shape: [batch, 1, height, width]
                    
                    # Fix: Expand dynamically to match frame channels
                    mask_true_expanded = mask_true_expanded.expand(-1, frames.shape[2], -1, -1,-1)  # Shape: [batch, 192, height, width]

                    # Debugging print
                    print(f"✅ mask_true_expanded shape: {mask_true_expanded.shape}, frames[:, t] shape: {frames[:, t].shape}")

                    # Apply the mask correctly
                    net = mask_true_expanded * frames[:, t] + (1 - mask_true_expanded) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])  # Update x_gen for next time step
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=1).permute(0, 1, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss
