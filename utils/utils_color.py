from torch import Tensor
import torch
import complextorch


class ColorModelConverter:

    def __init__(self, device):

        self.device = device

    def convert(self, img):

        return complextorch.CVTensor(r=img.real, i=img.imag).to(self.device), self.target.to(self.device)

    def convert_hsv_ihsv(self, inp:Tensor, target:Tensor) -> (Tensor, Tensor):

        h, s, v = inp.unbind(dim=1)

        h_star = v + 1j * s

        s_star = s * h + 1j * v

        v_star = s * torch.exp(1j * h)

        self.target = target

        ihsv = torch.stack((h_star, s_star, v_star), dim=-3)

        input, target = self.convert(ihsv)

        return input, target

    def convert_rgb_irgb(self, inp:Tensor, target:Tensor) -> (Tensor, Tensor):

        r, g, b = inp.unbind(dim=1)

        r_star = r + 1j * g

        g_star = g + 1j * b

        self.target = target

        irgb = torch.stack((r_star, g_star), dim=-3)

        input, target = self.convert(irgb)

        return input, target

    def convert_lab_ilab(self, inp:Tensor, target:Tensor) -> (Tensor, Tensor):

        l, a, b = inp.unbind(dim=1)

        l_star = l

        a_star = a + 1j * b

        ilab = torch.stack((l_star, a_star), dim=-3)

        self.target = target

        input, target = self.convert(ilab)

        return input, target



