import torch

from einops import rearrange
from geomloss import SamplesLoss
from torch.nn.modules.loss import _Loss


class BrenierW2Loss(_Loss):

    def __init__(self, loss="sinkhorn", backend="online", p=2, blur=0.01):
        """
        Computes the W2 distance between two images using the Brenier potential.
        This method looks at the images as essentially a point cloud and computes the w2 cost.
        :param loss: sinkhorn for wasserstein distance
        :param backend: always set to online or tensorized if using small images
        :param p: the power of the distance, default is 2
        :param blur: set to 1 / max(H, W) where H and W are the height and width of the image. This is the finest resolution of the image.
        """
        super().__init__()
        self.loss = SamplesLoss(loss=loss, p=p, blur=blur, backend=backend)

    def forward(self, source, target):
        # the source and target are images of shape (B, C, H, W)

        h, w = source.shape[-2:]
        source = rearrange(source, "b c h w -> (b c) h w")
        target = rearrange(target, "b c h w -> (b c) h w")

        # get the coordinates of the non-zero indices as (B, N, 2)
        source_coords = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=source.device),
            torch.arange(w, dtype=torch.float32, device=source.device),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)
        source_coords = source_coords.unsqueeze(0).expand(source.shape[0], -1, -1)
        source_coords.requires_grad_(True)

        target_coords = source_coords.detach().clone()

        # get the weights of the non-zero indices as (B, N)
        source_weights = rearrange(source, "b h w -> b (h w)")
        target_weights = rearrange(target, "b h w -> b (h w)")

        source_weights = source_weights / source_weights.sum(dim=-1, keepdim=True)
        target_weights = target_weights / target_weights.sum(dim=-1, keepdim=True)

        return self.loss(source_weights, source_coords, target_weights, target_coords)


def create_circle(_translation, inner_rad=3., val=0.5):
    image_of_circle = torch.zeros(32, 32)
    y, x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing='xy'
    )
    circle_inner = (x - 24 - _translation) ** 2 + (y - 24 - _translation) ** 2 <= inner_rad ** 2
    circle_outer = ((x - 24 - _translation) ** 2 + (y - 24 - _translation) ** 2 <= 3 ** 2) & (
            (x - 24 - _translation) ** 2 + (y - 24 - _translation) ** 2 > inner_rad ** 2)
    image_of_circle[circle_inner] = 1
    image_of_circle[circle_outer] = val
    circle_inner = (x - 8 - _translation) ** 2 + (y - 8 - _translation) ** 2 <= inner_rad ** 2
    circle_outer = ((x - 8 - _translation) ** 2 + (y - 8 - _translation) ** 2 <= 3 ** 2) & (
            (x - 8 - _translation) ** 2 + (y - 8 - _translation) ** 2 > inner_rad ** 2)
    image_of_circle[circle_inner] = 1
    image_of_circle[circle_outer] = val
    image_of_circle = image_of_circle.unsqueeze(0)
    return image_of_circle


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test Case 1
    _source = create_circle(0, inner_rad=3, val=0.5).unsqueeze(0).cuda()

    # Test Case 2
    # _source = torch.zeros(32, 32)
    # _source[8:24, 8:24] = 1
    # _source = _source.unsqueeze(0).unsqueeze(0).cuda()

    # _source = torch.rand(1, 1, 32, 32).cuda()
    _source.requires_grad_(True)

    # Test Case 3
    #_target = _source.detach().clone()

    # Test Case 1 and 2
    _target = create_circle(0, inner_rad=1.5, val=0.5).unsqueeze(0).cuda()

    # plot the images
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(_source.squeeze().detach().cpu().numpy(), cmap='gray')
    axs[1].imshow(_target.squeeze().cpu().numpy(), cmap='gray')
    plt.tight_layout()
    plt.show()

    brenier = BrenierW2Loss()
    _loss = brenier(_source, _target)

    [_g] = torch.autograd.grad(_loss, _source, create_graph=True)

    # plot the gradient and the image
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(_g.squeeze().detach().cpu().numpy())
    axs[0].set_title("Gradient")
    axs[1].imshow(_source.squeeze().detach().cpu().numpy())
    axs[1].set_title("Source")
    axs[2].imshow(_target.squeeze().cpu().numpy())
    axs[2].set_title("Target")

    # set the main title as the cost
    plt.suptitle(f"Cost: {_loss.item():.6f}")
    plt.tight_layout()
    plt.show()
