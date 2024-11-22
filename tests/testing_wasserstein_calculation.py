


import matplotlib.pyplot as plt

# generate an image of a square
_image_of_square = torch.zeros(32, 32)
# _image_of_square = torch.rand(32, 32)
_image_of_square[8:24, 8:24] = 1
_image_of_square = _image_of_square.unsqueeze(0)

_image_of_circle = create_circle(0, inner_rad=3, val=0.5)

# show the images
_, axes = plt.subplots(1, 2)
axes[0].imshow(_image_of_square.cpu().numpy().transpose(1, 2, 0), cmap='gray')
axes[1].imshow(_image_of_circle.cpu().numpy().transpose(1, 2, 0), cmap='gray')
plt.tight_layout()
plt.show()

# stack the images
# _source = torch.rand(2, 1, 32, 32, dtype=torch.float32, device='cuda')
_source = torch.cat([_image_of_square.unsqueeze(0), _image_of_circle.unsqueeze(0)], dim=0).cuda()
_source.requires_grad_(True)

_loss = FastConvolutionalW2Cost(scaling=0.9, reduction='none')

_target = torch.cat([_image_of_square.unsqueeze(0), create_circle(0, inner_rad=1.5, val=0.5).unsqueeze(0)], dim=0).cuda()
_image, _cost = _loss(_source, _target)

# plot the images
_, axes = plt.subplots(1, 2)
axes[0].imshow(_image[0].detach().cpu().numpy().transpose(1, 2, 0))
axes[1].imshow(_image[1].detach().cpu().numpy().transpose(1, 2, 0))
plt.tight_layout()
plt.show()

print(_cost)

[gi] = torch.autograd.grad(_cost.mean(), _source, retain_graph=False)

_, ax = plt.subplots(1, 2, figsize=(15, 5))
# compute the direction of descent at each point in the gi
ax[0].set_title(f'{_cost[0].item():.2f}')
ax[0].imshow(gi[0].detach().cpu().numpy().transpose(1, 2, 0))
ax[0].axis('off')
ax[1].set_title(f'{_cost[1].item():.2f}')
ax[1].imshow(gi[1].detach().cpu().numpy().transpose(1, 2, 0))
ax[1].axis('off')

plt.tight_layout()
plt.show()

# Testing if the code works for circles that are very close
circle_1 = create_circle(0, inner_rad=3, val=0.5)
circle_2 = create_circle(2.0, inner_rad=3, val=0.5)
# circle_1 = create_circle(0.0, inner_rad=3, val=1)
_, axes = plt.subplots(1, 2)
circle_1 = circle_1 / circle_1.sum()
circle_2 = circle_2 / circle_2.sum()
axes[0].imshow(circle_1.cpu().numpy().transpose(1, 2, 0), cmap='gray')
axes[1].imshow(circle_2.cpu().numpy().transpose(1, 2, 0), cmap='gray')
plt.tight_layout()
plt.show()

# Test the calculation by comparison to python optmal transport (POT)
import ot
import geomloss
from brenier_wasserstein_motion_prior import BrenierW2Loss
unit_square_grid = True

# Get the pairwise euclidean distances
if unit_square_grid:
    _y, _x = np.meshgrid(
        np.arange(32, dtype=np.float32) / 31.0,
        np.arange(32, dtype=np.float32) / 31.0,
        indexing='ij'
    )
else:
    _y, _x = np.meshgrid(
        np.arange(32, dtype=np.float32),
        np.arange(32, dtype=np.float32),
        indexing='ij'
    )

# Concatenate the things
yx = np.concatenate([_y[..., None], _x[..., None]], axis=-1)

# Get the flattened images and grid
yx_flat = np.reshape(yx, (-1, 2))
img1_flat = np.reshape(circle_1.cpu().numpy().squeeze(), (-1,))
img2_flat = np.reshape(circle_2.cpu().numpy().squeeze(), (-1,))

# Normalize the images
img1_flat = img1_flat / img1_flat.sum()
img2_flat = img2_flat / img2_flat.sum()

# Calculate the cost matrix
C = ot.dist(yx_flat, yx_flat, metric='sqeuclidean')

# Also calculate some things necessary for geomloss
w_s, h_s = 32, 32
#sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=1 / max(h_s, w_s), diameter=2 / max(w_s, h_s), scaling=0.9, debias=True)
# sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=1 / max(h_s, w_s), diameter=2 / max(w_s, h_s),
#                                      scaling=0.9, debias=True)
sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.0000001, scaling=0.9, backend="online")
geomloss_input_1 = torch.tensor(img1_flat)
geomloss_input_2 = torch.tensor(img2_flat)
grid = torch.tensor(yx_flat)
geomloss_val = sinkhorn_loss(geomloss_input_1[None, ...], grid[None, ...], geomloss_input_2[None, ...], grid[None, ...])

# Calculate the OT cost
loss = ot.emd2(img1_flat, img2_flat, C)
loss_sinkhorn = ot.sinkhorn2(img1_flat, img2_flat, C, 1 / max(h_s, w_s))

# Calculate the loss with the own debiased convolutional wasserstein distance
_loss = FastConvolutionalW2Cost(scaling=0.9, reduction='none')
loss_own = _loss(circle_1[None, ...], circle_2[None, ...])[-1]

# Get the BrenierW2Loss as well
brenier = BrenierW2Loss()
_loss = brenier(circle_1[None, ...], circle_2[None, ...])

# Printing some things
print("Python Optimal Transport: ", loss)
print("Python Optimal Transport sinkhorn:", loss_sinkhorn)
print("GeomLoss: ", geomloss_val)
print("Own debiased convolutional wasserstein distance:", loss_own)
print("Own Brenier-based loss:", _loss)
print("Finished running the tests...")

