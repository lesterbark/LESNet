### 6.4 CIFAR-10 with pixels as entities, against a convolutional network

A regular grid is the regime a convolutional network is built for. In CIFAR-10 (Krizhevsky, 2009) each of the 1024 pixels is treated as an entity with its three colour values; its partners are the 3×3 neighbourhood around
its grid cell (itself included), and the relation is the grid offset to the partner (the two signed components, their absolute values, the distance and the direction cosines). The LESNet model has three stages of two layers at resolutions 32, 16 and 8 with
2×2 average pooling between stages and stage widths 48, 96 and 192, followed by global average pooling and a linear classifier (150k parameters). The convolutional baseline uses the same three-stage layout (two 3×3 convolutions with batch normalisation and ReLU per stage, 2×2
average pooling, global average pooling and a linear classifier) with stage widths 24, 48 and 96 to match parameter count (163k parameters). Both models are trained for 40 epochs at batch size 128 with AdamW (weight decay 1e-4 for LESNet, 5e-4 for the CNN), a cosine schedule, random crops and horizontal flips, and a learning rate chosen from {1e-3, 3e-3, 1e-2, 3e-2} after 4
epochs on 5000 held-out training images. Reported accuracy is on the 10 000 test images, mean and 95% interval over three seeds.

LESNet reaches 63.0% against 88.5% for the CNN, a 25.5-point gap. Extending to a single 200-epoch run gives 68.5% and 89.0%: the CNN fits the training set (loss 0.016) while LESNet underfits it (0.74). Widening LESNet's partner window to 5×5 (25 partners per pixel, same parameter count) drops 40-epoch accuracy to 60.4%, below the 3×3 window; final training loss is higher in every seed (1.06–1.13 vs 0.95–1.03), though the three-seed intervals overlap (1.09 ± 0.10 vs 0.98 ± 0.11).

