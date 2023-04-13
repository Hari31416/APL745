---
title: "Assignment 7 Plots"
author: "Harikesh Kushwaha"
header-includes:
  - \usepackage{amssymb,amsmath,geometry}
  - \setmainfont{TeX Gyre Schola}
  - \setmathfont{TeX Gyre Schola Math}
output: 
  pdf_document
---

# Assignment 7

## Problem 1

![./plots/0101.png](./plots/0101.png)
This shows the boundary points chosen to calculate the boundary loss of the model.

---

![./plots/0102.png](./plots/0102.png)
These are the interior points. These wil be used as collocaiton points for training the model.

---

## Problem 2

![./plots/0201.png](./plots/0201.png)
The figure shows the learned function $u(x,y)$, the x-displacement. We can see that the neural network has learned the displacement field.

---

![./plots/0202.png](./plots/0202.png)
This shows the function $v(x,y)$, the y-displacement learned by the model. The plot matches very well with the plot given in the problem ppt.

---

## Problem 3

![./plots/0301.png](./plots/0301.png)
This shows different losses as a function of the number of iterations. You can see three loss lines:

- The first one is the boundary loss. This is the loss on the boundary points.
- The second one is the PDE loss. This is the residual loss on the interior points.
- The third one is the total loss. This is the sum of the boundary and interior loss.

We can see that all the losses are decreasing with the number of iterations. This shows that the model is learning the displacement.

---

## Problem 5

![./plots/0501.png](./plots/0501.png)
This is the learned $u(x,y)$ plotted on the generated mesh. The plot matches very well with the plot given in the problem ppt, though the plot is not as smooth as the one given in the ppt because of less number of collocation points.

---

![./plots/0502.png](./plots/0502.png)
This is the learned $v(x,y)$ plotted on the generated mesh. The plot matches very well with the plot given in the problem ppt.

---
