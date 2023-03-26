---
title: "Assignment 6 Plots"
author: "Harikesh Kushwaha"
header-includes:
  - \usepackage{amssymb,amsmath,geometry}
  - \setmainfont{TeX Gyre Schola}
  - \setmathfont{TeX Gyre Schola Math}
output: 
  pdf_document
---

# Assignment 6

## Problem 1

![./images/0101.png](./images/0101.png)
This is the displacement with x plot. The analytic solution and the solution learned by the model are plotted on top of each other. The solution by the model is the solid blue line and the analytic solution is the dashed orange line. The model is able to learn the solution to the differential equation almost exactly in just 10 epochs!

---

![./images/0102.png](./images/0102.png)
This is the total loss plot. The loss is the mean squared error between the analytic solution and the solution learned by the model. The loss by the boundary points and the residual is added together to get the total loss. The total loss is plotted here. The loss decreases as the model learns the solution to the differential equation and becomes almost zero after 10 epochs.

---

## Problem 2

![./images/0201.png](./images/0201.png)
This is the $EA$ with x plot. Same as before, the analytic solution and the solution learned by the model are plotted on top of each other. The solution by the model is the solid blue line and the analytic solution is the dashed orange line. The model is able to learn the solution to the differential equation almost exactly in just 10 epochs!

---

![./images/0202.png](./images/0202.png)
Again, this is the total loss plot. The loss is the mean squared error between the analytic solution and the solution learned by the model. The total loss is plotted here. The loss decreases as the model learns the solution to the differential equation.

---
