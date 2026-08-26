---
title: "Property-Driven Training"
---
---
**Note**

This section contains a lot of theory. For implementation details, please skip to [Logical Loss Functions In Vehicle](#logical-loss-functions-in-vehicle).

---

# Motivation
We will begin this chapter with a question: _how can we train a neural network to be more robust within a desirable $\epsilon$?_
The long tradition of robustifying neural networks in machine learning has a few methods
ready. For example, we can re-train the networks with new data that was augmented using images within the
desired $\epsilon$-balls, or generate adversarial examples (sample images closest to the boundary of the $\epsilon$-ball) during training. Let us briefly explore these approaches.


[need citations for data augmentation/adversarial training]: #

# Current Approaches
**Data Augmentation** works by generating additional data within the $\epsilon$-balls of the original training data points, usually done using methods such as rotation, cropping, flipping, random sampling, etc. The augmented data points are assigned the same label as the original ones they were augmented from. We can then use our usual training methods with this augmented dataset with the hope that it will improve the network's average-case robustness [@SK19].

Unfortunately, this approach has its problems. Firstly, if our original sampled data point is already very close to the decision boundary, there is a chance that an augmented data point will actually lie on the wrong side, even though it is still within the $\epsilon$-ball. This means it will have been assigned the wrong label:

![Data Manifold for D](../assets/images/SR-vs-CR-4-white-bg.png)

In the case where two data points' $\epsilon$-balls overlap, there is a chance we generate two new data points with the same position in the input space. Furthermore, if the two original data points lie both close to (and on opposite sides of) the decision boundary, the augmented data points may have _different labels_, despite occupying the same location in the input space:

![Data Manifold for D](../assets/images/SR-vs-CR-5-white-bg.png)

These inconsistencies mean this approach is generally unviable for network robustification.

**Adversarial Training** also involves generating new data to train the network, but unlike data augmentation where perturbations are sampled randomly, adversarial training aims to find the _worst-case_ perturbation within $\epsilon$-distance to a data point from the training dataset. Whilst data augmentation can be done using worst-case examples, it is still subtly different to adversarial training. Most notably, adversarial training is a process integrated into the network's training loop, so perturbations are regenerated at every iteration. This means the worst-case examples will _always_ be worst case, which is not true for data augmentation, as after a certain number of iterations the network will have learnt to account for these examples. The goal of adversarial training is to improve the network's worst-case robustness; it is a form of prophylaxis against adversarial attacks.

[the below paragraph was pasted from the previous version -- double check understanding and fix citation]: #

The main limitation of adversarial training turns out to be the logical property it optimises for.
Recall that we may encode an arbitrary property in Vehicle. However, as was discovered in [@CasadioKDKKAR22], projected gradient descent can only optimise for one concrete property.
Recall the property of $\epsilon$-ball robustness was defined as:
$\forall \mathbf{x} \in \mathbb{B}(\hat{\mathbf{x}}, \epsilon)\;.\;\text{robust}(f(\mathbf{x}))$. It turns out that adversarial training determines the definition of $\text{robust}$ to be
$|f(\mathbf{x}) - f(\hat{\mathbf{x}})| \leq \delta$.

# Beyond $\epsilon$-Balls
In the previous chapter, we learnt how to prove properties of neural networks, with a specific focus on $\epsilon$-ball robustness. It is of course nice that we can prove this property is true for a given neural network, but as we have seen above, it is not novel to be able to train for it; this can be done with relative ease simply by using adversarial training.

However, something that adversarial training cannot do is teach a network to abide by _any arbitrary logical property_. This is where Vehicle comes in: we can define arbitrary properties in our specifications, and use Vehicle's built-in functionality to compile these into a loss function to train a neural network. We can escape the world of $\epsilon$-balls and train our networks to satisfy any property we may desire. Nonetheless, $\epsilon$-ball robustness is a useful and intuitive property to understand, and we will continue to use it for our running example throughout the rest of this section,  as well as in the exercises at the end.

Before we explore exactly how we train a network on logical properties in practice, those uninitiated into the cults of machine learning and logic may appreciate some theoretical background of how this is possible. This we cover in the following few sections.

# Loss Functions
Humans learn by making mistakes. The same is true of neural networks. Loss functions are a way of measuring the "magnitude" of a mistake made by a neural network. For a given training input, loss functions compute a penalty proportional to the difference between the output of the network and the _true_ output (i.e., the training label). Formally, this is written as follows:

$$\mathcal{L}: \mathbb{R}^n \times \mathbb{R}^m \rightarrow \mathbb{R}$$

The function $f_\theta: \mathbb{R}^n \rightarrow \mathbb{R}^m$ represents the network, whose optimisation parameters are $\theta$, and $n$ and $m$ represent the sizes of the input and output tensors respectively.

One of the simplest (yet usable) loss functions is called **mean squared error**, defined as:

$$\mathcal{L}_\text{MSE}(\hat x, y)=\frac{1}{k}\sum_{i=1}^k(y_i-f_\theta(\hat x_i))^2$$

where $k$ is the total number of data points, $y_i$ is the label for data point ${\hat x}_i$, and $f_\theta(\hat x_i)$ is the model's predicted value for $\hat x_i$. I.e., we find the difference between the prediction and the actual value, square it, and take the average across the training dataset.

Models learn by iteratively tweaking their optimisation parameters with the goal of minimising the output of the loss function. The most common way to do this is using **gradient descent**. Formally, we wish to find the set of parameters $\theta$ that yields the least loss:

$$\min_\theta\mathcal{L}(\hat x,y)$$

Here we can explain further the mechanism that drives adversarial training. We can use a variant of gradient descent, called **projected gradient descent**, to _maximise_ loss in order to find worst-case perturbations. We ensure that the perturbation still lies within the $\epsilon$-ball of the original data point by _projecting_ those perturbations that escape the $\epsilon$-ball back inside. Our new training objective becomes:

$$\min_\theta\bigg[\max_{x:|x-\hat x|\le\epsilon} \mathcal{L}(x, y) \bigg]$$

In other words, we want to find the perturbation $x$ that is within $\epsilon$-distance of the data point $\hat x$ that produces the _largest_ loss (the worst-case perturbation). Then, we aim to find the optimisation parameters $\theta$ which _minimises_ this loss value.

# Logical Loss Functions
Traditional loss functions aim to minimise _task loss_. However, a neural network's performance on a given task is not necessarily correlated with the likelihood that it satisfies logical properties such as robustness. Let us recall our initial question: _how can we train a neural network to be more robust within a desirable $\epsilon$?_ Given what we now know about adversarial training and loss functions, we can reframe this question: _how can we train a network to abide by any arbitrary logical property?_

Gradient descent algorithms train networks to fit data. The big idea behind logical loss functions is to use that same algorithm to train the network to also obey the specification. Standard logic is insufficient for this task since we need a differentiable signal to find the gradient. Hence, we need a type of logic that we can differentiate.

# Differentiable Logics
Traditional logics are difficult to translate to loss functions for neural networks because they consist only of boolean values and operations, which are undifferentiable. If we want to use gradient-based techniques for logical properties, we need a logical calculus which uses connectives that are both mathematically rigorous in terms of semantics and differentiable.

Differentiable logics (DLs) convert booleans and operations over booleans into equivalent numerical operations that are differentiable. We have numerous DLs at our disposal (including DL2 [@FischerBDGZV19], DFLs [@KriekenAH22], and more), and Vehicle implements a variety of these. One such logic that has shown particular promise is quantitative linear logic (QLL), or Capucci Logic [@capucci2026]. QLL defines the logical connectives with the following real-valued functions:

**Negation:** $$\neg a:=-a$$

**Conjunction:** $$a\cap^pb:=\frac{1}{p}\log(e^{pa}+e^{pb})$$

**Disjunction:** $$a\cup^pb:=-\frac{1}{p}\log(e^{-pa}+e^{-pb})$$

**Implication:** $$a\implies b:=b-a$$

where $0<p<\infty$, representing the _hardness degree_. As $p\rightarrow\infty$, the QLL connectives converge on their traditional definitions. This is one approach that conserves logical semantics whilst allowing us to use gradient-based methods for property-driven training.

Note that in the Capucci logic the order is reversed: $0$ is the top and $\infty$ is the
bottom. This is usual in the differentiable logic literature, DL2 [@FischerBDGZV19]
included, because the value measures how far a formula is from being satisfied — the
less error there is, the more true the formula is.

# Logical Loss Functions in Vehicle
Vehicle supports several different differentiable logics from the literature, though we will not explore them here. Instead, we will use a simple example to explain how logical loss functions can be generated using Vehicle with PyTorch. Here, we use the Fashion MNIST dataset to train a neural network. All files used in this example can be found in the [chapter-4/chapter-code directory](https://github.com/vehicle-lang/tutorial/tree/exercises/chapter-4/chapter-code) of the tutorial repository.

First, we will load our Vehicle specification and define our constraint loss function:

<div class="tabs-container">
  <div class="tabs-header">
    <button class="tab-button active" data-index="0">PyTorch</button>
    <button class="tab-button" data-index="1">TensorFlow</button>
  </div>
  <div class="tabs-content">
<div>

```python
import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt

spec = loss_pt.load_specification(
    "fmnist-robustness.vcl",
    logic=vcl.VehicleDifferentiableLogic(),
)

constraint_loss_fn = spec["robust"]
```
</div>

<div>

```python
import vehicle_lang as vcl
from vehicle_lang.loss import tensorflow as loss_tf

spec = loss_tf.load_specification(
    "fmnist-robustness.vcl",
    logic=vcl.VehicleDifferentiableLogic(),
)

constraint_loss_fn = spec["robust"]
```
</div>

</div>
</div>

The first parameter to the `load_specification` function is the path to the Vehicle specification. The second parameter defines which logic to use -- this is optional, and defaults to DL2. We define which property from the specification to use as our constraint loss function by accessing it by name on the specification object.

Next, we will define a simple model and training procedure:

<div class="tabs-container">
  <div class="tabs-header">
    <button class="tab-button active" data-index="0">PyTorch</button>
    <button class="tab-button" data-index="1">TensorFlow</button>
  </div>
  <div class="tabs-content">
<div>

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

def network(x: torch.Tensor) -> torch.Tensor:
    return model(x.reshape(1, 1, 28, 28)).reshape(10)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
cross_entropy = nn.CrossEntropyLoss()

num_epochs = 5
alpha = 0.5

for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(images)
        loss = cross_entropy(logits, labels)

        constraint_loss = constraint_loss_fn(
            n=BATCH_SIZE,
            classifier=network,
            epsilon=torch.tensor(0.005),
            trainingImages=images.squeeze(1),
            trainingLabels=labels
        )

        constraint_loss = torch.stack(constraint_loss).mean()
        total_loss = alpha * loss + (1 - alpha) * constraint_loss

        total_loss.backward()
        optimizer.step()
```
</div>

<div>

```python
import tensorflow as tf
from tensorflow.keras import layers, Sequential

model = Sequential([
    layers.InputLayer(shape=(1, 28, 28)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10)
])

def network(x: tf.Tensor) -> tf.Tensor:
    return tf.reshape(model(tf.reshape(x, (1, 1, 28, 28))), (10,))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

num_epochs = 5
alpha = 0.5

for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            logits = model(images)
            task_loss = cross_entropy(labels, logits)

            constraint_loss = constraint_loss_fn(
                n=BATCH_SIZE,
                classifier=network,
                epsilon=tf.constant(0.005),
                trainingImages=tf.squeeze(images, axis=-1),
                trainingLabels=labels
            )

            constraint_loss = tf.reduce_mean(tf.stack(constraint_loss))
            total_loss = alpha * task_loss + (1 - alpha) * constraint_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```
</div>

</div>
</div>

Note that the `network` callable must match the type of the network declared in the Vehicle specification. The `alpha` parameter can be used to tweak the weighting of task loss vs. constraint loss, which are blended together in `total_loss`. We can now export this trained model to verify it using Vehicle, with the hope that it is more robust as a result of training with constraint loss. Model exportation can be done like so:

<div class="tabs-container">
  <div class="tabs-header">
    <button class="tab-button active" data-index="0">PyTorch</button>
    <button class="tab-button" data-index="1">TensorFlow</button>
  </div>
  <div class="tabs-content">
<div>

```python
import torch.onnx

model.eval()
input_tensor = torch.randn(1,1,28,28)

torch.onnx.export(
    model,
    input_tensor,
    "classifier.onnx", # file name
    external_data=False, # required for Marabou verification
)
```

Exporting an ONNX file in PyTorch works by tracing, which runs the model with an arbitrary input and records each operation. Hence, we provide the model with a randomly generated input tensor. At the time of writing, Marabou does not support external data locations, so we require that `external_data=False`.
</div>

<div>

```python
model.export("classifier")
```

This saves the model at the specified directory. To convert this to ONNX format, we can run the following command (this will require installing tf2onnx):

```bash
python -m tf2onnx.convert \
    --saved-model classifier \
    --output classifier.onnx
```
The model is now saved in ONNX format under the name specified with the `--output` parameter.
</div>

</div>
</div>


# Exercises

We will use symbols (⭑), (⭑⭑) and (⭑⭑⭑) to rate exercise difficulty: *easy, moderate and hard*.

## Exercise #1 (⭑): Run the Chapter code
Download the required materials (or produce them yourself) and repeat the steps described in this chapter. All code used in this chapter is available from the [tutorial repository](https://github.com/vehicle-lang/tutorial/tree/exercises/chapter-4/chapter-code).

## Exercise #2 (⭑): Verifying and comparing networks
Use a Vehicle specification (either the one provided, or your own) to verify a property (e.g., robustness) of a network trained _with_ a logical loss function, and compare this to one trained _without_ a logical loss function. Which is more robust, which has the better task accuracy, and why?

## Exercise #3 (⭑⭑): Further experimentation
Try various combinations of task loss functions, constraint loss functions, and alpha values. How do these affect each other? Is there a combination that makes the network more robust? Is there a combination that makes the network more accurate? What happens when you use multiple constraint loss functions simultaneously?

## Exercise #4 (⭑⭑⭑): Training a model from scratch
Finally, try creating your own model from scratch and repeat the experiments and comparisons described above. Explore the relationship between how complex a model is and to what degree it can satisfy robustness, and the effect robustness training can have on this.

Hint: a simple model is worse at spotting the difference between two different images. Does this make it more or less likely to be robust?
