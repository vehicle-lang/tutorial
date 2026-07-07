# Improvements for Vehicle Tutorial
This file is for making miscellaneous notes on how to improve the Vehicle tutorial pages. These may be anything from big ideas to minor changes.

**General note**: at some point we should decide where to talk about network normalisation. Matthew says this should take place within the network itself (i.e., in the ONNX file, viewable using [this site](https://netron.app/)), rather than inside of a specification.

## Ch. 4: Property-Driven Training
My overall vision for this chapter is to introduce a concrete example to aid with the explanation of how to use Vehicle specifications/properties to generate logical loss functions, with the task loss function to train a neural network. 

I will use the MNIST Fashion dataset to train a (very simple) neural network and use (some/various definition(s) of) robustness to generate a logical loss function. I will compare this network with one that hasn't been trained using a logical loss function (just the standard task loss function) and show the difference in both task accuracy and satisfaction of the robustness property.

It may also be interesting to investigate these metrics (accuracy, robustness) with different types of differentiable logics, weighting schemes, etc. Using multiple robustness properties is also a good opportunity to demonstrate how they can be implemented in Vehicle, and involving them in the statistical analysis may provide insights into their respective differences and use cases.

Whilst the theoritical background is useful in providing a motivation for generating logical loss functions, it is not entirely necessary for learning to generate/use logical loss functions. I may cut this down significantly.

Current roadblocks:
- Logical loss function generation is currently broken: see [here](https://github.com/vehicle-lang/vehicle/issues/1183).
- For a large number of images (which I would like to use for more accurate statistics), the `vehicle compile queries` command takes a _lot_ of memory: see [here](https://github.com/vehicle-lang/vehicle/issues/1184). I need to learn how to perform this in batches (and this might also have to feature in the tutorial so that it can be replicated by those learning the language). 

**Note on batch processing:** I am currently working on a script that performs batch query compilation. This will take a large idx file and generate $n$ batches of queries. The number of queries depends on the property being validated, but in general the number of queries in each batch is:

$$
    k=\frac{\text{number of images}\times Q}{n}
$$

where $Q$ is the constant number of queries that a property generates for a single data point.