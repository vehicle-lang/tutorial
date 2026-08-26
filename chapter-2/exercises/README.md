# Overview

The exercises in this chapter are designed to:

1. Give you practice with the provided chapter code
2. Practice writing more specs for the same model
3. Develop your new model and specs from scratch.

   Overview of what you need to solve the exercises:

##    Exercises #1 -3 
Start by simply running the code that was discussed in the above chapter. It is available from the chapter-2/chapter-code directory of the tutorial repository.
Then proceed to write more specs as instructed. 

ACAS Xu is a collection of 45 neural networks that together make up a collision avoidance system
for automonous unmanned aircraft.
The partial verification of the system was first described in the seminal
[Reluplex paper](https://arxiv.org/abs/1702.01135).
The entire specification consists of all
10 properties, can be written in a single file.


### The file set up remains the same:

You can find these files in the folder `chapter-code`:

- `acasXu.vcl` - the specification describing the desired behaviour. This needs to be edited

- `acasXu_1_7.onnx`, `acasXu_1_8.onnx`, `acasXu_1_9.onnx` - 3 out of the 45 networks. The remainder can be found [here](https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources/onnx/acasxu).

### Verifying using Marabou

The following command verifies all properties for the network `acasXu_1_7.onnx`:

```bash
vehicle \
  verify \
  --specification acasXu.vcl \
  --verifier Marabou \
  --network acasXu:acasXu_1_7.onnx \
```

## Exercise #4 (⭑⭑⭑). Your first independent Vehicle specification
Clone the tutorial repository.
Find the iris_model.onnx model under resources-by-dataset/iris/. This model was trained on the famous Iris flower data set.
Find the Iris data set files iris_test_data.idx and iris_test_label.idx under resources-by-dataset/iris/.
Examine the data set and try to define a few “obvious properties” that should hold for the model. You are free to look at the Wikipedia page for the Iris flower data set or consult other sources.
Write those properties as a Vehicle specification. Ensure that your specification is well-typed. See the Vehicle Manual for how to type-check a Vehicle specification file.
Verify that the properties in your Vehicle specification hold using the vehicle command.


