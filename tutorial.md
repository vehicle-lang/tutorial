
# A Vehicle Tutorial 

## Outline

1. A simple semantically meaningful example.
    - Maybe hierarchical classification? e.g. In this dogs dataset it shouldn't confuse an Afghan hound and a Border terrier.
    - Introduce basic syntax and properties.
2. A more complicated semantically meaningful example 
    - e.g. AcasXu
    - Introduces building and reusing functions.
3. Less semantically meaningful
    - e.g. MNIST robustness
    - Introduces concepts of datasets, parameters etc.
4. Semantically meaningful + integration with Agda.
    - Braking example?
    - Vehicle controller?

## Introduction
 
Neural networks are a widely-used and highly popular tool in the field of machine learning. As such, it is important to establish formal guarantees about their behaviour. 

This tutorial will introduce you to Vehicle, a tool for enforcing specifications on neural networks. 
Vehicle allows you to express specifications in a high-level, human-readable format. Then compiles them into low-level queries that can be passed to verifiers to prove whether the specification holds or provide a counterexample. 
Once a specification has been verified, Vehicle allows you to export the proof to an interactive theorem prover. Currently, Vehicle supports the verifier Marabou, the ITP Agda, and the ONNX format for neural networks.

## Motivation

- introduction of dataset
- describe a property in words
- example with NN?
- example of property not holding 
- fix in vehicle!

## Example 1

For this first example, we will consider a neural network that classifies dog breeds from images in the 
[Stanford Dog Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset). Although the dataset includes 120 breeds, for this example, we will focus on only 20 breeds.
That is, for a given input image, the network will give a probability distribution over the 20 dog breeds. 

We want to ensure that the trained network is not confusing breeds that do not have similar characteristics. 
For instance, it is acceptable for the network to classify a German Shepherd as a Great Dane, but it should not confuse a German Shepherd with a Chihuahua. 

Let us see how you can express and enforce this property in Vehicle. 
To do so, you will need the trained network in ONNX format and the dataset.

- Section0 : Introduce design choice for property specification. (some need data, some need networks, some none of them).

## Declaration of the network

We will start by declaring the network. In Vehicle specifications, networks are essentially black-box functions. 
Thus, we simply need to provide its input and output types.

```{.vcl}
    @network
    score : Tensor Rat[28, 28] -> Vector Rat 20
```

Function types are declared using `:` and function arguments are separated with `->`. 
For example, we have called our network `score`. It takes in a 2-dimension tensor of size 28 x 28
filled with rationals - an image -, and return a vector of 20 rationals -a score for each of the 0 to 19 dog breeds-.
Networks are annotated with the keyword `@network` on the previous line of their declaration.

## Declaration of properties

### A simple property

Let us start with a simple property stating that there is at least one image classified as a Great Dane.

First, we need to declare an auxiliary function `isFirstChoice` that 
checks whether the network classifies a given image `img` as a specific dog breed `d1`. In other words, that the 
vector returned by `score` for the image `img` will have its maximum in index `d1`.

```{.vcl}
    isFirstChoice : Tensor Rat[28, 28] -> Index 20 -> Bool
    isFirstChoice img d1 =
        forall d2 . d2 != d1 => (score img) ! d1 > (score img) ! d2
```

`isFirstChoice` takes in an image (`Tensor Rat[28,28]`) and a reference to a specific dog breed `d1`. Note how 
the type of `d1` is `Index 20`. The set of valid instances of this type are the natural numbers {0, ..., 19}.
By using the type `Index 20` instead of simply `Nat`, we make sure that no out-of-bound errors will arise from using `!`, which is the vector look-up operator, when accessing position `d1` of `(score img)`, i.e. `(score img) ! d1`, since we know that `score` will output a vector of size 20.
 
The `forall` quantifier allows us to check that all dog breeds `d2`, such that `d2` is different from `d1`, have a lower score than `d1` in the score vector for `img`, i.e. `(score img)`. 

Instead of repeatedly writing `Tensor Rat[28,28]`, we can declare a type synonym called `Image`.
Similarly, we can define `Dog` to be an index of the 20 breeds. Like this, the signature of 
`isFirstChoice` becomes more meaningful and simple.

```{.vcl}
    type Image  = Tensor Rat [28, 28]
    type Dog    = Index 20

    isFirstChoice : Image -> Dog -> Bool
``` 

We can now check whether at least one image will be classified as a Great Dane. 
We use `exists` to check whether the predicate `isFirstChoice img greatDane` holds for some image `img`,
where we have declared `greatDane` to be the index number 1 of all the breeds.


```{.vcl}

    GreatDane = 1

    @property
    existsGreatDane : Bool
    existsGreatDane = exists img . isFirstChoice img greatDane
```

Note how we have annotated `existsGreatDane` with the keyword `@property`. Properties 
are boolean expressions whose value cannot be decided within Vehicle and will instead be decided
by the verifier. 

If we compile this specification and pass it to the verifier, `existsGreatDane` will most likely 
hold for a nonsensical image. To avoid this, we must restrict our property to valid images only.
We will consider an image is valid if all its pixels are valid, i.e., normalized to be in the range 0 to 1.

```{.vcl}
    validPixel : Rat -> Bool
    validPixel p = 0 <= p <= 1

    validImage : Image -> Bool
    validImage x = forall i j . validPixel x ! i ! j
```
Here, `forall` conveniently infers the type of `i` and `j` as `Index 28`, based on the type of `x`, i.e. a 2-dimensional tensor of size 28x28. It then uses `i`, `j` as iterators to access each of the pixels in the image, i.e. `x ! i ! j`.

We can now rewrite our property restricted to valid images only.

```{.vcl}
    @property
    existsGreatDane : Bool
    existsGreatDane = exists img . (validImage img) and (isFirstChoice img GreatDane)
``` 

### A more meaningful property




```{.vcl}
    isSecondChoice : Image -> Dog -> Bool
    isSecondChoice x dog2 =
        let scores = score x in
        xists dog1 .
            isFirstChoice dog1 and
            forall d . d != dog1 and d != dog2 => scores ! dog2 > scores ! d

    noConfusionWith : Image -> List Dog -> List Dog -> Bool
    noConfusionWith x dogs1 dogs2 =
        forall dog1 in dogs1 .
            forall dog2 in dogs2 .
                not (isFirstChoice x dog1 and isSecondChoice x dog2)

    germanShepherd = 2

    chihuahua  = 11
    pekinese   = 12

    smallDogs : List Dog
    smallDogs = [chihuahua, pekinese]

    bigDogs : List Dog
    bigDogs = [greatDane, germanShepherd]


    @property
    doesNotConfuseBigAndSmall : Bool
    doesNotConfuseBigAndSmall =
        forall x . validImage x => noConfusionWith x bigDogs smallDogs
```

### Declaration of a dataset

- declaration of dataset
    







