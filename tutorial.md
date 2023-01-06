
# A Vehicle Tutorial 

## Outline

0. Introduction and Motivation for Vehicle

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

## Introduction and Motivation
 
### What is Neural Network Verification about?

Neural networks are a widely-used and highly popular tool in the field of machine learning. As such, it is important to establish formal guarantees about their behaviour. Following the pioneering work of [...,Katz17,...] that showed how domain-specific SMT-solving methods can be used to verify properties of simple neural networks, neural network verification has become an active research area. 

Formally, a neural network is a function $N : R^m \rightarrow R^n$, where $R^m$ and $R^n$ can be implemented as vectors of real or rational numbers. Verification of such functions most commonly boils down to specifying admissible intervals for the function's output given an interval for its inputs. For example, one can specify a set of inputs to belong to an $\epsilon$- neighborhood of some given input $\mathbf{x}$, and verify that for such inputs, the outputs of $N$ will be in $\delta$ distance to $N(\mathbf{x})$. This property is often called $\epsilon$*-ball robustness* (or just *robustness*), as it proves the network's output is robust (does not change drastically) in the neighborhood of certain inputs.

Seen as functions, neural networks have two particular features that play an important role in their verification: these functions are not written manually, but generated (or *fitted*) to model the given data distribution. As a consequence, "big data" often requires one to use large neural networks, and we often attribute very little semantic or structural meaning to the resulting function. 

### Challenges in Neural Network Verification

There are four main research challenges in this area: 
1. Scalability of (semi-)decision procedures that check the property satisfaction for neural networks. State-of-the art neural network verifiers [...], based on a combination of abstract interpretation algorithms and domain-specific heuristics can verify neural networks of size ... whereas large industrial models used by Amazon or Google reach the size of .... (I need help here from Matthew, Marco or Natalia)
2. Limited scope of neural network properties available in the literature. Arguably,  $\epsilon$-ball robustness has very limited practical applications. Various efforts of the community to broaden the range of verifiable properties mainly resulted in domain speciic solitions (see eg [] for verification of networks used in air collison avoidance). 
3. Neural networks are rarely used as stand-alone oracles. They are usually part of more complex systems. Verifying the network's behavior within a larger system is an area that still requires investigation. (maybe good to give a few citations here for existig work)
4. Because a given neural network is generated to fit the data, rather than to satisfy a given property, in majority of cases, a naive attempt to verify the network results in failure to establish that the property actually holds.  For example, as reported in [], a 99% accurate network may only be proven robust in the neighborhood of 1% of its images. However, one can re-train the network by translating a given property into a loss function, and this can dramatically increase the chances that the network satisfies the property: [] shows in the best case an increase from 1% to 90%. (Check DL2 paper and last year's stats from Marco's paper and give citations).    

Combined together, these four points make practical use of verification techniques inaccessible for majority of machine learning practicioners and even researchers, despite of the high demand for safety guarantees in complex intelligent systems. Imagine, for example, a designer of a chatbot who tries to prove that the chatbot does not offend or mislead a human user, and suppose they were lucky enough to install one available neural network verifier.  For a start, their neural network maybe too big to verify, so they need to make it small enough. Next, they need to define formally what "offend or mislead" is, the task that is made harder by having to use a relatively low-level language of the verifier that expects the properties to be stated on the level of individual inputs and neurons. If they have made through that hurdle, and finally can run the "verify" command line, they may suddenly find that their property fails for the model they have. So, they need to return to square one and train a different model, that does satisfy the property. Unfortunately, the installed verifier will not be able to help with this task!

### What does Vehicle Team Propose?

In this tutorial, we present the tool Vehicle that provides support for this complex verification cycle: It provides an environment that allows one to express neural network specifications in a high-level, human-readable format. Then it compiles them into low-level queries that can be passed automatically to verifiers to prove whether the specification holds or provide a counterexample. If the specification cannot be verified, Vehicle gives one an option to automatically generate a new loss fuction that can be used to train the model to satisfy the stated property. 
Once a specification has been verified (possibly after property-driven re-training), Vehicle allows one to export the proof to an interactive theorem prover, and reason about the behavior of the complex system that embeds the machine learning model. 

Vehicle programs can be compiled to an unusually broad set of backends,
including: 

 a) loss functions for Tensorflow which can be used to guide 
 both specification-directed training and gradient-based counter-example
 search.
 
 b) queries for the Marabou neural network verifier, which
 can be used to formally prove that the network obeys the specification.
 
 c) Agda specifications, which are tightly coupled to the original network
 and verification result, in order to scalably and maintainably construct
 larger proofs about machine learning-enhanced systems.
 
Currently, Vehicle supports the verifier Marabou, the ITP Agda, and the ONNX format for neural networks.

### Objectives of this Tutorial

This tutorial will give an introduction to the Vehicle tool 
(https://github.com/vehicle-lang/vehicle) and its conceptual approach
to modelling specifications for machine learning systems via functional
programming. It will teach the participants to understand the 
range of problems that arise in neural network property specification, 
verification and training, and will give a hands-on experience on 
solving these problems at a level of a higher-order specification 
language with dependent types.

## Vehicle Preliminaries

- introduction of dataset and models
- introduction of basic syntax
- describe a property in words
- example with NN?
- example of property not holding 
- fix in vehicle!

## Example 1 - Stanford Dogs Dataset

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
In this case, we have called our network `score`. It takes in a 2-dimension tensor of size 28 x 28
filled with rationals - an image -, and returns a vector of 20 rationals -a score for each of the 0 to 19 dog breeds-.
Networks are annotated with the keyword `@network` on the previous line of their declaration.

## Declaration of properties

### A simple property (ExistsGreatDane)

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
 
The `forall` quantifier allows us to check that all dog breeds `d2`, such that `d2` is different from `d1`, have a lower score than `d1` in the vector `(score img)`. 

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
by the verifier. Properties in Vehicle Specification should always have the `@property` keyword 
in the previous line.

If we compile this specification and pass it to the verifier, `existsGreatDane` will most likely 
hold for a nonsensical image. To avoid this, we must restrict our property to valid images only.
We consider an image valid if all its pixels are valid, i.e., normalized to be in the range 0 to 1.

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

### A more meaningful property (DoesNotConfuseBigAndSmall)

Let us now consider the property that our network is not confusing small dogs with big dogs. That is, 
if our first choice for a dog image is a big or small dog, the second choice should be too. 

Suppose the indexes 2, 11 and 12 of the score vector refer to German Shepherds, Chihuahuas and Pekinese dogs.
Type synonyms conveniently allow us to give a meaningful name to these indexes. 

```{.vcl}
    germanShepherd = 2
    chihuahua  = 11
    pekinese   = 12
```

We now define two lists of dogs (recall that the type of Dog is `Index 20`) holding what we consider as small and big dogs. Lists hold an arbitrary number of elements of a single type, and are created using `[x_1, ..., x_n]`. 


```{.vcl}
    smallDogs : List Dog
    smallDogs = [chihuahua, pekinese]

    bigDogs : List Dog
    bigDogs = [greatDane, germanShepherd]
```

A dog `d2` will be the second choice of our network for a certain image `img` if there exists another dog `d1` (different from `d2`) that is classified as first choice, such that any other dog `d3` has a lower score than `d1` and `d2`. The `let ... in` clause allows us to make the code more readable by defining an expression to be used 
after `in`.

```{.vcl}
    isSecondChoice : Image -> Dog -> Bool
    isSecondChoice img d2 =
        let scores = score img in
        exists d1 .
            isFirstChoice d1 and d1 != d2 and 
            forall d3 . d3 != d1 and d3 != d2 => scores ! d2 > scores ! d3
```



```{.vcl}
    noConfusionWith : Image -> List Dog -> List Dog -> Bool
    noConfusionWith x dogs1 dogs2 =
        forall dog1 in dogs1 .
            forall dog2 in dogs2 .
                not (isFirstChoice x dog1 and isSecondChoice x dog2)
```

(Comment on not using for loops to transverse the lists.)
    
```{.vcl}
    @property
    doesNotConfuseBigAndSmall : Bool
    doesNotConfuseBigAndSmall =
        forall x . validImage x => noConfusionWith x bigDogs smallDogs
```

## Declaration of a dataset

- declaration of dataset
    







