# Example 1 - Stanford Dogs Dataset

For this first example, we consider a neural network that classifies dog images from the
[Stanford Dog Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) [Khosla et al., 2011] into their breeds.
The dataset includes over 20000 images of 120 dog breeds. For simplicity, we reduce it to 3600 images of 20 breeds.

We assume the trained network takes an input image and outputs a probability distribution over the 20 breeds.
(Comment about specifics of the trained network).

We want to ensure that the trained network is not confusing breeds of similar characteristics.
For instance, it is acceptable for the network to classify a German Shepherd as a Great Dane, but it should not confuse a German Shepherd with a Chihuahua. We split the 20 breeds into two lists by dog size and then aim to verify that our network is not confusing
big dogs with small dogs. Let us see how we can express and enforce this property (possibly by retraining) in Vehicle.

- Section0 : Introduce design choice for property specification. (some need data, some need networks, some none of them).

## Declaration of the network

In Vehicle specifications, networks are treated as black-box functions.
They are declared as regular functions, but only require one to specify their type signature preceded by the keyword `@network`.

``` vehicle
    @network
    score : Tensor Rat[28, 28] -> Vector Rat 20
```

Functions in Vehicle are declared by stating their name, followed by a semicolon and its input and output types, separated by `->`.
In this case, we have called our network `score`. It takes in a `Tensor Rat[28,28]`, which corresponds to a 2-dimension tensor of
size 28x28 filled with rationals - an image -, and returns a `Vector Rat 20`, a vector of 20 rationals - a score for each of the 0 to 19
breeds-.

Vehicle can typecheck and compile a specification knowing only the type signature of the network. (and why this is good)
It requires the user to provide a specific network in onnx format at the verification step.

## Declaration of properties

Properties are boolean expressions whose value cannot be decided within Vehicle and will instead be decided
by the verifier. A specification can have multiple properties, which must be annotated with the `@property` keyword and can have
either `Bool`, `Vector Bool` or `Tensor Bool` as types.

For example, a property stating that some previously defined function `f` is always positive would look as follows.

``` vehicle
@property
fIsPositive : Bool
fIsPositive = forall x . f x > 0.0
```

### A simple property (ExistsGreatDane)

Let us start with an example of a simple property stating that there is at least one image classified as a Great Dane.
First, we need to declare an auxiliary function that determines for each image, what is the predicted breed by our network `score`.

``` vehicle
    isFirstChoice : Tensor Rat[28, 28] -> Index 20 -> Bool
    isFirstChoice img d1 =
        forall d2 . d2 != d1 => (score img) ! d1 > (score img) ! d2
```

`isFirstChoice` takes in an image `img` (with type `Tensor Rat[28,28]`) and an index `d1` referring to a specific dog breed. Note how
the type of `d1` is `Index 20`. The set of valid instances of the type `Index n` are the natural numbers in {0, ..., n-1}.
`!` is the vector look-up operator, i.e. `(score img) ! d1` retrieves position `d1` of the vector `(score img)`.
By using the type `Index 20` instead of simply a natural number, we make sure that no out-of-bound errors will arise, since `score` will output a vector of size 20 and `d1` will be in the range [0, ... 19].
The `forall` quantifier checks that all dog breeds indices `d2`, that are different from `d1`, have a lower score than `d1` in the network's output vector `(score img)`, i.e. that `d1` is the maximum and thus the predicted breed.

Vehicle supports type synonyms. For instance, instead of repeatedly writing `Tensor Rat[28,28]` and `Index 20`,
we can declare new types called `Image` and `Dog`. Like this, the signature of
`isFirstChoice` becomes more meaningful.

``` vehicle
    type Image  = Tensor Rat [28, 28]
    type Dog    = Index 20

    isFirstChoice : Image -> Dog -> Bool
```

We can now write the property `existsGreatDane` stating that at least one image will be classified as a Great Dane.
`exists img` checks whether the predicate `isFirstChoice img greatDane` holds for some image `img`,
where we have declared `greatDane` as the dog breed number 1.

``` vehicle
    GreatDane = 1

    @property
    existsGreatDane : Bool
    existsGreatDane = exists img . isFirstChoice img greatDane
```

But wait! If we compile this specification as is and pass it to the verifier, `existsGreatDane` will most likely
hold for a nonsensical image. To avoid this, we must restrict our property to valid images only.
We consider an image valid if all its pixels are valid, i.e., normalized to be in the range 0 to 1.

``` vehicle
    validPixel : Rat -> Bool
    validPixel p = 0 <= p <= 1

    validImage : Image -> Bool
    validImage img = forall i j . validPixel (img ! i ! j)
```

Here, `forall` conveniently infers the type of `i` and `j` as `Index 28`, based on the type of `img`, i.e. a 2-dimensional tensor of size 28x28. It then uses `i`, `j` to refer to each of the pixels in the image, i.e. `x ! i ! j`.

We can now rewrite our property restricted to valid images only.

``` vehicle
    @property
    existsGreatDane : Bool
    existsGreatDane = exists img . (validImage img) and (isFirstChoice img GreatDane)
```

### A more meaningful property (DoesNotConfuseBigAndSmall)

Let us now consider a more meaning property. We want to make sure the network is not confusing small dogs with big dogs. That is,
if the first choice for an image is a big or small dog, the second choice should be too.

We need an auxiliary function `isSecondChoice`. Given an image `img` and a breed `d2`, it checks whether `d2` is the network's second breed choice for `img`. This is the case if any other breeds `d3` that different from `d1` (the first choice) and `d2` (the second choice) have lower scores than `d1` and `d2`.

The `let ... in ...` clause lets us define a variable to be used after `in`. In this case, we give the name `scores` to the output of the network for `img`, i.e. `score img`.

``` vehicle
    isSecondChoice : Image -> Dog -> Bool
    isSecondChoice img d2 =
    let scores = score img in
    exists d1 . (isFirstChoice img d1) and (forall d . d != d1 and d != d2 => scores ! d2 > scores ! d)
```

``` vehicle
    noConfusionWith : Image -> List Dog -> List Dog -> Bool
    noConfusionWith img dogsList1 dogsList2 =
    forall d1 in dogsList1 .
        forall d2 in dogsList2 .
        not (isFirstChoice img d1 and isSecondChoice img d2)
```

`noConfusionWith` takes an image `img` and two lists of breeds, and checks whether the first and second choice for `img`
are in the same list. The nested `forall` quantifiers take all possible pairs of dogs, one in each list, and check that they are not the first and second choice of the network.

We split the breeds into two lists, `bigDogs` and `smallDogs`.
Lists in vehicle can hold an arbitrary number of elements of a single type and are declared using `[x_1, ..., x_n]`.
We define `smallDogs`and `bigDogs` with type `List Dog` (recall that this is really `List Index 20`).

``` vehicle
    greatDane       = 1
    germanShepherd  = 2
    chihuahua       = 11
    pekinese        = 12

    smallDogs : List Dog
    smallDogs = [chihuahua, pekinese]

    bigDogs : List Dog
    bigDogs = [greatDane, germanShepherd]
```

We are finally ready to state our desired property `doesNotConfuseBigAndSmall`: for all images `img` that are valid,
`noConfusionWith` will return true for `img` `bigDogs` and `smallDogs`, i.e. if all valid images have their first and second
choice in the same list.

(Comment on not using for loops to transverse the lists.)

``` vehicle
    @property
    doesNotConfuseBigAndSmall : Bool
    doesNotConfuseBigAndSmall =
        forall img . validImage img => noConfusionWith img bigDogs smallDogs
```

## Declaration of a dataset

(do we need the dataset for this example? I am not sure)

## References

Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.
