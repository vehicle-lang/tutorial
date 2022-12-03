
# Vehicle Tutorial 

## Contents of tutorial

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


## Motivation
- introduction of a dataset
- describe a property in words
- example with NN?
- example of property not holding 
- fix in vehicle!

- Section0 : Introduce design choice for property specification. (some need data, some need networks, some none of them).

## How to write Specs

### Declaration of the network

```agda
    network score : Tensor Rat[28, 28] -> Vector Rat 20
```

Function types are declared using `:` and function arguments are separated with `->`. 
For example, the above function is called `score` and will take in a 2-dimension tensor of size 28 x 28
filled with rationals - an image -, do something with it and then return a vector of 20 rationals -a score for each of the 0 to 19 dog breeds-.
Networks are annotated with the keyword `network` before their name. 

### Declaration of properties

We can define properties that will later be passed on to the verifier. 
For instance, imagine you want to check that there is at least an image that will be classified as a Great Dane.

To write this property, we first need to declare an auxiliary predicate `isFirstChoice` that will 
check whether the network will classify a given image `img` as a specific dog breed `d1`. In other words, that the 
vector returned by `score` for `img` will have its maximum in index `d1`.

```agda
    isFirstChoice : Tensor Rat[28, 28] -> Index 20 -> Bool
    isFirstChoice img d1 =
        forall d2 . d2 != d1 => (score img) ! d1 > (score img) ! d2
```
Vector lookup is done with `!`. For example `(score img) ! d1` will return the score for the dog breed in position `d1`. 
The `forall` quantifier allows us to check that all dog breeds `d2`, such that `d2` is different from `d1`, have a lower score than `d1`. 

Note how the type of `d1` is `Index 20`. The set of valid instances of this type are the natural numbers {0, ..., 19}.
By using the type `Index 20` instead of simply `Nat`, we make sure that no out-of-bound errors will arise from using `!` 
in the vector `score img` of size 20.

To improve readability, vehicle allows us to introduce type synonyms with the keyword `type`. For example, 
instead of repeatedly writing `Tensor Rat[28,28]`, we can give it the more meaningful name `Image`. 

```agda
    type Image  = Tensor Rat [28, 28]
    type Dog    = Index 20

    isFirstChoice : Image -> Dog -> Bool
``` 

We can now check whether at least one image will be classified as a Great Dane. 

```agda
    property existsGreatDane : Bool
    existsGreatDane = exists x . isFirstChoice x GreatDane
```
As with networks, we declare properties with the keyword `property` before the function's name. 
We use `exists` to check whether the predicate `isFirstChoice x GreatDane` holds for some image `x`.

However, the previous property may hold for a nonsensical image. To avoid this, we must 
restrict our property to valid images only!
Usually, images are normalized to have pixels in the range 0 to 1. In our case, an image is valid if all of its pixels 
are in the range (0, 1).

```agda
    validPixel : Rat -> Bool
    validPixel p = 0 <= p <= 1

    validImage : Image -> Bool
    validImage x = forall i j . validPixel x ! i ! j
```
Here, `forall` conveniently infers the type of `i` and `j` as `Index 28`, based on the type of `x`, i.e. a 2-dimensional tensor of size 28x28. It then uses `i`, `j` as iterators to access each of the pixels in the image, i.e. `x ! i ! j`.

We can now rewrite our property restricted to valid images only.

```agda
    property existsGreatDane : Bool
    existsGreatDane = exists x . (validImage x) and (isFirstChoice x GreatDane)
``` 
    
### Declaration of a dataset

- declaration of dataset
    







