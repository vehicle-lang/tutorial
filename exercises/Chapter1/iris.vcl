--------------------------------------------------------------------------------
-- Inputs

-- define a new name for the type of inputs of the network.
type InputVector = Vector Rat 4

-- add meaningful names for the input indices.
sepalLength = 0   -- measured in centimetres
sepalWidth  = 1   -- measured in centimetres
petalLength = 2   -- measured in centimetres
petalWidth  = 3   -- measured in centimetres

--------------------------------------------------------------------------------
-- Outputs

-- define output format - a vector of 3 rationals, 
-- each representing the score for the 3 classes.

type OutputVector = Vector Rat 3

-- add meaningful names for the output indices.
setosa      = 0
versicolor  = 1
virginica   = 2

--------------------------------------------------------------------------------
-- Network

-- use the `network` annotation to declare the name and the type of the network
@network
iris : InputVector -> OutputVector

--------------------------------------------------------------------------------
-- Check input data validity 
-- Define normal input ranges (based on training data - min, max values)
normalSepalLength : InputVector -> Bool
normalSepalLength x = 4.3 <= x ! sepalLength <= 7.9

normalSepalWidth : InputVector -> Bool
normalSepalWidth x = 2.0 <= x ! sepalWidth <= 4.4

normalPetalLength : InputVector -> Bool
normalPetalLength x = 1.0 <= x ! petalLength <= 6.9

normalPetalWidth : InputVector -> Bool
normalPetalWidth x = 0.1 <= x ! petalWidth <= 2.5

validInput : InputVector -> Bool
validInput x = normalSepalLength x and normalSepalWidth x 
    and normalPetalLength x and normalPetalWidth x
    and x ! sepalLength > x ! sepalWidth
    and x ! petalLength > x ! petalWidth

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Property 0

-- to see files work fine

@property
property0 : Bool
property0 = exists x . validInput x => 
    ((iris x ! versicolor <= 0) and (iris x ! versicolor > 0))

--------------------------------------------------------------------------------
-- Property 1

-- If (sepal length - sepal width) is beteween 1.3cm and 1.5cm, 
-- it is setosa

slAndSw : InputVector -> Bool
slAndSw x =
    1.3  <= x ! sepalLength - x ! sepalWidth <= 1.5

isSetosa : InputVector -> Bool
isSetosa x = 
    let scores = iris x in 
    forall d . d != setosa => scores ! setosa > scores ! d

@property
property1 : Bool
property1 = forall x . validInput x and slAndSw x and x ! sepalWidth > 3 =>
    isSetosa x

--------------------------------------------------------------------------------
-- Property 2

-- If the sepal length (sl) is shorter than 6 and 
-- peteal length (pl) is shorter than 2, then it is setosa

slAndPl : InputVector -> Bool
slAndPl x =
    x ! sepalLength <= 6 and
    x ! petalLength <= 2

@property
property2 : Bool
property2 = forall x . validInput x and slAndPl x =>
    isSetosa x

--------------------------------------------------------------------------------
-- Property 3

-- If the sepal length (sl) is shorter than 6 and 
-- peteal width (pw) is shorter than 0.8, then it is setosa

slAndPw : InputVector -> Bool
slAndPw x =
    x ! sepalLength <= 6 and
    x ! petalWidth <= 0.8

@property
property3 : Bool
property3 = forall x . validInput x and slAndPw x =>
    isSetosa x

--------------------------------------------------------------------------------
-- Property 4

-- If the sepal length (sl) is longer than 7.5, 
-- then it is virginica

longSl : InputVector -> Bool
longSl x =
    x ! sepalLength >= 7.5 

isVirginica : InputVector -> Bool
isVirginica x = 
    let scores = iris x in 
    forall d . d != virginica => scores ! virginica > scores ! d

@property
property4 : Bool
property4 = forall x . validInput x and longSl x =>
    isVirginica x

--------------------------------------------------------------------------------
-- Property 5

-- If the petal length is shorter than 2 and the petal width is shorter than 0.5,
-- then it is setosa

smallPetal : InputVector -> Bool
smallPetal x =
    x ! petalLength < 2 and x ! petalWidth < 0.5

@property
property5 : Bool
property5 = forall x . validInput x and smallPetal x =>
    isSetosa x

--------------------------------------------------------------------------------
-- Property 6

-- If the petal length (sl) is longer than 6 and the petal width is longer than 2, 
-- then it is virginica

bigPetal : InputVector -> Bool
bigPetal x =
    x ! petalLength > 6 and x ! petalWidth > 2

@property
property6 : Bool
property6 = forall x . validInput x and bigPetal x =>
    isVirginica x

--------------------------------------------------------------------------------