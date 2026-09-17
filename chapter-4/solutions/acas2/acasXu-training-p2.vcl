--------------------------------------------------------------------------------
-- TRAINING specification for the ACAS XU networks: Property 3 only.
--
-- This is chapter-2/chapter-code/acasXu.vcl with ONE change, in `advises`, which is
-- needed to compile the property into a loss function with Vehicle 0.28.0. The original
-- file, unchanged, is kept next to this one as acasXu.vcl and is used for verification.
--
-- The original definition is
--
--   advises i x = forall j . i != j => normAcasXu x ! i < normAcasXu x ! j
--
-- Under `vehicle compile loss` (Vehicle 0.28.0) the guard `i != j =>` is compiled with
-- its polarity inverted: the comparison is kept only for j = i (where it is trivially
-- `s < s`, i.e. margin 0) and every j != i receives the logic's true element. The
-- compiled `advises` is therefore the constant 0 whatever the network does, and so is
-- the compiled `property3`, with zero gradient. (The verifier back-end is unaffected;
-- Chapter 2's Marabou results use the original file.)
--
-- The workaround is the one Chapter 4 already uses for Fashion MNIST
-- (chapter-code/fmnist-robustness.vcl, capucci-pdt/fashionRobustness-capucci.vcl):
-- drop the guard and make the comparison non-strict, so that the j = i case is
-- trivially true and needs no guard.
--
--   advises i x = forall j . normAcasXu x ! i <= normAcasXu x ! j
--
-- The two forms differ only on ties. Under the training form, `not (advises COC x)`
-- asks that some other advisory scores *strictly* below clear-of-conflict, which is
-- exactly what Marabou checks after its own strict-to-non-strict conversion (see the
-- warning it prints, and vehicle issue 74).
--
-- Everything from here down to the `qllAdditive` logic at the end is verbatim from
-- acasXu.vcl apart from `advises`. The logic block is verbatim from
-- chapter-code/capucci-pdt/fashionRobustness-capucci.vcl.
--------------------------------------------------------------------------------
-- Full specification of the ACAS XU networks

-- Taken from Appendix VI of "Reluplex: An Efficient SMT Solver for Verifying
-- Deep Neural Networks" at https://arxiv.org/pdf/1702.01135.pdf

-- Comments describing the properties are taken directly from the text.

--------------------------------------------------------------------------------
-- Utilities

-- The value of the constant `pi`.
pi = 3.141592

--------------------------------------------------------------------------------
-- Inputs

-- We first define a new name for the type of inputs of the network.
-- In particular, it takes inputs of the form of a tensor of 5 real numbers.

type Input = Tensor Real [5]

-- Next we add meaningful names for the indices.
-- The fact that all tensor types come annotated with their size means that it
-- is impossible to mess up indexing into vectors, e.g. if you changed
-- `distanceToIntruder = 0` to `distanceToIntruder = 5` the specification would
-- fail to type-check.

distanceToIntruder = 0   -- measured in feet
angleToIntruder    = 1   -- measured in radians
intruderHeading    = 2   -- measured in radians
speed              = 3   -- measured in feet/second
intruderSpeed      = 4   -- measured in feet/second

--------------------------------------------------------------------------------
-- Outputs

-- Outputs are also a tensor of 5 reals. Each one representing the score
-- for the 5 available courses of action.

type Output = Tensor Real [5]

-- Again we define meaningful names for the indices into output vectors.

clearOfConflict = 0
weakLeft        = 1
weakRight       = 2
strongLeft      = 3
strongRight     = 4

--------------------------------------------------------------------------------
-- The network

-- Next we use the `network` annotation to declare the name and the type of the
-- neural network we are verifying. The implementation is passed to the compiler
-- via a reference to the ONNX file at compile time.

@network
acasXu : Input -> Output

--------------------------------------------------------------------------------
-- Normalisation

-- As is common in machine learning, the network operates over
-- normalised values, rather than values in the problem space
-- (e.g. using standard units like m/s).
-- This is an issue for us, as we would like to write our specification in
-- terms of the problem space values .
-- Therefore before applying the network, we first have to normalise
-- the values in the problem space.

-- For clarity, we therefore define a new type synonym
-- for unnormalised input vectors which are in the problem space.
type UnnormalisedInput = Tensor Real [5]

-- Next we define the minimum and maximum values that each input can take.
-- These correspond to the range of the inputs that the network is designed
-- to work over.
minimumInputValues : UnnormalisedInput
minimumInputValues = [0, -pi, -pi, 0, 0]

maximumInputValues : UnnormalisedInput
maximumInputValues = [60261.0, pi, pi, 1200.0, 1200.0]

-- We can therefore define a simple predicate saying whether a given input
-- vector is in the right range.
validInput : UnnormalisedInput -> Bool
validInput x = forall i .
  minimumInputValues ! i <= x ! i <= maximumInputValues ! i

-- Then the mean values that will be used to scale the inputs.
meanScalingValues : UnnormalisedInput
meanScalingValues = [19791.091, 0.0, 0.0, 650.0, 600.0]

-- We can now define the normalisation function that takes an input vector and
-- returns the unnormalised version.
normalise : UnnormalisedInput -> Input
normalise x = foreach i .
  (x ! i - meanScalingValues ! i) / (maximumInputValues ! i - minimumInputValues ! i)

-- Using this we can define a new function that first normalises the input
-- vector and then applies the neural network.
normAcasXu : UnnormalisedInput -> Output
normAcasXu x = acasXu (normalise x)

-- A constraint that says the network chooses output `i` when given the
-- input `x`. We must necessarily provide a finite index that is less than 5
-- (i.e. of type Index 5). The `a ! b` operator lookups index `b` in vector `a`.
advises : Index 5 -> UnnormalisedInput -> Bool
advises i x = forall j . normAcasXu x ! i <= normAcasXu x ! j


--------------------------------------------------------------------------------
-- Property 3

-- If the intruder is directly ahead and is moving towards the
-- ownship, the score for COC will not be minimal.

-- Tested on: all networks except N_{1,7}, N_{1,8}, and N_{1,9}.

directlyAhead : UnnormalisedInput -> Bool
directlyAhead x =
  1500  <= x ! distanceToIntruder <= 1800 and
  -0.06 <= x ! angleToIntruder    <= 0.06

movingTowards : UnnormalisedInput -> Bool
movingTowards x =
  x ! intruderHeading >= 3.10  and
  x ! speed           >= 980   and
  x ! intruderSpeed   >= 960

@property
property3 : Bool
property3 = forall x .
  validInput x and directlyAhead x and movingTowards x =>
  not (advises clearOfConflict x)

--------------------------------------------------------------------------------
-- A custom differentiable logic: the additive QLL / Capucci logic
--
-- Vehicle's default logic is not the only way to turn a specification into a loss.
-- A differentiable logic is a choice of interpretation for each connective, and
-- Vehicle lets us supply one by declaring a `DifferentiableTensorLogic` and naming
-- it on the Python side (see pdt-Capucci.py).
--
-- Note the truth direction. Here `trueElement` is -infinity and `falseElement` is
-- +infinity, so *smaller is truer* and the loss is unbounded below. That is the usual
-- convention in this literature: the less error there is, the truer the formula.
--
-- `p` controls how sharply the conjunction and disjunction approximate min and max.
-- As p grows the log-sum-exp terms approach the hard operations; small p gives a
-- smoother surface with gradients that reach further.
--
-- `p` is deliberately a plain definition rather than an @parameter. Vehicle 0.27.1
-- cannot compile a logic that refers to an @parameter: doing so fails with
--   Internal scoping error: declaration 'User.p' not found in scope
-- raised as a developerError from FunctionaliseResources. Until that is fixed, p is a
-- compile-time constant and changing it means editing the line below.

p : Real
p = 2.0   -- as in the chapter (acasXu-training.vcl uses 5.0; this file differs only here)

qllAdditive : DifferentiableTensorLogic
qllAdditive =
  { trueElement                = -infinity
  , falseElement               = infinity
  , pointwiseNegation          = \x -> -x
  , pointwiseConjunction       = \{dims} x y -> (const (1/p) dims) * log(exp(const p dims * x) + exp(const p dims * y))
  , pointwiseDisjunction       = \{dims} x y -> -(const (1/p) dims) * log(exp(const (-p) dims * x) + exp(const (-p) dims * y))
  , pointwiseLessThan          = \x y -> x - y
  , pointwiseLessEqualThan     = \x y -> x - y
  , pointwiseGreaterThan       = \x y -> y - x
  , pointwiseGreaterEqualThan  = \x y -> y - x
  , pointwiseEqual             = \x y -> max (x - y) (y - x)
  , pointwiseNotEqual          = \x y -> - max (x - y) (y - x)
  , reduceConjunction          = \{dims} xs -> (1/p) * log(reduceAdd (exp (const p dims * xs)))
  , reduceDisjunction          = \{dims} xs -> (1/p) * log(reduceAdd (exp (const (-p) dims * xs)))
  }
