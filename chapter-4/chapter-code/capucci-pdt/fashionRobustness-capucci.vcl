-- Training specification for the Capucci property-driven training experiment.
--
-- The property below follows chapter-3/exercises/FMNIST/fashionRobustness-solution.vcl,
-- the specification Exercise #7 verifies against, so that training and verification
-- concern as nearly as possible the same property -- the first of the three difficulties
-- discussed in the chapter.
--
-- Two things are added or changed, both required to compile the property into a loss:
--
--   1. the differentiable logic declaration at the end, which Vehicle needs in order to
--      translate the property into a loss function. It does not affect the property.
--
--   2. `advises` is stated in the non-strict form. Exercise #7 writes
--        forall j . j != label => classifier image ! label > classifier image ! j
--      but `j != label` compares two values of type `Index 10`, and the loss backend
--      rejects that with "Loss functions do not yet support compilation of
--      'CompareIndex'". The guard cannot simply be dropped while keeping `>`, because
--      the case j = label would then demand a score strictly greater than itself. So the
--      comparison is weakened to `>=`, which makes the j = label case trivially true and
--      the guard unnecessary.
--
--      The two forms therefore differ only on ties: Exercise #7's property is strict,
--      this one admits a tie between the advised label and another. The chapter's other
--      training specification, fmnist-robustness.vcl, uses the same workaround for the
--      same reason.

--The input for the network is a 28 * 28 image
type Image = Tensor Real [28, 28]

--A label is an int between 0 and 9
type Label = Index 10


--All pixels in the image have values between 0 and 1
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

--The network takes an image and returns a vector of scores
@network
classifier : Image -> Tensor Real [10]

--The classifier scores a given label above all others
advises : Image -> Label -> Bool
advises image label = forall j . classifier image ! label >= classifier image ! j


--The radius of the epsilon ball that we are checking robustness within
@parameter
epsilon : Real

--Every pixel in the perturbation is less than or equal to epsilon
boundedByEpsilon : Image -> Bool
boundedByEpsilon perturbation = forall i j . -epsilon <= perturbation ! i ! j <= epsilon

--Check that every valid perturbation of an image is classified as the given label
robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
	let perturbedImage = image - perturbation in
	boundedByEpsilon perturbation and validImage perturbedImage =>
		advises perturbedImage label


--The size of the data set. The `infer` option means the compiler works this
--out from the data sets below, so it need not be supplied on the command line
@parameter(infer=True)
n : Nat

--Take two datasets

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

--Test the image for robustness around the label
@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)

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
p = 2.0

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
