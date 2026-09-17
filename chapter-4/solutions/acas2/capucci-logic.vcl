--------------------------------------------------------------------------------
-- The Capucci / additive QLL differentiable logic, verbatim from
-- chapter-code/capucci-pdt/fashionRobustness-capucci.vcl (the chapter's training
-- specification), kept here so that this folder is self-contained. split-spec.py appends
-- it to each one-property training specification, substituting the hardness p given on
-- its command line for the `p = 2.0` below. This file is not a complete specification on
-- its own (it declares no network or property) and is not meant to be compiled directly.
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
