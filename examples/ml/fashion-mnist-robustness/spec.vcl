-- Try p = 1, 2, 4.
@parameter
p : Real

@logic
capucciAdditive : DifferentiableLogic
capucciAdditive = 
  { trueElement                = -infinity
  , falseElement               = infinity
  , pointwiseNegation          = \x -> -x
  , pointwiseConjunction       = \x y -> (1/p) * log(exp(p * x) + exp(p * y))
  , pointwiseDisjunction       = \x y -> -(1/p) * log(exp(-p * x) + exp(-p * y))
  , pointwiseLessThan          = \x y -> x - y
  , pointwiseLessEqualThan     = \x y -> x - y
  , pointwiseGreaterThan       = \x y -> y - x
  , pointwiseGreaterEqualThan  = \x y -> y - x
  , pointwiseEqual             = \x y -> max (x - y) (y - x)
  , pointwiseNotEqual          = \x y -> - max (x - y) (y - x)
  , reduceConjunction          = \xs -> (1/p) * log(reduceAdd (exp (p * xs)))
  , reduceDisjunction          = \xs -> (1/p) * log(reduceAdd (exp (-p * xs)))
  }

