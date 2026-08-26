--------------------------------------------------------------------------------
-- Classification robustness under the Euclidean distance (Exercise #6)

-- The chapter defines the epsilon-ball using the L-infinity norm, which asks
-- that *every* pixel moves by at most epsilon. Here we use the Euclidean
-- (L2) norm instead, which bounds the *total* change across the whole image:
--
--   |p|_2 = sqrt (sum_ij p_ij^2)  <=  epsilon
--
-- We compare squared quantities rather than taking a square root, which is
-- equivalent for non-negative values:
--
--   sum_ij p_ij^2  <=  epsilon^2
--
-- Note that this makes the constraint *polynomial* rather than linear, so it
-- falls outside the linear real arithmetic that Marabou supports. A solver
-- with polynomial support, such as vibecheck, is needed.

-- The input for the network is a 28 * 28 image
type Image = Tensor Real [28, 28]

-- A label is an integer between 0 and 9
type Label = Index 10

-- All pixels in a valid image have values between 0 and 1
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

@network
classifier : Image -> Tensor Real [10]

-- The classifier scores the given label above all others
advises : Image -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

-- The radius of the Euclidean ball we check robustness within
@parameter
epsilon : Real

-- The squared Euclidean length of a perturbation, summed over every pixel.
-- `reduceAdd` sums the elements of a tensor, and `p * p` squares it
-- element-wise, so this is sum_ij p_ij^2.
squaredNorm : Image -> Real
squaredNorm p = reduceAdd (p * p)

-- |p|_2 <= epsilon, stated without a square root
withinEuclideanBall : Image -> Bool
withinEuclideanBall p = squaredNorm p <= epsilon * epsilon

-- Every perturbation inside the Euclidean ball leaves the advised label alone
robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
  let perturbedImage = image - perturbation in
  withinEuclideanBall perturbation and validImage perturbedImage =>
    advises perturbedImage label

-- The size of the data set, inferred by the compiler from the data sets below
@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)
