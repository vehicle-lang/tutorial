type Image = Tensor Real [28, 28]
type Label = Index 10

@network
classifier : Image -> Tensor Real [10]

advises : Image -> Label -> Bool
advises x i = forall j . classifier x ! i >= classifier x ! j

@parameter
epsilon : Real

-- No implication anywhere. `delta` is squashed into [-1,1] and scaled by epsilon, so
-- the perturbation is inside the epsilon-ball by construction; the perturbed image is
-- clamped into [0,1], so it is a valid image by construction. There is therefore no
-- antecedent that can be false, and hence no way for the property to be satisfied
-- vacuously.
squash : Real -> Real
squash x = max (-1.0) (min 1.0 x)

clamp01 : Real -> Real
clamp01 x = max 0.0 (min 1.0 x)

robustAround : Image -> Label -> Bool
robustAround image label = forall (delta : Image) .
  let perturbed = foreach i j . clamp01 (image ! i ! j - epsilon * squash (delta ! i ! j)) in
  advises perturbed label

@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)
