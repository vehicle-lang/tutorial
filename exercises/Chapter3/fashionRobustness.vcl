-- Inputs and outputs definitions, check dimensions of in and out for the onnx?
type Image = Tensor Rat [28, 28]

type Label = Index 10


--Questions; what are the . and ! symbols, what is the logic here?/Check that all Rat's in image are between 0 and 1
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1


@network
classifier : Image -> Vector Rat 10

--Function advises takes an image and returns a function from label to bool. x is the image, i is the label?
advises : Image -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

@parameter
epsilon : Rat

--Questions; what is -epsilon, negative epsilon?
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
	let perturbedImage = image - perturbation in
	boundedByEpsilon perturbation and validImage perturbedImage =>
		advises perturbedImage label


@dataset
image : Vector Image 1 

@dataset
label : Vector Label 1

@property
robust : Bool
robust = robustAround (image ! 0) (label ! 0)
