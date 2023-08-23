--The input for the network is a 28 * 28 image
type Image = Tensor Rat [28, 28]

--A label is an int between 0 and 9
type Label = Index 10


--All pixels in the image have values between 0 and 1
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

--The network takes an image and returns a vector of scores
@network
classifier : Image -> Vector Rat 10

--The classifier scores a given label above all others
advises : Image -> Label -> Bool
advises image label = forall j . j != label => classifier image ! label > classifier image ! j


--The radius of the epsilon ball that we are checking robustness within
@parameter
epsilon : Rat

--Every pixel in the perturbation is less than or equal to epsilon
boundedByEpsilon : Image -> Bool
boundedByEpsilon perturbation = forall i j . -epsilon <= perturbation ! i ! j <= epsilon

--Check that every valid perturbation of an image is classified as the given label
robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
	let perturbedImage = image - perturbation in
	boundedByEpsilon perturbation and validImage perturbedImage =>
		advises perturbedImage label


--Take two datasets one containing an image and one a label
@dataset
imageDataset : Vector Image 1

@dataset
labelDataset : Vector Label 1

--Test the image for robustness around the label
@property
robust : Bool
robust = robustAround (imageDataset ! 0) (labelDataset ! 0)
