--------------------------------------------------------------------------------
-- Classes
numberOfClasses = 10
type Class = Index numberOfClasses

tshirt    = 0
trouser   = 1
pullover  = 2
dress     = 3
coat      = 4
sandal    = 5
shirt     = 6
sneaker   = 7
bag       = 8
boot      = 9

clothes : List Class
clothes = [tshirt, trouser, pullover, dress, coat, shirt]

complementsAndShoes : List Class
complementsAndShoes = [sandal, sneaker, boot]

--------------------------------------------------------------------------------
-- Network

type Image = Tensor Rat [28, 28]
type Score = Rat

@network
score : Image -> Vector Score numberOfClasses

validPixel : Rat -> Bool
validPixel p = 0 <= p <= 1

validImage : Image -> Bool
validImage img = forall i j . validPixel (img ! i ! j)

--------------------------------------------------------------------------------
-- Predicates

isFirstChoice : Image -> Class -> Bool
isFirstChoice img c1 =
  let scores = score img in
  forall c . c != c1 => scores ! c1 > scores ! c

isSecondChoice : Image -> Class -> Bool
isSecondChoice img c2 =
  let scores = score img in
  exists c1 . (isFirstChoice img c1) and (forall c . c != c1 and c != c2 => scores ! c2 > scores ! c)

noConfusionWith : Image -> List Class -> List Class -> Bool
noConfusionWith img classList1 classList2 =
  forall c1 in classList1 .
    forall c2 in classList2 .
      not (isFirstChoice img c1 and isSecondChoice img c2)


-------------------------------------------------------------------------------
-- Properties

@property
doesNotConfuseClotheswithOthers : Bool
doesNotConfuseClotheswithOthers =
  forall img . validImage img => noConfusionWith img clothes complementsAndShoes