--------------------------------------------------------------------------------
-- Doggos

numberOfDogs = 20
type Dog = Index numberOfDogs

unknownDog     = 0

greatDane      = 1
germanShepherd = 2

chihuahua  = 11
pekinese   = 12

smallDogs : List Dog
smallDogs = [chihuahua, pekinese]

bigDogs : List Dog
bigDogs = [greatDane, germanShepherd]

--------------------------------------------------------------------------------
-- Network

type Image = Tensor Rat [28, 28]
type Score = Rat

@network
score : Image -> Vector Score numberOfDogs

validPixel : Rat -> Bool
validPixel p = 0 <= p <= 1

validImage : Image -> Bool
validImage img = forall i j . validPixel (img ! i ! j)

--------------------------------------------------------------------------------
-- Predicates

isFirstChoice : Image -> Dog -> Bool
isFirstChoice img d1 =
  let scores = score img in
  forall d . d != d1 => scores ! d1 > scores ! d

isSecondChoice : Image -> Dog -> Bool
isSecondChoice img d2 =
  let scores = score img in
  exists d1 . (isFirstChoice img d1) and (forall d . d != d1 and d != d2 => scores ! d2 > scores ! d)

noConfusionWith : Image -> List Dog -> List Dog -> Bool
noConfusionWith img dogsList1 dogsList2 =
  forall d1 in dogsList1 .
    forall d2 in dogsList2 .
      not (isFirstChoice img d1 and isSecondChoice img d2)


-------------------------------------------------------------------------------
-- Properties

@property
doesNotConfuseBigAndSmall : Bool
doesNotConfuseBigAndSmall =
  forall img . validImage img => noConfusionWith img bigDogs smallDogs
