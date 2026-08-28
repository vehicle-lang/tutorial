--------------------------------------------------------------------------------
-- Inputs and outputs

type InputVector = Tensor Real [2]

currentSensor  = 0
previousSensor = 1

type OutputVector = Tensor Real [1]

velocity = 0

--------------------------------------------------------------------------------
-- Network

@network
controller : InputVector -> OutputVector

-- Normalises the input values from the range [-4, 4] metres to [0, 1].
normalise : InputVector -> InputVector
normalise x = foreach i . (x ! i + 4.0) / 8.0

--------------------------------------------------------------------------------
-- Original safety property

safeInput : InputVector -> Bool
safeInput x = forall i . -3.25 <= x ! i <= 3.25

safeOutput : InputVector -> Bool
safeOutput x = let y = controller (normalise x) ! velocity in
  -1.25 < y + 2 * (x ! currentSensor) - (x ! previousSensor) < 1.25

@property
safe : Bool
safe = forall x . safeInput x => safeOutput x

--------------------------------------------------------------------------------
-- Exercise 4: stable readings near the road centre

stableNearCenterInput : InputVector -> Bool
stableNearCenterInput x =
  -0.2 <= x ! currentSensor <= 0.2 and
  -0.2 <= (x ! currentSensor - x ! previousSensor) <= 0.2

-- This is not an extra assumption: the two constraints above imply the bound.
-- Vehicle 0.26.1 needs it written explicitly to find a finite range for every
-- network input.
derivedPreviousSensorBound : InputVector -> Bool
derivedPreviousSensorBound x =
  -0.4 <= x ! previousSensor <= 0.4

smallControllerOutput : InputVector -> Bool
smallControllerOutput x = let y = controller (normalise x) ! velocity in
  -0.5 <= y <= 0.5

@property
sensorNearCenter : Bool
sensorNearCenter = forall x .
  stableNearCenterInput x and derivedPreviousSensorBound x =>
    smallControllerOutput x
