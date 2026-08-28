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

-- The supplied model expects readings normalised from [-4, 4] to [0, 1].
normalise : InputVector -> InputVector
normalise x = foreach i . (x ! i + 4.0) / 8.0

--------------------------------------------------------------------------------
-- Parameterised centred stability challenge

@parameter
maxDistance : Real

@parameter
maxSensorChange : Real

@parameter
maxSpeed : Real

stableNearCenterInput : InputVector -> Bool
stableNearCenterInput x =
  -maxDistance <= x ! currentSensor <= maxDistance and
  -maxSensorChange <= (x ! currentSensor - x ! previousSensor) <= maxSensorChange

-- The two meaningful constraints imply this range for the previous reading.
-- Vehicle 0.26.1 needs the finite range written explicitly.
derivedPreviousSensorBound : InputVector -> Bool
derivedPreviousSensorBound x =
  -(maxDistance + maxSensorChange) <= x ! previousSensor <=
    maxDistance + maxSensorChange

smallControllerOutput : InputVector -> Bool
smallControllerOutput x = let y = controller (normalise x) ! velocity in
  -maxSpeed <= y <= maxSpeed

@property
sensorNearCenter : Bool
sensorNearCenter = forall x .
  stableNearCenterInput x and derivedPreviousSensorBound x =>
    smallControllerOutput x
