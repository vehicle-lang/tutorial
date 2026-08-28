--------------------------------------------------------------------------------
-- Inputs and outputs

@tensor
record Input where
  { currentSensor  : Real
  , previousSensor : Real
  }

@tensor
record Output where
  { velocity : Real
  }

--------------------------------------------------------------------------------
-- Network

@network
controller : Input -> Output

-- The supplied model expects both readings normalised from [-4, 4] to [0, 1].
normalise : Input -> Input
normalise x =
  { currentSensor  = (x.currentSensor + 4.0) / 8.0
  , previousSensor = (x.previousSensor + 4.0) / 8.0
  }

--------------------------------------------------------------------------------
-- Safety property

safeInput : Input -> Bool
safeInput x =
  -3.25 <= x.currentSensor  <= 3.25 and
  -3.25 <= x.previousSensor <= 3.25

safeOutput : Input -> Bool
safeOutput x = let y = (controller (normalise x)).velocity in
  -1.25 < y + 2 * x.currentSensor - x.previousSensor < 1.25

@property
safe : Bool
safe = forall x . safeInput x => safeOutput x
