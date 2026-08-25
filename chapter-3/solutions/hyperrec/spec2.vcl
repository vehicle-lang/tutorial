--------------------------------------------------------------------------------
-- Inputs

inputSize = 30
type Input = Tensor Real [inputSize]


validInput : Input -> Bool
validInput x = forall i . 0 <= x ! i  <= 1

--------------------------------------------------------------------------------
-- Outputs

type Output = Tensor Real [2]
type Label = Index 2

pos = 0
neg = 1

--------------------------------------------------------------------------------
-- Network

@network
classifier : Input -> Output

advises : Input -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Dataset

@parameter(infer=True)
n : Nat

@dataset
-- inputs : Tensor Real [n, inputSize]
inputs : Vector (Vector Real 30) 5


inputTranspose : Vector (Vector Real 5) 30
inputTranspose = foreach i . foreach j . inputs ! j ! i

-- identity : Tensor Real [2, 3]
-- identity = [ [1, 0, 1],  [0, 1, 0] ]

-- vectorMin : Tensor Real [inputSize, 5]  -> Index 5 -> Bool
-- vectorMin x i = forall j k. x ! j ! i  <= x ! j ! k

vectorMin :  Index 30 -> Index 5 -> Bool
vectorMin i j  = forall k . inputs ! j ! i  <= inputs ! j ! k

vectorMax :  Index 30  -> Index 5 -> Bool
vectorMax i j = forall k . inputs ! j ! k  <= inputs ! j ! i

@property
property : Bool
property = forall x. forall j i l k.  validInput x and vectorMin i j and vectorMax l k and x ! i  >= inputs ! j ! i and inputs ! k ! l >=  x ! l =>  advises x pos

{-
vectorMax : Vector Real 5 -> Index 5 -> Vector Bool 5
vectorMax x i = foreach j . x ! j <= x ! i

vectorOut : Tensor Real [inputSize, n] -> Vector Real inputSize
vectorOut x = foreach i . x ! i -}

-- @property
-- property : Bool
-- property = True

{-
vectorMax :  Input
vectorMax = foreach i . 1 -- maxList (inputTranspose ! i)

vectorMin : Input
vectorMin = foreach i . 0 --minList (inputTranspose ! i)

vectorMax :  Input
vectorMax = foreach i . 1 -- maxList (inputTranspose ! i)

hyperRectangle : Input -> Bool
hyperRectangle x = forall i . vectorMin ! i  <= x ! i <= vectorMax ! i

@property
property : Bool
property = forall x . hyperRectangle x =>  advises x pos
-}
