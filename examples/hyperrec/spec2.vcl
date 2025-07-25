--------------------------------------------------------------------------------
-- Inputs

inputSize = 30
type InputVector = Tensor Rat [inputSize]


validInput : InputVector -> Bool
validInput x = forall i . 0 <= x ! i  <= 1

--------------------------------------------------------------------------------
-- Outputs

type OutputVector = Vector Rat 2
type Label = Index 2

pos = 0
neg = 1

--------------------------------------------------------------------------------
-- Network

@network
classifier : InputVector -> OutputVector

advises : InputVector -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Dataset

@parameter(infer=True)
n : Nat

@dataset
-- inputs : Tensor Rat [n, inputSize]
inputs : Vector (Vector Rat 30) 5


min : Rat -> Rat -> Rat
min x y = if x <= y then x else y

minList : Vector Rat n -> Rat
minList v = fold min 0 v

max : Rat -> Rat -> Rat
max x y = if y <= x then x else y

maxList : Vector Rat n -> Rat
maxList v = fold max 1 v

inputTranspose : Vector (Vector Rat 5) 30
inputTranspose = foreach i . foreach j . inputs ! j ! i

-- identity : Tensor Rat [2, 3]
-- identity = [ [1, 0, 1],  [0, 1, 0] ]

-- vectorMin : Tensor Rat [inputSize, 5]  -> Index 5 -> Bool
-- vectorMin x i = forall j k. x ! j ! i  <= x ! j ! k

vectorMin :  Index 30 -> Index 5 -> Bool
vectorMin i j  = forall k . inputs ! j ! i  <= inputs ! j ! k

vectorMax :  Index 30  ->  Index 5 -> Bool
vectorMax i j = forall k . inputs ! j ! k  <= inputs ! j ! i

@property
property : Bool
property = forall x. forall j i l k.  validInput x and vectorMin i j and vectorMax l k and x ! i  >= inputs ! j ! i and inputs ! k ! l >=  x ! l =>  advises x pos

{-
vectorMax : Vector Rat 5 -> Index 5 -> Vector Bool 5
vectorMax x i = foreach j . x ! j <= x ! i

vectorOut : Tensor Rat [inputSize, n] -> Vector Rat inputSize
vectorOut x = foreach i . x ! i -}

-- @property
-- property : Bool
-- property = True

{-
vectorMax :  InputVector
vectorMax = foreach i . 1 -- maxList (inputTranspose ! i)

vectorMin : InputVector
vectorMin = foreach i . 0 --minList (inputTranspose ! i)

vectorMax :  InputVector
vectorMax = foreach i . 1 -- maxList (inputTranspose ! i)

hyperRectangle : InputVector -> Bool
hyperRectangle x = forall i . vectorMin ! i  <= x ! i <= vectorMax ! i

@property
property : Bool
property = forall x . hyperRectangle x =>  advises x pos
-}
