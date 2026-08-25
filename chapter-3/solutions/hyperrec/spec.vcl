--------------------------------------------------------------------------------
-- Inputs

inputSize = 30

type Input = Tensor Real [inputSize]

--------------------------------------------------------------------------------
-- Outputs

type Output = Tensor Real [2]

type Label = Index 2

pos : Label
pos = 0

neg : Label
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
inputs : Tensor Real [n, inputSize]

inputTranspose : Tensor Real [inputSize, n]
inputTranspose = foreach i . foreach j . inputs ! j ! i

vectorMin : Input
vectorMin = foreach i . reduceMin (inputTranspose ! i)

vectorMax :  Input
vectorMax = foreach i . reduceMax (inputTranspose ! i)

hyperRectangle : Input -> Bool
hyperRectangle x = forall i . vectorMin ! i  <= x ! i <= vectorMax ! i

@property
property : Bool
property = forall x . hyperRectangle x =>  advises x pos
