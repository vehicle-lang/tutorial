type Input = Tensor Real [2]

@network
f : Input -> Tensor Real [1]

@parameter
epsilon : Real

valid : Input -> Bool
valid x = forall i . 0 <= x ! i <= 1

@property
p : Bool
p = forall x . valid x and reduceAdd x <= epsilon => f x ! 0 >= 0
