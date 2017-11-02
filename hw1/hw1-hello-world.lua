require 'torch'
require 'nn'
-- Loading data
math.randomseed(os.time())
trainCount = 0; testCount = 0
file = io.open('hw1-train-split.txt')
for line in file:lines() do
if (string.len(line) > 0) then
-- Read line from file.
print(line)
--x1, x2, x3, x4, species = unpack(line:split(","))
--input = torch.Tensor({x1, x2, x3, x4});
--output = torch.Tensor(3):zero();
--trainset={}
print("This NN classifier classifies iris data into 3 classes")
print("I am about to code something awesome!")
end
end


