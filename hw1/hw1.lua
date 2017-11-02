require 'torch'
require 'nn'
trainSet = {}
testSet = {}

function trainSet:size()
    return trainCount
end

-- Function to find the index of maximum element
function argmax(v)
  local maxvalue = torch.max(v)
  for i=1,v:size(1) do
    if v[i] == maxvalue then
      return i
    end
  end
end

print("This NN classifies iris data into 3 different classes.")
print("These are the user-specified options and hyper-parameters:")
print("Training File: hw1-train-split.txt")
print("Test File: hw1-test-split.txt")


trainCount = 0; testCount = 0
-- Loading training data
file = io.open('hw1-train-split.txt')
for line in file:lines() do
    if (string.len(line) > 0) then
        x1, x2, x3, x4, species = line:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
        input = torch.Tensor({x1, x2, x3, x4});
        output = torch.Tensor(1);
   		output[1]=species;
   		table.insert(trainSet, {input, output})
        trainCount = trainCount + 1

    end
end

-- Loading testing data
file = io.open('hw1-test-split.txt')
for line in file:lines() do
    if (string.len(line) > 0) then
        x1, x2, x3, x4, species = line:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
        input = torch.Tensor({x1, x2, x3, x4});
        output = torch.Tensor(1)
   		output[1]=species;
   		table.insert(testSet, {input, output})
        testCount = testCount + 1

    end
end



-- Initialise the network.

inputs = 4; outputs = 3; hidden = 10;
print("Training Data Size: ".. trainCount)
print("Testing Data Size: ".. testCount)
print("No. of Inputs: 4")
print("No. of Outputs: 3")
print("No. of Hidden Layers: 3")
print("No. of Hidden Nodes: " .. hidden)

-- Define the network

mlp = nn.Sequential();
mlp:add(nn.Linear(inputs, hidden))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hidden, outputs))
mlp:add(nn.SoftMax())

print(" ")
print("The network looks like this: ")
print(mlp)
print(" ")

-- Train the network.
print("Training the network")
criterion = nn.ModuleCriterion(nn.ClassNLLCriterion(), nn.Log())
criterion.sizeAverage=false
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 25
print("Learning Rate: "..trainer.learningRate)
print("Maximum no. of iterations: "..trainer.maxIteration)
print(" ")
trainer:train(trainSet)


print("Testing the network")
-- Test the network.
tot = 0
pos = 0
for i = 1, testCount do
  val = mlp:forward(testSet[i][1])
  out = testSet[i][2]
     
  local prediction = argmax(val)
  
  if prediction == out[1] then
    pos = pos + 1
  end
  tot = tot + 1
  print("Predicted: "..prediction.." Desired: "..out[1].." Correct: "..pos.."/"..tot)
end
print(tot)
print("Accuracy(%) is " .. pos/tot*100)



-- Saving torch model
torch.save('hw1.torchModel',mlp)

