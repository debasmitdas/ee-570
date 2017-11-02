require 'torch'
require 'nn'

function argmax(v)
  local maxvalue = torch.max(v)
  for i=1,v:size(1) do
    if v[i] == maxvalue then
      return i
    end
  end
end

testSet={}
testCount=0
-- Loading testing data
file = io.open('hw1-test-split.txt')
for line in file:lines() do
    if (string.len(line) > 0) then
        print(line)
        -- Read line from file.
        x1, x2, x3, x4, species = line:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
        print(x3)
        input = torch.Tensor({x1, x2, x3, x4});
        output = torch.Tensor(1)
   		output[1]=species;
   		table.insert(testSet, {input, output})
        testCount = testCount + 1

    end
end

-- Loading torch model
model = torch.load('hw1.torchmodel')

-- Test the network.

tot = 0
pos = 0
for i = 1, testCount do
  val = model:forward(testSet[i][1])
  out = testSet[i][2]
  print(val)
   
  local prediction = argmax(val)
  --print(prediction)
  --print(out[1])
  --print(",")
  if prediction == out[1] then
    pos = pos + 1
  end
  tot = tot + 1
end
print(tot)
print("Accuracy(%) is " .. pos/tot*100)

