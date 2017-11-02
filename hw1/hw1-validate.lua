-------------------------------------------------------------------------------
--	Filename: hw1-validate.lua
--	Author: Jared Johansen
-------------------------------------------------------------------------------
require 'torch'
require 'nn'

-------------------------------------------------------------------------------
-- 	Grader-script, hard-coded values (DO NOT CHANGE THESE)
-------------------------------------------------------------------------------
testFileName = 'hw1.torchModel'

-------------------------------------------------------------------------------
-- 	Print instructions
-------------------------------------------------------------------------------
print("\nThis script will check several if your NN meets basic submission criteria.\n")

-------------------------------------------------------------------------------
--	Check if the file is named correctly
-------------------------------------------------------------------------------
local f=io.open(testFileName,"r")
if f~=nil then 
	print("\tFile named correctly:       SUCCESS")
	io.close(f)
	fileCorrect = 1
else
	print("\tFile named correctly:       FAILURE")
	print("\tYou must have a file named 'hw1.torchModel'. <-- notice the capitalization")
	os.exit()
end

-------------------------------------------------------------------------------
--	Check if the network loads
-------------------------------------------------------------------------------
local status1, err1 = pcall(torch.load, testFileName)
if status1 then
	print("\tAble to load your file:     SUCCESS")
	testModel = torch.load(testFileName)
else
	print("\tAble to load your file:     FAILURE")
	print("\tYou should use this command to save your NN: 'torch.save(fileName, myNN)'.")
	os.exit()
end

-----------------------------------------------------------------------------
--	Check if the input dimensions are correct
-----------------------------------------------------------------------------
local x = torch.DoubleTensor(1,4)
local status2, err2 = pcall(testModel.forward, testModel, x)
if status2 then 
	print("\tInput dimensions correct:   SUCCESS")
	y = testModel:forward(x)
else
	print("\tInput dimensions correct:   FAILURE")
	print("\tThe input to your NN should be a torch.DoubleTensor of size 1x4") 
	os.exit()
end

-----------------------------------------------------------------------------
--	Check if the output dimensions are correct
-----------------------------------------------------------------------------
if y:nDimension() ~= 2 or y:size()[1] ~= 1 or y:size()[2] ~= 3 then
	print("\tOutput dimensions correct:  FAILURE")
	print("\tThe output to your NN should be a torch.DoubleTensor of size 1x3")
	os.exit()
else
	print("\tOutput dimensions correct:  SUCCESS")
end

-----------------------------------------------------------------------------
--	Print submission instructions
-----------------------------------------------------------------------------
print("\nSUBMISSION INSTRUCTIONS:")
print("\tPut your 'hw1.torchModel' and 'hw1.lua' files into a folder called 'hw1'.")
print("\tZIP that folder.  (No tarballs, please.)")
print("\tUpload the ZIPPED folder to Blackboard.")
print("\tDo not upload anything else.\n\n")

