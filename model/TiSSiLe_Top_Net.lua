
--[[
Taken the hidden-unit values of the hidden units from RNN modules, This Top_Net performs the following operations:
sum of the hidden-unit values
average of the sum
element-wise multiplication between mean value of two rnn modules  
linear transformation and 
employ the sigmoid classifier as the output layer.  

]]--

require 'nn'
require 'model/MulConst'

local Top_Net = {}

function Top_Net.net(rnn_size)
  local net1 = nn.Sequential()
  net1:add(nn.JoinTable(1))
  net1:add(nn.Sum(1))
  local mulconst1 = nn.MulConst(1)
  Top_Net.mulconst1 = mulconst1 
  net1:add(mulconst1)
  local net2 = nn.Sequential()
  net2:add(nn.JoinTable(1))
  net2:add(nn.Sum(1))
  local mulconst2 = nn.MulConst(1) 
  net2:add(mulconst2)
  Top_Net.mulconst2 = mulconst2
  
  -- define a ParalellTable to combine two rnns as the input of the next the CMulTable
  local p = nn.ParallelTable()
  p:add(net1)
  p:add(net2)
  
  -- perfrom the element-wise multiplication
  local net = nn.Sequential()
  net:add(p)
  net:add(nn.CMulTable())
  net:add(nn.Linear(rnn_size, 1))
  net:add(nn.Sigmoid())
  
  return net
end

function Top_Net:setMulConst(const1, const2)
  self.mulconst1:setConst(const1)
  self.mulconst2:setConst(const2)
end

return Top_Net