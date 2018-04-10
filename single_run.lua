
require 'torch'
require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'
local path = require 'pl.path'

local train_process = require 'train_process'
local evaluate_process = require 'evaluate_process'

local single_run = {}

function single_run.run_single_experiment(opt)
  print('data set: ', opt.data_name)
  print('class index: ', opt.class_number)
  print('fold_index: ', opt.data_index)
  print('dropout: ', opt.dropout)
  print('seed: ', opt.seed)

  if opt.gpuid < 0 then
    print('Perform calculation by CPU using the optimization method: ' .. opt.opt_method)
  elseif opt.opencl == 0 then 
    print('Perform calculation by GPU with CUDA using the optimization method: ' .. opt.opt_method)
  else
    print('Perform calculation by GPU with OpenCL using the optimization method: ' .. opt.opt_method)
  end
  print('the model type is: ', opt.model)

  torch.manualSeed(opt.seed)

  -- begin to train the model
  print('Begin to train the model...')
  train_process.train(opt)
  print("Training Done!")
  local results = torch.zeros(8)
  results[3] = opt.dropout
  results[1], results[2], results[3], results[4], results[5], results[6] = evaluate_process.evaluate_from_scratch(opt)
  results[7], results[8] = evaluate_process.zero_shot_learning(opt)
  
  return results
end




return single_run