--[[

This is the implementation of the siamese recurrent neural networks (SRNs) for the time series similarity learning. 
Acknowledgement: The code in based on the code 'https://github.com/karpathy/char-rnn', which is torch implements of lstm and rnn by Karpathy.

Copyright (c) 2015 Wenjie Pei
Delft University of Technology 

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'
local path = require 'pl.path'

local train_process = require 'train_process'
local evaluate_process = require 'evaluate_process'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a rnn-based TiSSiLe model')
cmd:text()
cmd:text('Options')
--- data
cmd:option('-data_name','arabic_voice_mixed','data directory.') -- options: 'arabic', 'MCYT'
cmd:option('-data_dir','../../rnn_TiSSiLe_data/','data directory.')
cmd:option('-result_dir','result','result directory.')
cmd:option('-class_number', 88, 'the class number taken into account in the data set')
cmd:option('-data_index', 1, 'the data index out of 5 for each case')
cmd:option('-digit_index', 1, 'only for the arabic voice data.')
cmd:option('-window_size', 2, 'the windowing size')
--- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'rnn') -- can be extended to lstm in the future
cmd:option('-lambda', 0, 'the coefficient value for the regularization term')
--- optimization
cmd:option('-opt_method', 'rmsprop', 'the optimization method with options: 1. "rmsprop"  2. "gd" (exact gradient descent)')
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-learning_rate_decay',0.4,'learning rate decay')
cmd:option('-learning_rate_decay_after',0,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-max_iterations',50000,'max iterations to run')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-blowup_threshold',1e1,'the blowup threshold')
cmd:option('-check_gradient', false, 'whether to check the gradient value') 
-- for now we just perform on the training set, the more standard way should be on the validation set
cmd:option('-stop_iteration_threshold', 16,
       'if better than the later @ iterations , then stop the optimization')
cmd:option('-decay_threshold', 4, 'if better than the later @ iterations , then decay the learning rate')
cmd:option('-if_init_from_check_point', false, 'initialize network parameters from checkpoint at this path')
cmd:option('-if_direct_test_from_scratch', false)
cmd:option('-if_direct_zero_shot_learning', false)
cmd:option('-validation_set_ratio', 0.15)
--- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-evaluate_every',10000,'how many samples between evaluate the whole data set')
cmd:option('-checkpoint_dir', 'result', 'output directory where checkpoints get written')

--- GPU/CPU
-- currently, only supports CUDA
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
local opt = cmd:parse(arg)

print('data set: ', opt.data_name)

-- for learning rate
if opt.data_name == 'Sign'then
  opt.learning_rate = 0.001
else
  opt.learning_rate = 0.002
end

opt.learning_rate = opt.learning_rate / math.floor(opt.rnn_size/32)
print('learning rate: ', opt.learning_rate)

if opt.gpuid < 0 then
  print('Perform calculation by CPU using the optimization method: ' .. opt.opt_method)
else
  print('Perform calculation by GPU with OpenCL using the optimization method: ' .. opt.opt_method)
end
print('the model type is: ', opt.model)

torch.manualSeed(opt.seed)

-- make sure output directory exists
if not path.exists(opt.result_dir) then lfs.mkdir(opt.result_dir) end
if not path.exists(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size)) then lfs.mkdir(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size)) end
if not path.exists(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size, opt.data_name)) then lfs.mkdir(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size, opt.data_name)) end
local current_result_dir = path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size, 
  opt.data_name, 'class_' .. tostring(opt.class_number))
if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
opt.current_result_dir = current_result_dir  


if opt.if_direct_test_from_scratch or opt.if_direct_zero_shot_learning then
  if opt.if_direct_test_from_scratch then
    evaluate_process.evaluate_from_scratch(opt)
  end
  if opt.if_direct_zero_shot_learning then
    evaluate_process.zero_shot_learning(opt)
  end
else
  -- begin to train the model
  print('Begin to train the model...')
  train_process.train(opt)
  print("Training Done!")
  evaluate_process.evaluate_from_scratch(opt)
  evaluate_process.zero_shot_learning(opt)
end
