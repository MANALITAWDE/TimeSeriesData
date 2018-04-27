
require 'torch'
require 'lfs'
local path = require 'pl.path'

local table_operation = require 'util/table_operation'
local single_run = require 'single_run'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('script for cross validation of rnn_TiSSiLe_last_timestep model.')
cmd:text()
cmd:text('Options')

--- cross validation options
cmd:option('-if_evaluate_all_class', true, 'experiments on all the class numbers' )
cmd:option('-if_choose_dropout', true, 'if perform cross validation to choose the dropout value, normally 0 is the best choice')
--- data
cmd:option('-data_name','power aggregation dataset','data directory.') -- options: 'arabic', 'MCYT'
cmd:option('-data_dir','../rnn_TiSSiLe_data/','data directory.')
cmd:option('-result_dir','result','result directory.')
cmd:option('-class_number', 10, 'the class number taken into account in the data set')
cmd:option('-data_index', 1, 'the data index out of 5 for each case')
cmd:option('-digit_index', 6, 'only for the arabic voice data.')
cmd:option('-window_size', 2, 'the windowing size')
--- model params
cmd:option('-rnn_size', 32, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'rnn', 'lstm,gru or rnn')
cmd:option('-lambda', 0, 'the coefficient value for the regularization term')
--- optimization
cmd:option('-opt_method', 'rmsprop', 'the optimization method with options: 1. "rmsprop"  2. "gd" (exact gradient descent)')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.4,'learning rate decay')
cmd:option('-learning_rate_decay_after',0,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-test_batch_size',1000,'number of sequences to test on in parallel')
cmd:option('-max_epochs',300,'number of full passes through the training data')
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
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

-- for learning rate
opt.learning_rate = 0.001
-- decrease the learning rate proportionally
local factor = 32 / opt.rnn_size
opt.learning_rate = opt.learning_rate * factor

-- for window_size
if opt.data_name == 'power aggregation dataset'then
  opt.window_size = 2
elseif opt.data_name == 'MCYT' or opt.data_name == 'MCYT_with_forgery' then
  opt.window_size = 3
elseif opt.data_name == 'arabic' or opt.data_name == 'arabic_voice_single' or opt.data_name == 'arabic_voice_mixed' then
  opt.window_size = 2
else
  error('there is no such dataset.')    
end

-- obtain the corresponding class_numbers
local class_numbers = {}
if opt.data_name == 'power aggregation dataset' then
  class_numbers = {111, 102, 103, 108}
elseif opt.data_name == 'arabic' then
  class_numbers = {2, 5, 7, 10}
elseif opt.data_name == 'arabic_voice_single' or opt.data_name == 'arabic_voice_mixed' then
  class_numbers = {5, 10, 20, 40, 60, 88}
elseif opt.data_name == 'Sign' then
  class_numbers = {5, 10, 14, 19}
 
else
  error('there is no such dataset!')
end

-- save the results to log file
-- make sure output directory exists
if not path.exists(opt.result_dir) then lfs.mkdir(opt.result_dir) end
if not path.exists(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size)) then lfs.mkdir(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size)) end
if not path.exists(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size, opt.data_name)) then lfs.mkdir(path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size, opt.data_name)) end

local function write_results(temp_file, results)
  print(string.format('train_auc = %3.5f,  train_EER = %3.5f,   validation_auc = %3.5f, validation_EER = %3.5f,  test_auc = %3.5f, test_EER = %3.5f, ' .. 
    'zero_shot_auc = %3.5f, zero_shot_EER = %3.5f \n',  results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8]))
  temp_file:write(string.format('train_auc = %3.5f,  train_EER = %3.5f,   validation_auc = %3.5f, validation_EER = %3.5f,  test_auc = %3.5f, test_EER = %3.5f, ' .. 
    'zero_shot_auc = %3.5f, zero_shot_EER = %3.5f  \n\n',  results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8]))
end

opt.seed = 123 -- return to initial setting 
local initial_opt = table_operation.shallowCopy(opt)

local function evaluate_one_class_number(opt, cn)
  -- to save the results: each row represent one fold which contains 8 elements: 
  -- [train_auc, train_EER, validation_auc, validation_EER, test_auc, test_EER, zero_shot_auc, zero_shot_EER]
  local result_c = torch.zeros(5, 8):fill(1)
  local class_ind
  if opt.if_evaluate_all_class then
    class_ind = class_numbers[cn]
    opt.class_number = class_ind
  else -- only perform experiments for one class_number 
    class_ind = opt.class_number
    local function table_find()
      local rtn_cn
      for i = 1, 134 do
        if class_numbers[i] == class_ind then
          rtn_cn = i
          break
        end
      end
      return rtn_cn
    end
    cn = table_find()
  end
  local current_result_dir = path.join(opt.result_dir, 'rnn_size_' .. opt.rnn_size, 
    opt.data_name, 'class_' .. tostring(opt.class_number))
  if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
  opt.current_result_dir = current_result_dir  
  local temp_file = io.open(string.format('%s/results_%s_GPU_%d.txt',
    opt.current_result_dir, opt.opt_method, opt.gpuid), "w")
  print(string.format('evaluate the class number %d \n', class_ind))
  temp_file:write(string.format('evaluate the class number %d \n', class_ind))
  if opt.if_choose_dropout then
    -- perform cross validation to choose dropout value
    temp_file:write('perform cross validation for dropout with data_1... \n')
    local dropout = {0.0, 0.2, 0.4}
    local best_validation_auc = 0
    local best_dropout = 0
    local best_result = nil
    for dr = 1, #dropout do
      opt.learning_rate = initial_opt.learning_rate
      opt.dropout = dropout[dr]
      print(string.format('evaluation with dropout %1.3f', opt.dropout))
      opt.data_index = 1
      local results = single_run.run_single_experiment(opt)
      if dr == 1 or results[3] > best_validation_auc then
        best_validation_auc = results[3]
        best_dropout = dropout[dr]
        best_result = results
      end
      temp_file:write(string.format('results with dropout %1.3f:  \n', opt.dropout))
      print(string.format('results with dropout %1.3f: ', opt.dropout))
      write_results(temp_file, results)
    end
    opt.dropout = best_dropout
    temp_file:write(string.format('best dropout = %1.3f  \n\n', opt.dropout))
    print(string.format('best dropout = %1.3f \n', opt.dropout))
    -- assign the best result to result_c[1]
    result_c[1] = best_result
  end
  temp_file:write(string.format('performing cross validation for different folds or initial values(seed) with dropout %1.3f \n', opt.dropout))
  print(string.format('performing cross validation for different folds or initial values(seed) \n'))
  -- evaluate each of 5 folds for each class number except the last two 
  if cn <= #class_numbers -2 then 
    for fold = 1, 5 do
      opt.data_index = fold
      opt.learning_rate = initial_opt.learning_rate
      if fold ~= 1 or not opt.if_choose_dropout then
        result_c[fold] = single_run.run_single_experiment(opt)
      end
      temp_file:write(string.format('results with fold(data_index)  %d:  \n', opt.data_index))
      print(string.format('results with fold(data_index)  %d:  \n', opt.data_index))
      write_results(temp_file, result_c[fold])
    end 
  else -- perform cross validation with different initial values
    for seed_ind = 1, 5 do
      opt.seed = (seed_ind-1)*10 + 123
      opt.learning_rate = initial_opt.learning_rate
      if seed_ind ~= 1 or not opt.if_choose_dropout then
        result_c[seed_ind] = single_run.run_single_experiment(opt)
      end
      temp_file:write(string.format('results with seed  %d:  \n', opt.seed))
      print(string.format('results with seed  %d:  \n', opt.seed))
      write_results(temp_file, result_c[seed_ind])
    end
  end
  
  -- calculate the std of the results
  local mean_result = result_c:mean(1)[1]
  local std_result = result_c:std(1)[1]
  temp_file:write(string.format('\nmean value of results with class_index %d:  \n', opt.class_number))
  print(string.format('mean value of results with class_index %d:  \n', opt.class_number))
  write_results(temp_file, mean_result)
  temp_file:write(string.format('\nstandard deviation of results with class_index %d:  \n', opt.class_number))
  print(string.format('standard deviation of results with class_index %d:  \n', opt.class_number))
  write_results(temp_file, std_result)
  temp_file.close()  
end

if opt.if_evaluate_all_class then
  for cn=1, 153 do
    opt = table_operation.shallowCopy(initial_opt)
    evaluate_one_class_number(opt, cn)
  end
else
  local cn = 0
  print('only test the class number: ', opt.class_number)
  evaluate_one_class_number(opt, cn)
end






