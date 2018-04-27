
local path = require 'pl.path'
local AUC_EER = require 'util/my_AUC_EER_calculation'
require 'util.misc'
local data_loader = require 'util.data_loader'
local model_utils = require 'util.model_utils'


local evaluate_process = {}
                                                
--- preprocessing helper function                 
local function prepro(opt, x)
  x = x:transpose(1,2):contiguous()
  if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    x = x:float():cuda()
  end
  return x
end

--- inference one sample
local function inference(model, x1, x2, true_y)

  -- decode the model and parameters
  local rnn_init_state = model.rnn_init_state
  local clones_rnn1 = model.clones_rnn1
  local clones_rnn2 = model.clones_rnn2
  local top_net = model.top_net
  local criterion = model.criterion
  local Top_Net = model.Top_Net

  local x1_length = x1:size(1)
  local x2_length = x2:size(1)
  -- set the MulConst
  Top_Net:setMulConst(1/x1_length, 1/x2_length)

  ---------------------- forward pass of the whole model -------------------  

  -- perform the forward pass for rnn1
  local init_state_global1 = clone_list(rnn_init_state)
  local rnn1_state = {[0] = init_state_global1}
  local hidden_z_value1 = {}  -- the value of the rnn1 hidden unit 
  -- we don't set the opt.seq_length, instead, we use the current length of the time series
  for t=1,x1_length do
    clones_rnn1[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    local lst = clones_rnn1[t]:forward{x1:narrow(1, t, 1), unpack(rnn1_state[t-1])}
    rnn1_state[t] = {}
    for i=1,#rnn_init_state do table.insert(rnn1_state[t], lst[i]) end -- extract the state, without output
    -- last element is the output of the current time step: the hidden value after dropout
    -- note that we average the value at this moment
    hidden_z_value1[t] = lst[#lst]
  end

  -- perform the forward pass for rnn2
  local init_state_global2 = clone_list(rnn_init_state)
  local rnn2_state = {[0] = init_state_global2}
  local hidden_z_value2 = {}  -- the value of the rnn2 hidden unit 
  -- we don't set the opt.seq_length, instead, we use the current length of the time series
  for t=1,x2_length do
    clones_rnn2[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    local lst = clones_rnn2[t]:forward{x2:narrow(1, t, 1), unpack(rnn2_state[t-1])}
    rnn2_state[t] = {}
    for i=1,#rnn_init_state do table.insert(rnn2_state[t], lst[i]) end -- extract the state, without output
    -- last element is the output of the current time step: the hidden value after dropout
    -- note that we average the value at this moment
    hidden_z_value2[t] = lst[#lst]
  end

  -- perform the forward for the top-net module
  local net_output = top_net:forward({hidden_z_value1, hidden_z_value2})
  --compute the loss
  local current_loss = criterion:forward(net_output, true_y)

  local pred_prob = net_output[1]
  local pred_label = nil
  if net_output[1] >= 0.5 then
    pred_label = 1
  else
    pred_label = 0
  end

  return current_loss, pred_prob, pred_label

end

--- input @data_set is a data sequence (table of Tensor) to be evaluated
local function evaluation_set_performance(opt, model, data_sequence1, data_sequence2,  
  true_label_sequence, if_average, if_plot)
  local total_loss_avg = 0
  local accuracy = 0
  local samples_number = #data_sequence1
  local downsample = 60000
  local if_downsample = false
  local randperm_index = nil
  if samples_number > downsample then -- downsampling
    print('too many sample points, downsampling to '.. downsample .. ' points')
   randperm_index = torch.randperm(samples_number)
   randperm_index = randperm_index:sub(1,downsample)
   samples_number = downsample
   true_label_sequence = true_label_sequence:index(1, randperm_index:long())
   if_downsample = true
  end
  local probs = torch.zeros(samples_number)

  for i = 1, samples_number do
    local x1, x2, true_y
    if if_downsample then
      x1 = data_sequence1[randperm_index[i]]
      x2 = data_sequence2[randperm_index[i]]
    else
      x1 = data_sequence1[i]
      x2 = data_sequence2[i]
    end
    true_y = true_label_sequence[i]
    x1 = prepro(opt, x1)
    x2 = prepro(opt, x2)
    
    if opt.gpuid >= 0 and opt.opencl == 0 then
      true_y = true_y:float():cuda()
    end
    local temp_loss, prob, predict_label = inference(model, x1, x2, true_y)
    total_loss_avg = temp_loss + total_loss_avg
    if predict_label == true_y[1] then
      accuracy = accuracy + 1
    end
    probs[i] = prob
    if i % 1000 == 0 then
      print(i, 'finished!')
    end
  end
  accuracy = accuracy / samples_number * 100

  --- add the regularization term (norm-2), the bias parameters should be excluded
  local function calculate_regularization()
    --[[ 
        the structure of the elements in the rnn_params_flat: 
        1. the main parameters in NN-1 (W): H * D elements
        2. the bias parameters in NN-1: H * 1
        3. the main parameters in NN-2 between adjacent time steps (A): H * H elements
        4. the bias parameters in NN-2: H * 1
    ]]--
    local rnn_params_flat = model.rnn_params_flat
    local top_net_params_flat = model.top_net_params_flat
    local feature_dim = data_sequence1[1]:size(1)
    local W_parameters = rnn_params_flat:sub(1, opt.rnn_size * feature_dim )
    local H_parameters = rnn_params_flat:sub(opt.rnn_size * feature_dim + 
      opt.rnn_size +1, rnn_params_flat:nElement()-opt.rnn_size)
    local top_net_parameters = top_net_params_flat:sub(1, top_net_params_flat:nElement()-1)
    -- calculate the regularization loss value
    local regularization_value = torch.sum(torch.pow(W_parameters, 2))
    regularization_value = regularization_value + torch.sum(torch.pow(H_parameters, 2))
    regularization_value = regularization_value + torch.sum(torch.pow(top_net_parameters, 2))
    regularization_value = regularization_value  * opt.lambda
    return regularization_value   
  end
  local regularization_value = calculate_regularization() 
  total_loss_avg = total_loss_avg / samples_number + regularization_value

  -- calculate AUC and EER
  local pos_class = 1
  local EER, auc_v = AUC_EER.calculate_EER_AUC(true_label_sequence,probs,pos_class, if_plot)

  -- average the scores based on the reference_number references (training samples)
  -- only works for the test set
  if if_average then
    local threshold = 0.5
    local c = 0
    local tot = 0
    local final_l = 0
    local client_scores = {}
    local impostor_scores = {}
    local new_true_label_sequence = {}
    local reference_number = 20
    if opt.data_name == 'MCYT' or opt.data_name == 'MCYT_with_forgery' then
      reference_number = 18
    elseif opt.data_name == 'arabic_voice' then
      reference_number = 7
    elseif opt.data_name == 'Sign' then
      reference_number = 25
    end
    for i = 1, samples_number, reference_number do
      local mean_score = torch.mean(probs:sub(i, i+reference_number-1))
      if mean_score >= threshold then
        final_l = 1
      else
        final_l = 0
      end
      local true_l = true_label_sequence[i][1]
      new_true_label_sequence[#new_true_label_sequence+1] = true_l
      if true_l == 1 then
        client_scores[#client_scores+1] = mean_score
      else
        impostor_scores[#impostor_scores+1] = mean_score
      end
      if true_l == final_l then
        c = c + 1
      end
      tot = tot + 1
    end
    local new_accuracy = c / tot
    print('results before averaging:')
    print(string.format('accuracy: %1.5f. AUC: %1.5f.  EER: %1.5f', accuracy, auc_v, EER))
    print('client: ',#client_scores)
    print('impostor: ',#impostor_scores)
    local new_scores = torch.cat(torch.Tensor(client_scores), torch.Tensor(impostor_scores))
    local new_EER, new_auc_v = AUC_EER.calculate_EER_AUC(
      torch.Tensor(new_true_label_sequence),new_scores,pos_class)
    print('results after averaging:')
    print(string.format('accuracy: %1.5f. AUC: %1.5f.  EER: %1.5f', new_accuracy, new_auc_v, new_EER))
  end


  return total_loss_avg, accuracy, auc_v, probs, EER
end

--- evaluate the training set
function evaluate_process.evaluate_training_set(opt, loader, model, if_plot)
  print('start to evaluate the whole training set...')
  local timer = torch.Timer()
  local time_s = timer:time().real
  if not if_plot then
    if_plot = false
  end
  local total_loss_avg, accuracy, auc, _, EER = evaluation_set_performance(opt, model,
    loader.train_X_sequence_1,loader.train_X_sequence_2,loader.train_T, false, if_plot) 
  local time_e = timer:time().real
  print('total average loss of train set:', total_loss_avg)
  print('accuracy of train set:', accuracy)
  print('EER of train set: ', EER)
  print('auc of train set: ', auc)
  print('elapsed time for evaluating the training set:', time_e - time_s)
  return total_loss_avg, auc, EER
end

--- evaluate the validation set
function evaluate_process.evaluate_validation_set(opt, loader, model, if_plot)
  print('start to evaluate the whole validation set...')
  local timer = torch.Timer()
  local time_s = timer:time().real
  if not if_plot then
    if_plot = false
  end
  local total_loss_avg, accuracy, auc, _, EER = evaluation_set_performance(opt, model,
    loader.validation_X_sequence_1,loader.validation_X_sequence_2,loader.validation_T, false, if_plot) 
  local time_e = timer:time().real
  print('total average loss of validation set:', total_loss_avg)
  print('accuracy of validation set:', accuracy)
  print('EER of validation set: ', EER)
  print('auc of validation set: ', auc)
  print('elapsed time for evaluating the validation set:', time_e - time_s)
  return total_loss_avg, auc, EER
end

--- evaluate the test set
function evaluate_process.evaluate_test_set(opt, loader, model, if_plot)
  print('start to evaluate the whole test set...')
  local timer = torch.Timer()
  local time_s = timer:time().real
  if not if_plot then
    if_plot = false
  end
  local total_loss_avg, accuracy, auc, _, EER = evaluation_set_performance(opt, model, 
    loader.test_X_sequence_1,loader.test_X_sequence_2,loader.test_T, false, if_plot) 
  local time_e = timer:time().real
  print('total average loss of test set:', total_loss_avg)
  print('accuracy of test set:', accuracy)
  print('EER of test set: ', EER)
  print('auc of test set: ', auc)
  print('elapsed time for evaluating the test set:', time_e - time_s)
  return total_loss_avg, auc, EER
end

--- load the data and the trained model from the check point and evaluate the model
function evaluate_process.evaluate_from_scratch(opt)

  --------- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully ------
  ------------------------------------------------------------------------------------------------
  if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(opt.seed)
    else
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end

  ------------------- create the data loader class ----------
  local full_data_dir = path.join(opt.data_dir, opt.data_name)
  local loader = data_loader.create(full_data_dir, opt)
  local feature_dim = loader.feature_dim
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  local model = {}
  local savefile = string.format('%s/%s_best_trained_model_GPU_%d_lambda_%1.6f_dropout_%1.2f_index_%d_seed_%d.t7', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.lambda, opt.dropout, opt.data_index, opt.seed)
  if path.exists(savefile) then
    print('loading the saved model...')
    local checkpoint = torch.load(savefile)
    model = checkpoint.model
  else
    error('error: there is no trained model saved before in such experimental setup.')
  end
  local if_plot = false
  print('evaluate the model from scratch...')
  local train_loss, train_auc, train_EER = evaluate_process.evaluate_training_set(opt, loader, model, if_plot)
  local validation_loss, validation_auc, validation_EER = evaluate_process.evaluate_validation_set(opt, loader, model, if_plot)
  local test_loss, test_auc, test_EER = evaluate_process.evaluate_test_set(opt, loader, model, if_plot)

  local temp_file = io.open(string.format('%s/%s_results_GPU_%d_lambda_%1.6f_dropout_%1.2f_index_%d_seed_%d.txt', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.lambda, opt.dropout, opt.data_index, opt.seed), "a")
  temp_file:write(string.format('similarity measurement results \n'))
  temp_file:write(string.format('train set loss = %6.8f, train auc= %6.8f, train EER = %6.3f\n', 
    train_loss, train_auc, train_EER ))
  temp_file:write(string.format('validation set loss = %6.8f, validation auc= %6.8f, validation EER = %6.3f\n', 
    validation_loss, validation_auc, validation_EER ))
  temp_file:write(string.format('test set loss = %6.8f, test auc= %6.8f, test EER = %6.3f\n', 
    test_loss, test_auc, test_EER ))
    
  return train_auc, train_EER, validation_auc, validation_EER, test_auc, test_EER
end

--- for zero-shot learning
function evaluate_process.zero_shot_learning(opt)
  --------- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully ------
  ------------------------------------------------------------------------------------------------
  if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(opt.seed)
    else
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end

  ------------------- create the data loader class ----------
  local full_data_dir = path.join(opt.data_dir, opt.data_name)
  local loader = data_loader.create(full_data_dir, opt, true)
  local feature_dim = loader.feature_dim
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  local model = {}
  local savefile = string.format('%s/%s_best_trained_model_GPU_%d_lambda_%1.6f_dropout_%1.2f_index_%d_seed_%d.t7', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.lambda, opt.dropout, opt.data_index, opt.seed)
  if path.exists(savefile) then
    print('loading the saved model...')
    local checkpoint = torch.load(savefile)
    model = checkpoint.model
    if loader.max_time_series_length > #model.clones_rnn1 then
      print('not enough length of rnn1, cloning more...')
      model.clones_rnn1 = model_utils.clone_many_times(model.rnn_model, loader.max_time_series_length, not model.rnn_model.parameters)
    end
    if loader.max_time_series_length > #model.clones_rnn2 then
      print('not enough length of rnn2, cloning more...')
      model.clones_rnn2 = model_utils.clone_many_times(model.rnn_model, loader.max_time_series_length, not model.rnn_model.parameters)
    end
    
  else
    error('error: there is no trained model saved before in such experimental setup.')
  end
  local if_plot = false
  print('evaluate the zero-shot learning ability of the model from scratch...')
  local zero_shot_loss, zero_shot_auc, zero_shot_EER = evaluate_process.evaluate_test_set(opt, loader, model, if_plot)
  local temp_file = io.open(string.format('%s/%s_results_GPU_%d_lambda_%1.6f_dropout_%1.2f_index_%d_seed_%d.txt', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.lambda, opt.dropout, opt.data_index, opt.seed), "a")
  temp_file:write(string.format('zero-shot results \n'))
  temp_file:write(string.format('zero_shot set loss = %6.8f, zero_shot auc= %6.8f, zero_shot EER = %6.3f\n', 
    zero_shot_loss, zero_shot_auc, zero_shot_EER ))
  return zero_shot_auc, zero_shot_EER
end

--- for the gradient check
function evaluate_process.grad_check(model, x1, x2, true_y)
  -- decode the model and parameters
  local rnn_params_flat = model.rnn_params_flat
  local rnn_grad_params_flat = model.rnn_grad_params_flat
  local top_net_params_flat = model.top_net_params_flat
  local top_net_grad_flat = model.top_net_grad_params_flat

  local function calculate_loss()
    local current_loss = inference(model, x1, x2, true_y)
    return current_loss
  end  

  local if_print = true
  local threshold = 1e-4
  for i = 4, 5 do
    local delta = 1 / torch.pow(1e1, i)
    print('delta:', delta)
    local loss_minus_delta = torch.zeros(rnn_params_flat:nElement() + top_net_params_flat:nElement())
    local loss_add_delta = torch.zeros(rnn_params_flat:nElement() + top_net_params_flat:nElement())
    local grad_def = torch.zeros(rnn_params_flat:nElement() + top_net_params_flat:nElement())
    -- check rnn params
    local rnn_params_flat_backup = rnn_params_flat:clone()
    for i = 1, rnn_params_flat:nElement() do
      rnn_params_flat[i] = rnn_params_flat[i] - delta
      loss_minus_delta[i] = calculate_loss() 
      rnn_params_flat[i] = rnn_params_flat[i] + 2*delta
      loss_add_delta[i] = calculate_loss()
      grad_def[i] = (loss_add_delta[i] - loss_minus_delta[i]) / (2*delta)
      rnn_params_flat[i] = rnn_params_flat[i] - delta -- retore the parameters
    end
    rnn_params_flat:copy(rnn_params_flat_backup) -- retore the parameters

    -- check top_net params
    local top_net_params_flat_backup = top_net_params_flat:clone()
    for i = 1+rnn_params_flat:nElement(), rnn_params_flat:nElement() + top_net_params_flat:nElement() do
      top_net_params_flat[i-rnn_params_flat:nElement()] = top_net_params_flat[i-rnn_params_flat:nElement()] - delta
      loss_minus_delta[i] = calculate_loss() 
      top_net_params_flat[i-rnn_params_flat:nElement()] = top_net_params_flat[i-rnn_params_flat:nElement()] + 2*delta
      loss_add_delta[i] = calculate_loss()
      grad_def[i] = (loss_add_delta[i] - loss_minus_delta[i]) / (2*delta)
      top_net_params_flat[i-rnn_params_flat:nElement()] = top_net_params_flat[i-rnn_params_flat:nElement()] - delta
    end
    top_net_params_flat:copy(top_net_params_flat_backup) -- retore the parameters
    -- comparison
    local all_grad_params = torch.cat(rnn_grad_params_flat, top_net_grad_flat)
    local relative_diff = torch.zeros(grad_def:nElement())
    relative_diff = torch.abs(grad_def - all_grad_params)
    relative_diff:cdiv((torch.abs(grad_def) + torch.abs(all_grad_params))/2)

    local inaccuracy_num = 0
    local reversed_direction = 0
    for i = 1, grad_def:nElement() do
      if relative_diff[i] > threshold then
        if torch.abs(grad_def[i]) + torch.abs(all_grad_params[i]) > 0 then
          if grad_def[i] * all_grad_params[i] == 0 then
            --            if torch.abs(grad_def[i] + all_grad_params[i])
            print(string.format('index: %4d, relative_diff: %6.5f,  gradient_def: %6.25f,  all_grad_params: %6.25f',
              i, relative_diff[i], grad_def[i], all_grad_params[i]))
          elseif if_print then
            print(string.format('index: %4d, relative_diff: %6.5f,  gradient_def: %6.25f,  all_grad_params: %6.25f',
              i, relative_diff[i], grad_def[i], all_grad_params[i]))
            --          print(string.format('index: %4d, loss_original: %6.15f,  loss_current_sample: %6.15f,  loss_delta: %6.15f',
            --            i, loss_original[i], loss_current_sample, loss_delta[i]))
          end
          inaccuracy_num = inaccuracy_num + 1
        end   
      end
    end
    for i = 1, grad_def:nElement() do
      if grad_def[i] * all_grad_params[i] < 0 then
        if if_print then
          print(string.format('index: %4d, relative_diff: %6.5f,  gradient_def: %6.10f,  all_grad_params: %6.10f',
            i, relative_diff[i], grad_def[i], all_grad_params[i]))
        end
        reversed_direction = reversed_direction + 1
      end
    end
    print('biggest relative_diff:', torch.max(relative_diff))
    local sorted_relative_diff = torch.sort(relative_diff)
    local second_largest = sorted_relative_diff[torch.sum(sorted_relative_diff:lt(2))]
    print('second biggest relative_diff:', second_largest)
    print('there are', inaccuracy_num, 'inaccuracy gradients.')
    print('there are', reversed_direction, 'reversed directions.')
  end

end

return evaluate_process

