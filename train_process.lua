-- internal library
local optim = require 'optim'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local Top_Net = require 'model/TiSSiLe_Top_Net'
local evaluate_process = require 'evaluate_process'
require 'model/MulConst'
local table_operation = require 'util/table_operation'

local train_process = {}

--- process one batch to get the gradients for optimization and update the parameters 
-- return the loss value of one minibatch of samples
local function feval(opt, loader, model, rmsprop_para)
  -- decode the model and parameters, 
  -- since it is just the reference to the same memory location, hence it is not time-consuming.
  local rnn_init_state = model.rnn_init_state
  local clones_rnn1 = model.clones_rnn1
  local clones_rnn2 = model.clones_rnn2
  local top_net = model.top_net
  local rnn_params_flat = model.rnn_params_flat
  local rnn_grad_params_flat = model.rnn_grad_params_flat
  local top_net_params_flat = model.top_net_params_flat
  local top_net_grad_params_flat = model.top_net_grad_params_flat
  local criterion = model.criterion
  local rnn_grad_all_batches = model.rnn_grad_all_batches
  local top_net_grad_all_batches = model.top_net_grad_all_batches
  
  ---------------------------- get minibatch --------------------------
  ---------------------------------------------------------------------
  
  local data_index = loader:get_next_batch('train', opt.batch_size)
  local loss_total = 0
  rnn_grad_all_batches:zero()
  top_net_grad_all_batches:zero()
  -- Process the batch of samples one by one, since different sample contains different length of time series, 
  -- hence it's not convenient to handle them together
  for batch = 1, opt.batch_size do
    local current_data_index = data_index[batch]
    local x1 = loader.train_X_sequence_1[current_data_index]
    local x2 = loader.train_X_sequence_2[current_data_index]
    local true_y = loader.train_T[current_data_index]
    --- preprocessing helper function
    local function prepro(x)
      x = x:transpose(1,2):contiguous()
--      if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
--        -- have to convert to float because integers can't be cuda()'d
--        x = x:float():cuda()
--      end
      return x
    end
    x1 = prepro(x1)
    x2 = prepro(x2)
--    if opt.gpuid >= 0 and opt.opencl == 0 then
--      true_y = true_y:float():cuda()
--    end
    local x1_length = x1:size(1)
    local x2_length = x2:size(1)
    
    -- set the MulConst
    Top_Net:setMulConst(1/x1_length, 1/x2_length)
        
    ---------------------- forward pass of the whole model -------------------  
    --------------------------------------------------------------------------  
    
    -- perform the forward pass for rnn1
    local init_state_global1 = clone_list(rnn_init_state)
    local rnn1_state = {[0] = init_state_global1}
    local hidden_z_value1 = {}  -- the value of the rnn1 hidden unit 
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=1,x1_length do
      clones_rnn1[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
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
      clones_rnn2[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      local lst = clones_rnn2[t]:forward{x2:narrow(1, t, 1), unpack(rnn2_state[t-1])}
      rnn2_state[t] = {}
      for i=1,#rnn_init_state do table.insert(rnn2_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      -- note that we average the value at this moment
      hidden_z_value2[t] = lst[#lst]
    end
  
--  -- for check
--  print('max value in hidden z:', torch.max(hidden_z_value1[x1_length]), torch.max(hidden_z_value2[x2_length]))
    -- perform the forward for the top-net module
    local net_output = top_net:forward({hidden_z_value1, hidden_z_value2})
    --compute the loss
    local current_loss = criterion:forward(net_output, true_y)
    loss_total = loss_total + current_loss

    ---------------------- backward pass of the whole model ---------------------
    -----------------------------------------------------------------------------
    
    -- peform the backprop on the top_net
    rnn_grad_params_flat:zero()
    top_net_grad_params_flat:zero()
    local grad_net = top_net:backward({hidden_z_value1, hidden_z_value2}, 
      criterion:backward(net_output, true_y))
    top_net_grad_all_batches:add(top_net_grad_params_flat) 
  
    -- perform backward pass for the rnn1
    -- initialize gradient at time T to be zeros (there's no influence from future)
    local drnn1_state = {[x1_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = x1_length,1,-1 do
      local doutput_t = grad_net[1][t]
      table.insert(drnn1_state[t], doutput_t)
      local dlst = clones_rnn1[t]:backward({x1:narrow(1, t, 1), unpack(rnn1_state[t-1])}, drnn1_state[t])
      drnn1_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k > 1 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          drnn1_state[t-1][k-1] = v
        end
      end
    end
    
    -- perform backward pass for the rnn2
    -- initialize gradient at time T to be zeros (there's no influence from future)
    local drnn2_state = {[x2_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = x2_length,1,-1 do
      local doutput_t = grad_net[2][t]
      table.insert(drnn2_state[t], doutput_t)
      local dlst = clones_rnn2[t]:backward({x2:narrow(1, t, 1), unpack(rnn2_state[t-1])}, drnn2_state[t])
      drnn2_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k > 1 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          drnn2_state[t-1][k-1] = v
        end
      end
    end
    
--    rnn_grad_params_flat:clamp(-opt.grad_clip, opt.grad_clip)
    rnn_grad_all_batches:add(rnn_grad_params_flat)
        
    -- for gradient check
    if opt.check_gradient then
      if batch > 7 and batch < 11 then
        print('batch id: ', batch)
        evaluate_process.grad_check(model, x1, x2, true_y, rnn_grad_params_flat)
        print('\n')
      elseif batch == 11 then
        os.exit()
      end
    end
    
  end
  --- add the regularization term (norm-2), the bias parameters should be excluded
  local function calculate_regularization()
    --[[ 
        the structure of the elements in the rnn_params_flat: 
        1. the main parameters in NN-1 (W): H * D elements
        2. the bias parameters in NN-1: H * 1
        3. the main parameters in NN-2 between adjacent time steps (A): H * H elements
        4. the bias parameters in NN-2: H * 1
    ]]--

    local W_parameters = rnn_params_flat:sub(1, opt.rnn_size * loader.feature_dim )
    local H_parameters = rnn_params_flat:sub(opt.rnn_size * loader.feature_dim + 
      opt.rnn_size +1, rnn_params_flat:nElement()-opt.rnn_size)
    local top_net_parameters = top_net_params_flat:sub(1, top_net_params_flat:nElement()-1)
    -- add grad of the regularization term
    rnn_grad_all_batches:sub(1, opt.rnn_size * loader.feature_dim):add(2*opt.lambda, W_parameters)
    rnn_grad_all_batches:sub(W_parameters:nElement()+opt.rnn_size+1,
      rnn_params_flat:nElement()-opt.rnn_size):add(2*opt.lambda, H_parameters)
    top_net_grad_all_batches:sub(1, top_net_parameters:nElement()):add(2*opt.lambda, top_net_parameters)
    -- calculate the regularization loss value
    local regularization_value = torch.sum(torch.pow(W_parameters, 2))
    regularization_value = regularization_value + torch.sum(torch.pow(H_parameters, 2))
    regularization_value = regularization_value + torch.sum(torch.pow(top_net_parameters, 2))
    regularization_value = regularization_value  * opt.lambda
    return regularization_value   
  end
  local regularization_value = 0
  regularization_value = calculate_regularization()
  
  -- udpate all the parameters
  if opt.opt_method == 'rmsprop' then
    -- for rnn_params_flat
    local function feval_rmsprop_rnn(p)
      rnn_grad_all_batches:clamp(-opt.grad_clip, opt.grad_clip)
      return loss_total, rnn_grad_all_batches
    end
    optim.rmsprop(feval_rmsprop_rnn, rnn_params_flat, rmsprop_para.config,
       rmsprop_para.state.rnn_state)
    -- for top_net_params_flat
    local function feval_rmsprop_top_net(p)
--    print('max in top net grad: ', torch.max(torch.abs(top_net_grad_all_batches)))
      top_net_grad_all_batches:clamp(-opt.grad_clip, opt.grad_clip)
      return loss_total, top_net_grad_all_batches
    end    
    optim.rmsprop(feval_rmsprop_top_net, top_net_params_flat, rmsprop_para.config,
      rmsprop_para.state.top_net_state)
  elseif opt.opt_method == 'gd' then -- 'gd' simple direct minibatch gradient descent
    rnn_params_flat:add(-opt.learning_rate, rnn_grad_all_batches)
    top_net_params_flat:add(-opt.learning_rate, top_net_grad_all_batches)
  else
    error("there is no such optimization option!")  
  end

  -- return the mean value of loss
  loss_total = loss_total / opt.batch_size + regularization_value

  return loss_total
end

--- major function 
function train_process.train(opt)

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
  -----------------------------------------------------------
  local full_data_dir = path.join(opt.data_dir, opt.data_name)
  local loader = data_loader.create(full_data_dir, opt)
  local feature_dim = loader.feature_dim
  local do_random_init = true
  
  ------------------ begin to define the whole model --------------------------
  -----------------------------------------------------------------------------
  -- either from the check point or create the model from the scratch
  local model = {}
  local savefile = string.format('%s/%s_best_trained_model_GPU_%d_lambda_%1.6f_dropout_%1.2f_index_%d_seed_%d.t7', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.lambda, opt.dropout, opt.data_index, opt.seed)
  if opt.if_init_from_check_point and path.exists(savefile) then
    print('Init the model from the check point saved before...')
    local checkpoint = torch.load(savefile)
    model = checkpoint.model
    Top_Net = model.Top_Net
    opt.learning_rate = checkpoint.opt.learning_rate
  -- create the model from the scratch
  else
    print('Define the model from the scratch...')
    ------------------ for RNN parts --------------------------
    -- define the rnn model: prototypes for one timestep, then clone them in time
    print('creating two rnn model with ' .. opt.num_layers .. ' layers')
    local rnn_model = RNN.rnn(feature_dim, opt.rnn_size, opt.num_layers, opt.dropout)
    -- ship the model to the GPU if desired
    if opt.gpuid >= 0 and opt.opencl == 0 then rnn_model:cuda()  end
    -- put the above things into one flattened parameters tensor
    -- including the parameters W(H*D), A(H*H) and 2 bias_b(H*1) (one for W and one for A), here H = opt.rnn_size
    local rnn_params_flat, rnn_grad_params_flat = rnn_model:getParameters()
    print('number of parameters in the rnn model: ' .. rnn_params_flat:nElement())
    model.rnn_params_flat = rnn_params_flat
    model.rnn_grad_params_flat = rnn_grad_params_flat
    model.rnn_model = rnn_model
    -- the initial state of the cell/hidden states
    local rnn_init_state = {}
    for L=1,opt.num_layers do
      local h_init = torch.zeros(1, opt.rnn_size)
      if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
      table.insert(rnn_init_state, h_init:clone())
    end
    model.rnn_init_state = rnn_init_state

    -- make a bunch of clones for each of 2 input time series after flattening, sharing the same memory
    -- note that: it is only performed once for the reason of efficiency, 
    -- hence we clone the max length of times series in the data set for each rnn time series 
    print('cloning rnn1')
    local clones_rnn1 = model_utils.clone_many_times(rnn_model, loader.max_time_series_length, not rnn_model.parameters)
    print('cloning rnn2')
    local clones_rnn2 = model_utils.clone_many_times(rnn_model, loader.max_time_series_length, not rnn_model.parameters)
    print('cloning ' .. loader.max_time_series_length ..  ' for each time series finished! ');   
    model.clones_rnn1 = clones_rnn1
    model.clones_rnn2 = clones_rnn2
 
    ------------ define the top net module to connect two rnn modules---------
    print('creating the top net module to connect two rnn modules...')
    local top_net = Top_Net.net(opt.rnn_size) 
    -- ship the model to the GPU if desired
    if opt.gpuid >= 0 and opt.opencl == 0 then top_net:cuda() end
    local top_net_params_flat, top_net_grad_params_flat = top_net:getParameters()
    print('number of parameters in the top_net model: ' .. top_net_params_flat:nElement())
    model.Top_Net = Top_Net
    model.top_net = top_net
    model.top_net_params_flat = top_net_params_flat
    model.top_net_grad_params_flat = top_net_grad_params_flat
    print('number of parameters in the whole model: ' .. rnn_params_flat:nElement() + top_net_params_flat:nElement())
    
    -- define the criterion (loss)
    local criterion = nn.BCECriterion()
    model.criterion = criterion
    -- ship the model to the GPU if desired
    if opt.gpuid >= 0 and opt.opencl == 0 then criterion:cuda() end
--    -- init the parameters  
--    rnn_params_flat:uniform(-0.1, 0.1)
--    top_net_params_flat:uniform(-0.1, 0.1)
         
    -- pre-allocate the memory for the temporary variable used in the training phase
    local rnn_grad_all_batches = torch.zeros(rnn_params_flat:nElement())
    local top_net_grad_all_batches = torch.zeros(top_net_params_flat:nElement())
    if opt.gpuid >= 0 and opt.opencl == 0 then
      rnn_grad_all_batches = rnn_grad_all_batches:float():cuda()
      top_net_grad_all_batches = top_net_grad_all_batches:float():cuda()
    end
    model.rnn_grad_all_batches = rnn_grad_all_batches
    model.top_net_grad_all_batches = top_net_grad_all_batches
  end
  
  --------------- start optimization here -------------------------
  -----------------------------------------------------------------
  -- for rmsprop
  local rmsprop_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
  local rmsprop_state = {rnn_state = {}, top_net_state = {}}
  local rmsprop_para = {config = rmsprop_config, state = rmsprop_state}
  
  local iterations = math.floor(opt.max_epochs * loader.nTrain / opt.batch_size)
  local iterations_per_epoch = math.floor(loader.nTrain / opt.batch_size)
  local train_losses = torch.zeros(iterations)
  local timer = torch.Timer()
  local time_s = timer:time().real
  local epoch = 0
  local better_times_total = 0
  local better_times_decay = 0
  local current_best_auc = 0-1
  --- save the current trained best model
  -- for the continution of training or for the test
  local function save_model()
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {model = model, opt = opt}
    torch.save(savefile, checkpoint)
  end
  for i = 1, iterations do
    
    epoch = i / loader.nTrain * opt.batch_size
    if epoch > opt.max_epochs then break end
    if i>opt.max_iterations then break end
    local time_ss = timer:time().real
    -- optimize one batch of training samples
    train_losses[i] = feval(opt, loader, model, rmsprop_para)
    local time_ee = timer:time().real
    local time_current_iteration = time_ee - time_ss
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    -- check if the loss value blows up
    local function is_blowup(loss_v)
      local loss0 = -torch.log(0.5) -- random guess
      if loss_v > loss0 * opt.blowup_threshold then
        print('loss is exploding, aborting:', loss_v)
        return true
      else 
        return false
      end
    end
    if is_blowup(train_losses[i]) then
      local decay_factor = opt.learning_rate_decay
      rmsprop_config.learningRate = rmsprop_config.learningRate * decay_factor -- decay it
      opt.learning_rate = rmsprop_config.learningRate -- update the learning rate in opt
      -- reinit the parameters  
      model.rnn_params_flat:uniform(-0.1, 0.1)
      model.top_net_params_flat:uniform(-0.1, 0.1)
      print('the loss blowup probably because the learning rate is too big...\n' ..
        'decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rmsprop_config.learningRate)
    end
    
    local function isnan(x) return x ~= x end
    if isnan(train_losses[i]) then
      --reinit the parameters  
      model.rnn_params_flat:uniform(-0.1, 0.1)
      model.top_net_params_flat:uniform(-0.1, 0.1)
      local decay_factor = opt.learning_rate_decay
      rmsprop_config.learningRate = rmsprop_config.learningRate * decay_factor -- decay it
      opt.learning_rate = rmsprop_config.learningRate -- update the learning rate in opt
      print('loss is NaN. .' .. 
      'decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rmsprop_config.learningRate)
    end

    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs", 
        i, iterations, epoch, train_losses[i], time_current_iteration))
    end
    
    if i * opt.batch_size % opt.evaluate_every == 0 then
      local temp_sum_loss = torch.sum(train_losses:sub(i - opt.evaluate_every/opt.batch_size+1, i))
      local temp_mean_loss = temp_sum_loss / opt.evaluate_every * opt.batch_size
      print(string.format('average loss in the last %d iterations = %6.8f', opt.evaluate_every, temp_mean_loss))
      print('learning rate: ', opt.learning_rate)
--      local whole_train_loss, whole_train_accuracy, auc_train = evaluate_process.evaluate_training_set(opt, loader, model)
      local whole_validation_loss,  auc_validation, EER_validation = evaluate_process.evaluate_validation_set(opt, loader, model)
      local time_e = timer:time().real
      print(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs', 
        opt.evaluate_every, time_e-time_s, time_e))
      if auc_validation > current_best_auc then
        current_best_auc = auc_validation
        better_times_total = 0
        better_times_decay = 0
        save_model()
      else
        better_times_total = better_times_total + 1
        better_times_decay = better_times_decay + 1
        if better_times_total >= opt.stop_iteration_threshold then
          print(string.format('no more better result in %d iterations! hence stop the optimization!', 
            opt.stop_iteration_threshold))
          break
        elseif better_times_decay >= opt.decay_threshold then
          print(string.format('no more better result in %d iterations! hence decay the learning rate', 
            opt.decay_threshold))
          local decay_factor = opt.learning_rate_decay
          rmsprop_config.learningRate = rmsprop_config.learningRate * decay_factor -- decay it
          opt.learning_rate = rmsprop_config.learningRate -- update the learning rate in opt
          print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rmsprop_config.learningRate)
          better_times_decay = 0 
          -- back to the currently optimized point
          print('back to the currently best optimized point...')
          local checkpoint = torch.load(savefile)
          model = checkpoint.model
        end
      end     
      print('better times: ', better_times_total, '\n\n')
      -- save to log file
      local temp_file = nil
      if i  == 1 and not opt.if_init_from_check_point then
        temp_file = io.open(string.format('%s/%s_results_GPU_%d_lambda_%1.6f_dropout_%1.2f_index_%d_seed_%d.txt',
           opt.current_result_dir, opt.opt_method, opt.gpuid, opt.lambda, opt.dropout, opt.data_index, opt.seed), "w")
      else
        temp_file = io.open(string.format('%s/%s_results_GPU_%d_lambda_%1.6f_dropout_%1.2f_index_%d_seed_%d.txt', 
          opt.current_result_dir, opt.opt_method, opt.gpuid, opt.lambda, opt.dropout, opt.data_index, opt.seed), "a")
      end
      temp_file:write('better times: ', better_times_total, '\n')
      temp_file:write('learning rate: ', opt.learning_rate, '\n')
      temp_file:write(string.format("%d/%d (epoch %.3f) \n", i, iterations, epoch))
      temp_file:write(string.format('average loss in the last %d (%5d -- %5d) iterations = %6.8f \n', 
        opt.evaluate_every/opt.batch_size, i-opt.evaluate_every/opt.batch_size+1, i, temp_mean_loss))
--      temp_file:write(string.format('training set loss = %6.8f, train accuracy = %6.8f, train auc = %6.8f\n', 
--        whole_train_loss, whole_train_accuracy, auc_train ))
      temp_file:write(string.format('validation set loss = %6.8f, validation auc= %6.8f, validation EER = %6.3f\n', 
        whole_validation_loss, auc_validation, EER_validation ))
      temp_file:write(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs\n', 
        opt.evaluate_every, time_e-time_s, time_e))
      temp_file:write(string.format('\n'))
      temp_file.close()
      time_s = time_e
    end
  end
  local time_e = timer:time().real
  print('total elapsed time:', time_e)
end

return train_process
    
    