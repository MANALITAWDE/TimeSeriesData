local path = require 'pl.path'
--local matio = require 'matio'
local table_operation = require 'util.table_operation'


local data_loader = {}
data_loader.__index = data_loader


function data_loader.create(data_dir, opt, if_zero_shot_data)

  local self = {}
  setmetatable(self, data_loader)
  
  if_zero_shot_data = if_zero_shot_data or false
  
  local index_file = nil
  if opt.data_name == 'arabic_voice_single' then
    data_dir = path.join(data_dir, 'digit_' .. opt.digit_index)
  end
  local data_file = path.join(data_dir, 'processed_data_window_' .. opt.window_size .. '.t7')
  if not if_zero_shot_data then
    index_file = path.join(data_dir, opt.data_name .. '_pairs_X_' .. opt.class_number .. '_' .. opt.data_index .. '.t7')
  else
    index_file = path.join(data_dir, opt.data_name .. '_pairs_X_zero_shot.t7')
  end
  self:load_data_by_index_t7(opt, data_file, index_file, if_zero_shot_data)
  self.feature_dim = self.test_X_sequence_1[1]:size(1);
  print('Feature dimension: ', self.feature_dim);
  
  -- calculate the max time series length
  local max_length = 0
  if not if_zero_shot_data then
    for i = 1, self.nTrain do
      max_length = math.max(self.train_X_sequence_1[i]:size(2), max_length)
      max_length = math.max(self.train_X_sequence_2[i]:size(2), max_length)
    end
    for i = 1, self.nValidation do
      max_length = math.max(self.validation_X_sequence_1[i]:size(2), max_length)
      max_length = math.max(self.validation_X_sequence_2[i]:size(2), max_length)
    end
  end
  for i = 1, self.nTest do
    max_length = math.max(self.test_X_sequence_1[i]:size(2), max_length)
    max_length = math.max(self.test_X_sequence_2[i]:size(2), max_length)
  end
  print('The max length of the time series in the generated data set (pairs): ', max_length)
  self.max_time_series_length = max_length 

  -- analyze the time series length
  if not if_zero_shot_data then
    local total_length = 0
    local length_tensor = torch.zeros(2*(self.nTrain+self.nTest+self.nValidation))
    local append = 0
    for i = 1, self.nTrain do
      total_length = total_length + self.train_X_sequence_1[i]:size(2)
      total_length = total_length + self.train_X_sequence_2[i]:size(2)
      length_tensor[2*i-1] = self.train_X_sequence_1[i]:size(2)
      length_tensor[2*i] = self.train_X_sequence_2[i]:size(2)
    end
    append = append + 2 * self.nTrain
    for i = 1, self.nValidation do
      total_length = total_length + self.validation_X_sequence_1[i]:size(2)
      total_length = total_length + self.validation_X_sequence_2[i]:size(2)
      length_tensor[append+2*i-1] = self.validation_X_sequence_1[i]:size(2)
      length_tensor[append+2*i] = self.validation_X_sequence_2[i]:size(2)
    end
    append = append + 2 * self.nValidation
    for i = 1, self.nTest do
      total_length = total_length + self.test_X_sequence_1[i]:size(2)
      total_length = total_length + self.test_X_sequence_2[i]:size(2)
      length_tensor[append+2*i-1] = self.test_X_sequence_1[i]:size(2)
      length_tensor[append+2*i] = self.test_X_sequence_2[i]:size(2)
    end
    local average_frames = total_length / (2*(self.nTrain+self.nValidation+self.nTest))
    local std = torch.std(length_tensor)
    print('the average frames in the data set is: ', average_frames)
    print('the standard deviation of the length is: ', std)
    
--    -- ship the model to the GPU if desired
--    if opt.gpuid >= 0 and opt.opencl == 0 then
--      for i = 1, self.nTrain do
--        self.train_X_sequence_1[i] = self.train_X_sequence_1[i]:float():cuda()
--        self.train_X_sequence_2[i] = self.train_X_sequence_2[i]:float():cuda()
--      end
--      for i = 1, self.nValidation do
--        self.validation_X_sequence_1[i] = self.validation_X_sequence_1[i]:float():cuda()
--        self.validation_X_sequence_2[i] = self.validation_X_sequence_2[i]:float():cuda()
--      end
--      for i = 1, self.nTest do
--        self.test_X_sequence_1[i] = self.test_X_sequence_1[i]:float():cuda()
--        self.test_X_sequence_2[i] = self.test_X_sequence_2[i]:float():cuda()
--      end
--
--      self.train_T = self.train_T:float():cuda()
--      self.validation_T = self.validation_T:float():cuda()
--      self.test_T = self.test_T:float():cuda()
--    end  
    
    self.batch_ix = {1, 1}
    self.previous_batch_ix = table_operation.shallowCopy(self.batch_ix)
    self.randperm_index = torch.randperm(self.nTrain)
  end
  print('data load done. ')
  collectgarbage()
  return self
end

-- Return the tensor of index of the data with batch size 
function data_loader:get_next_batch(set_name, batch_size)
  self.previous_batch_ix = table_operation.shallowCopy(self.batch_ix)
  local ind = {train = 1, test = 2}
  local set_size = 0;
  if set_name == 'train' then
    set_size = self.nTrain;
  else
    set_size = self.nTest;
  end
  
  local rtn_index = torch.zeros(batch_size)
--  print('current index starting point: ', self.batch_ix[ind[set_name]]);
  for i = 1, batch_size do
    local temp_ind = i + self.batch_ix[ind[set_name]] - 1
    if temp_ind > set_size then -- cycle around to beginning
      temp_ind = temp_ind - set_size
    end
    rtn_index[i] = self.randperm_index[temp_ind]
  end
  self.batch_ix[ind[set_name]] = self.batch_ix[ind[set_name]] + batch_size;
  -- cycle around to beginning
  if self.batch_ix[ind[set_name]] >set_size then
    self.batch_ix[ind[set_name]] = self.batch_ix[ind[set_name]] - set_size -- cycle around to beginning
  end
  return rtn_index;
end

function data_loader:recover_batch_ix()
  self.batch_ix = table_operation.shallowCopy(self.previous_batch_ix)
end

function data_loader.data_to_tensor(input_train_file, input_test_file, tensor_file)
  -- load the data
  local train_X_sequence_1, train_X_sequence_2, test_X_sequence_1, test_X_sequence_2, 
    validation_X_sequence_1, validation_X_sequence_2,train_T, validation_T, test_T, 
    train_sequence_length, validation_sequence_length, test_sequence_length

  local data = matio.load(input_train_file)
  -- there is no training data and validation data in the zero-shot data
  if data.train_pair_labels then
    train_sequence_length = data.train_pair_labels:size(1)
    print('the length of the training sequence: ', train_sequence_length)
    train_X_sequence_1 = table_operation.subrange(data.train_pair_X, 1, train_sequence_length) -- a table of Tensor
    train_X_sequence_2 = table_operation.subrange(data.train_pair_X,
      train_sequence_length+1, train_sequence_length * 2) -- a table of Tensor
    train_T = data.train_pair_labels

    validation_sequence_length = data.validation_pair_labels:size(1)
    print('the length of the validation sequence: ', validation_sequence_length)
    validation_X_sequence_1 = table_operation.subrange(data.validation_pair_X, 1, validation_sequence_length) -- a table of Tensor
    validation_X_sequence_2 = table_operation.subrange(data.validation_pair_X, validation_sequence_length+1, 
      validation_sequence_length * 2) -- a table of Tensor
    validation_T = data.validation_pair_labels -- a Tensor
  end
  test_sequence_length = data.test_pair_labels:size(1)
  print('the length of the test sequence: ', test_sequence_length)
  test_X_sequence_1 = table_operation.subrange(data.test_pair_X, 1, test_sequence_length) -- a table of Tensor
  test_X_sequence_2 = table_operation.subrange(data.test_pair_X, test_sequence_length+1, 
    test_sequence_length * 2) -- a table of Tensor
  test_T = data.test_pair_labels -- a Tensor
  
  -- save output preprocessed files
  print('saving ' .. tensor_file)
  local data = {}
  table.insert(data, train_X_sequence_1);
  table.insert(data, train_X_sequence_2);
  table.insert(data, train_T);
    table.insert(data, validation_X_sequence_1);
  table.insert(data, validation_X_sequence_2);
  table.insert(data, validation_T);
  table.insert(data, test_X_sequence_1);
  table.insert(data, test_X_sequence_2);
  table.insert(data, test_T);
  torch.save(tensor_file, data)
  
end

function data_loader:load_data_by_index_t7(opt, data_file, index_file, if_zero_shot)
  print('loading data: '.. index_file)
  local data = torch.load(data_file)
  local indexd = torch.load(index_file)
  local data_X = data.new_X
  local data_T = data.new_labels
  
  local max_length = 0
  local length_tensor = torch.zeros(#data_X)
  for i = 1, #data_X do
    length_tensor[i] = data_X[i]:size(2)
  end
  self.length_tensor = length_tensor
  max_length = torch.max(length_tensor)
  local std_length = torch.std(length_tensor)
  local mean_length = torch.mean(length_tensor)
  print('The max length of the time series in the original data set: ', max_length)
  print('The mean length of the time series in the original data set: ', mean_length)
  print('The std variance of the time series in the original data set: ', std_length)

  -- ship the model to the GPU if desired
  if opt.gpuid >= 0 and opt.opencl == 0 then
    self.length_tensor = self.length_tensor:float():cuda()
    for i = 1, #data_X do
      data_X[i] = data_X[i]:float():cuda()
    end
    if not if_zero_shot then
      indexd.train_labels = indexd.train_labels:float():cuda()
      indexd.validation_labels = indexd.validation_labels:float():cuda()
    end
      indexd.test_labels = indexd.test_labels:float():cuda()
  end    

  if not if_zero_shot then
    self.train_T = indexd.train_labels
    self.nTrain = self.train_T:size(1)
    self.train_X_sequence_1 = {}
    self.train_X_sequence_2 = {}
    for i = 1, self.nTrain do
      local ind = indexd.train_pair[i]
      self.train_X_sequence_1[i] = data_X[ind[1]]
      self.train_X_sequence_2[i] = data_X[ind[2]]
    end
    self.validation_T = indexd.validation_labels
    self.nValidation = self.validation_T:size(1)
    self.validation_X_sequence_1 = {}
    self.validation_X_sequence_2 = {}
    for i = 1, self.nValidation do
      local ind = indexd.validation_pair[i]
      self.validation_X_sequence_1[i] = data_X[ind[1]]
      self.validation_X_sequence_2[i] = data_X[ind[2]]
    end
    local positive_train = torch.sum(self.train_T:eq(1)) -- positive number
    print('The length of training pairs: ' .. self.nTrain, 'positive ratio: '.. positive_train/self.nTrain)
    local positive_validation = torch.sum(self.validation_T:eq(1))
    print('The length of validation pairs: ' .. self.nValidation, 'positive ratio: '.. positive_validation/self.nValidation)
  end
  self.test_T = indexd.test_labels
  self.nTest = self.test_T:size(1)
  self.test_X_sequence_1 = {}
  self.test_X_sequence_2 = {}
  for i = 1, self.nTest do
    local ind = indexd.test_pair[i]
    self.test_X_sequence_1[i] = data_X[ind[1]]
    self.test_X_sequence_2[i] = data_X[ind[2]]
  end
  local positive_test = torch.sum(self.test_T:eq(1)) -- positive number
  print('The length of test pairs: ' .. self.nTest, 'positive ratio: '.. positive_test/self.nTest)
end


function data_loader:load_data_by_index_mat(data_file, index_file, if_zero_shot)
  print('loading data: '.. index_file)
  local data = matio.load(data_file)
  local indexd = matio.load(index_file)
  local data_X = data.new_X
  local data_T = data.new_labels

  if not if_zero_shot then
    self.train_T = indexd.train_labels
    self.nTrain = self.train_T:size(1)
    self.train_X_sequence_1 = {}
    self.train_X_sequence_2 = {}
    for i = 1, self.nTrain do
      local ind = indexd.train_pair[i]
      self.train_X_sequence_1[i] = data_X[ind[1]]
      self.train_X_sequence_2[i] = data_X[ind[2]]
    end
    self.validation_T = indexd.validation_labels
    self.nValidation = self.validation_T:size(1)
    self.validation_X_sequence_1 = {}
    self.validation_X_sequence_2 = {}
    for i = 1, self.nValidation do
      local ind = indexd.validation_pair[i]
      self.validation_X_sequence_1[i] = data_X[ind[1]]
      self.validation_X_sequence_2[i] = data_X[ind[2]]
    end
    local positive_train = torch.sum(self.train_T:eq(1)) -- positive number
    print('The length of training pairs: ' .. self.nTrain, 'positive ratio: '.. positive_train/self.nTrain)
    local positive_validation = torch.sum(self.validation_T:eq(1))
    print('The length of validation pairs: ' .. self.nValidation, 'positive ratio: '.. positive_validation/self.nValidation)
  end
  self.test_T = indexd.test_labels
  self.nTest = self.test_T:size(1)
  self.test_X_sequence_1 = {}
  self.test_X_sequence_2 = {}
  for i = 1, self.nTest do
    local ind = indexd.test_pair[i]
    self.test_X_sequence_1[i] = data_X[ind[1]]
    self.test_X_sequence_2[i] = data_X[ind[2]]
  end
  local positive_test = torch.sum(self.test_T:eq(1)) -- positive number
  print('The length of test pairs: ' .. self.nTest, 'positive ratio: '.. positive_test/self.nTest)
end


return data_loader
