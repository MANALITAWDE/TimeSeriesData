
local matio = require 'matio'
local path = require 'pl.path'

function mat_to_t7_data(input_mat_path)
  print('loading data: '.. input_mat_path)
  local data = matio.load(input_mat_path)
  local output_path = string.sub(input_mat_path, 1, input_mat_path:len()-4)-- remove '.mat'
  local output = string.format(output_path .. '.t7')
  torch.save(output, data)

end

