
require 'convert_mat_to_t7_data'
local path = require 'pl.path'


local function dirLookup(dir)
  local all_files = {}
   local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
   for file in p:lines() do                         --Loop through all files
       print(file)
       all_files[#all_files+1] = file       
   end
   return all_files
end


function convert_TiSSiLe_mat_to_t7()

  local data_name = {'MCYT', 'MCYT_with_forgery', 'arabic', 'arabic_voice', 'Sign'}
  for i = 1, #data_name do
    local pathd = path.join('../rnn_TiSSiLe_data', data_name[i] )
    local all_files = dirLookup(pathd)
    for i = 1, #all_files do
      mat_to_t7_data(all_files[i])
    end
  end

end



convert_TiSSiLe_mat_to_t7()