
--[[

This is the implementation for the MulConst module, which is modified from nn.Mul() module.

]]--


local MulConst, parent = torch.class('nn.MulConst', 'nn.Module')

function MulConst:__init(const)
   assert(self)
   parent.__init(self)
   self:setConst(const)
end

function MulConst:updateOutput(input)
   assert(self and input and torch.isTensor(input))
   self.output:resizeAs(input):copy(input);
   self.output:mul(self.const);
   return self.output
end

function MulConst:updateGradInput(input, gradOutput)
   assert(self and input      and torch.isTensor(input)
               and gradOutput and torch.isTensor(gradOutput))
   self.gradInput:resizeAs(input):zero()
   self.gradInput:add(self.const, gradOutput)
   return self.gradInput
end

function MulConst:setConst(const)
   assert(const and type(const) == 'number')
   self.const = const
end