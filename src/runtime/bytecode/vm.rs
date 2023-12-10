use super::model;
use core::fmt;

static mut INTERPRETED_BYTECODE_COUNT: usize = 0;

pub struct Interpreter {
    stack: Vec<model::Value>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(64),
        }
    }

    pub fn run(mut self, executable: Executable) -> model::Value {
        let return_value = self.run_automat(&executable, executable.entry_point());

        let count = unsafe { INTERPRETED_BYTECODE_COUNT };
        println!("Interpreted {count} bytecodes.");

        return_value
    }

    fn run_automat(&mut self, executable: &Executable, start: model::Label) -> model::Value {
        let mut frame = ActivationFrame::default();
        frame.continue_at(start);

        while let Continuation::Resume(model::Label(block_id)) = frame.continuation {
            let block = executable.resolve_block(block_id);
            self.interpret_block(executable, &mut frame, block)
        }

        let Continuation::Return(return_value) = frame.continuation else {
            unreachable!()
        };

        return_value
    }

    fn interpret_block(
        &mut self,
        executable: &Executable,
        frame: &mut ActivationFrame,
        block: &model::BasicBlock,
    ) {
        for bytecode in block.instruction_stream() {
            unsafe {
                INTERPRETED_BYTECODE_COUNT += 1;
            }
            self.interpret(executable, frame, bytecode)
        }
    }

    fn duplicate_top_of_stack(&mut self) {
        let index = self.stack.len() - 1;
        let element = self.stack[index].clone();
        self.stack.push(element);
    }

    fn interpret(
        &mut self,
        executable: &Executable,
        frame: &mut ActivationFrame,
        bytecode: &model::Bytecode,
    ) {
        match bytecode {
            model::Bytecode::LoadConstant(constant) => self.stack.push(constant.clone()),
            model::Bytecode::LoadLocal(index) => self.stack.push(frame.get_local(*index).clone()),
            model::Bytecode::StoreLocal(index) => {
                frame.put_local(*index, self.stack.pop().expect("expected 1 stack operands"))
            }
            model::Bytecode::Arithmetic(op) => {
                // make a thing that will pop a specific type.
                // pop the first, require the second to have the same type
                let rhs = self.stack.pop().expect("expected 1 stack operands");
                let lhs = self.stack.pop().expect("expected 1 stack operands");
                self.stack.push(lhs.apply(op, rhs))
            }
            model::Bytecode::Invoke(index) => {
                let target = executable.resolve_function_target(*index).target;
                let return_value = self.run_automat(executable, target);
                self.stack.push(return_value)
            }
            model::Bytecode::InvokeBuiltin(index, arg_count) => {
                let target = executable.resolve_builtin_target(*index);
                let mut arguments = Vec::with_capacity(*arg_count as usize);
                for _ in 0..*arg_count {
                    // Check and compare types here.
                    // insert(0, ...) because the functions expect the parameters
                    // in their natural order; popping them off a stack reverses that
                    arguments.insert(0, self.stack.pop().unwrap());
                }
                let return_value = target.stub.call(&arguments).expect("Call failed");
                self.stack.push(return_value);
            }
            model::Bytecode::Return => {
                let return_value = self.stack.pop().expect("no value to return on the stack");
                frame.make_return(return_value)
            }
            model::Bytecode::Dup => self.duplicate_top_of_stack(),
            model::Bytecode::Discard => {
                self.stack
                    .pop()
                    .expect("Discarding a non-existent top of stack");
            }
            model::Bytecode::Jump(target) => frame.continue_at(*target),
            model::Bytecode::ConditionalJump(consequent, alternate) => {
                let top = self
                    .stack
                    .pop()
                    .expect("Expected a (boolean) at the top of the stack");
                if let model::Value::Boolean(test) = top {
                    if test {
                        frame.continue_at(*consequent);
                    } else {
                        frame.continue_at(*alternate);
                    }
                } else {
                    panic!("Mis-typed if statement.")
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct Executable {
    pub constants: Vec<model::Value>,
    pub functions: Vec<model::Function>,
    pub builtins: Vec<model::BuiltinFunction>,
    pub entry_point_block_id: usize,
    pub blocks: Vec<model::BasicBlock>,
}

impl Executable {
    fn resolve_function_target(&self, index: u16) -> &model::Function {
        &self.functions[index as usize]
    }

    fn resolve_builtin_target(&self, index: u16) -> &model::BuiltinFunction {
        &self.builtins[index as usize]
    }

    fn entry_point(&self) -> model::Label {
        model::Label(self.entry_point_block_id as u16)
    }

    fn resolve_block(&self, id: u16) -> &model::BasicBlock {
        &self.blocks[id as usize]
    }
}

impl fmt::Display for Executable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Executable {
            constants,
            functions,
            builtins,
            entry_point_block_id,
            blocks,
        } = self;
        writeln!(f, "Constants:")?;
        for (index, constant) in constants.iter().enumerate() {
            writeln!(f, "{index}: {constant:?}")?;
        }
        writeln!(f, "")?;

        writeln!(f, "Functions:")?;
        for (index, function) in functions.iter().enumerate() {
            write!(f, "{index}: {function}")?;
        }
        writeln!(f, "")?;

        writeln!(f, "Builtin functions:")?;
        for (index, builtin) in builtins.iter().enumerate() {
            write!(f, "{index}: {builtin}")?;
        }
        writeln!(f, "")?;

        writeln!(f, "\nEntry point:")?;
        for bytecode in self.blocks[*entry_point_block_id].instruction_stream() {
            writeln!(f, "\t{bytecode}")?;
        }

        writeln!(f, "\nBlocks:")?;
        for (index, block) in blocks.iter().enumerate() {
            writeln!(f, "#{index}:")?;
            for bytecode in block.instruction_stream() {
                writeln!(f, "\t{bytecode}")?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct ActivationFrame {
    locals: Vec<Option<model::Value>>,
    continuation: Continuation,
}

impl Default for ActivationFrame {
    fn default() -> Self {
        Self {
            locals: vec![None; 8],
            continuation: Continuation::default(),
        }
    }
}

impl ActivationFrame {
    fn put_local(&mut self, index: u8, value: model::Value) {
        for _ in self.locals.len()..=(index as usize) {
            self.locals.push(None)
        }
        self.locals[index as usize] = Some(value);
    }

    fn get_local(&self, index: u8) -> &model::Value {
        if let Some(x) = &self.locals[index as usize] {
            &x
        } else {
            panic!("Reading unitialzed local slot")
        }
    }

    fn make_return(&mut self, val: model::Value) {
        self.continuation = Continuation::Return(val)
    }

    fn continue_at(&mut self, target: model::Label) {
        self.continuation = Continuation::Resume(target)
    }
}

#[derive(Debug)]
enum Continuation {
    Return(model::Value),
    Resume(model::Label),
}

impl Default for Continuation {
    fn default() -> Self {
        Self::Return(model::Value::default())
    }
}
