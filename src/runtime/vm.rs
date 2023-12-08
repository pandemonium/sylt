// Stfu for a second, k?
#![allow(dead_code)]
use crate::ast;
use core::fmt;
use std::rc;

static mut INTERPRETED_BYTECODE_COUNT: usize = 0;

pub struct Interpreter {
    stack: Vec<Value>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(64),
        }
    }

    pub fn run(mut self, executable: Executable) -> Value {
        let return_value = self.run_automat(&executable, executable.entry_point());

        let count = unsafe { INTERPRETED_BYTECODE_COUNT };
        println!("Interpreted {count} bytecodes.");

        return_value
    }

    fn run_automat(&mut self, executable: &Executable, start: Label) -> Value {
        let mut frame = ActivationFrame::default();
        frame.continue_at(start);

        while let Continuation::Resume(Label(block_id)) = frame.continuation {
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
        block: &BasicBlock,
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
        bytecode: &Bytecode,
    ) {
        match bytecode {
            Bytecode::LoadConstant(constant) => self.stack.push(constant.clone()),
            Bytecode::LoadLocal(index) => self.stack.push(frame.get_local(*index).clone()),
            Bytecode::StoreLocal(index) => {
                frame.put_local(*index, self.stack.pop().expect("expected 1 stack operands"))
            }
            Bytecode::Arithmetic(op) => {
                // make a thing that will pop a specific type.
                // pop the first, require the second to have the same type
                let rhs = self.stack.pop().expect("expected 1 stack operands");
                let lhs = self.stack.pop().expect("expected 1 stack operands");
                self.stack.push(lhs.apply(op, rhs))
            }
            Bytecode::Invoke(index) => {
                let target = executable.resolve_function_target(*index).target;
                let return_value = self.run_automat(executable, target);
                self.stack.push(return_value)
            }
            Bytecode::InvokeBuiltin(index) => {
                let target = executable.resolve_builtin_target(*index);
                let mut arguments = Vec::with_capacity(target.prototype.len());
                for _ in &target.prototype {
                    // Check and compare types here.
                    // insert(0, ...) because the functions expect the parameters
                    // in their natural order; popping them off a stack reverses that
                    arguments.insert(0, self.stack.pop().unwrap());
                }
                let return_value = target.stub.call(&arguments).expect("Call failed");
                self.stack.push(return_value);
            }
            Bytecode::Return => {
                let return_value = self.stack.pop().expect("no value to return on the stack");
                frame.make_return(return_value)
            }
            Bytecode::Dup => self.duplicate_top_of_stack(),
            Bytecode::Discard => {
                self.stack
                    .pop()
                    .expect("Discarding a non-existent top of stack");
            }
            Bytecode::Jump(target) => frame.continue_at(*target),
            Bytecode::ConditionalJump(consequent, alternate) => {
                let top = self
                    .stack
                    .pop()
                    .expect("Expected a (boolean) at the top of the stack");
                if let Value::Boolean(test) = top {
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

pub fn compile(program: ast::Program) -> Executable {
    let mut compile = Compile::default();
    compile.register_builtin(BuiltinFunction {
        name: ast::Name::simple("print_line"),
        prototype: vec![ast::Type::named(&ast::Name::intrinsic("Text"))],
        stub: rc::Rc::new(PrintLine),
    });
    compile.program(program);
    compile.link()
}

#[derive(Debug, Default)]
pub struct Executable {
    constants: Vec<Value>,
    functions: Vec<Function>,
    builtins: Vec<BuiltinFunction>,
    entry_point_block_id: usize,
    blocks: Vec<BasicBlock>,
}

impl Executable {
    fn resolve_function_target(&self, index: u16) -> &Function {
        &self.functions[index as usize]
    }

    fn resolve_builtin_target(&self, index: u16) -> &BuiltinFunction {
        &self.builtins[index as usize]
    }

    fn entry_point(&self) -> Label {
        Label(self.entry_point_block_id as u16)
    }

    fn resolve_block(&self, id: u16) -> &BasicBlock {
        &self.blocks[id as usize]
    }
}

#[derive(Debug)]
struct BuiltinFunction {
    name: ast::Name,
    prototype: Vec<ast::Type>,
    stub: rc::Rc<dyn BuiltinStub>,
}

trait BuiltinStub: fmt::Debug {
    fn call(&self, parameters: &[Value]) -> Option<Value>;
}

#[derive(Debug)]
struct PrintLine;

impl BuiltinStub for PrintLine {
    fn call(&self, parameters: &[Value]) -> Option<Value> {
        for p in parameters {
            println!("{p}")
        }
        Some(Value::Unit)
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
struct Function {
    name: ast::Name,
    target: Label,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Function {
            name,
            target: Label(target),
        } = self;
        writeln!(f, "{}, at #{target}", name.local_name())
    }
}

#[derive(Debug)]
struct ActivationFrame {
    locals: Vec<Option<Value>>,
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
    fn put_local(&mut self, index: u8, value: Value) {
        for _ in self.locals.len()..=(index as usize) {
            self.locals.push(None)
        }
        self.locals[index as usize] = Some(value);
    }

    fn get_local(&self, index: u8) -> &Value {
        if let Some(x) = &self.locals[index as usize] {
            &x
        } else {
            panic!("Reading unitialzed local slot")
        }
    }

    fn make_return(&mut self, val: Value) {
        self.continuation = Continuation::Return(val)
    }

    fn continue_at(&mut self, target: Label) {
        self.continuation = Continuation::Resume(target)
    }
}

#[derive(Debug)]
enum Continuation {
    Return(Value),
    Resume(Label),
}

impl Default for Continuation {
    fn default() -> Self {
        Self::Return(Value::default())
    }
}

#[derive(Clone, Debug, Default)]
pub enum Value {
    Int(i64),
    Float(f64),
    Boolean(bool),
    Text(Box<str>),
    #[default]
    Unit,
}

fn try_compute(lhs: &Value, op: &AluOp, rhs: &Value) -> Option<Value> {
    use AluOp::*;
    use Value::*;

    // Can this be done golfer than this?
    match (lhs, op, rhs) {
        (Float(lhs), Add, Float(rhs)) => Some(Float(lhs + rhs)),
        (Int(lhs), Add, Int(rhs)) => Some(Int(lhs + rhs)),

        (Float(lhs), Subtract, Float(rhs)) => Some(Float(lhs - rhs)),
        (Int(lhs), Subtract, Int(rhs)) => Some(Int(lhs - rhs)),

        (Float(lhs), Multiply, Float(rhs)) => Some(Float(lhs * rhs)),
        (Int(lhs), Multiply, Int(rhs)) => Some(Int(lhs * rhs)),

        (Float(lhs), Divide, Float(rhs)) => Some(Float(lhs / rhs)),
        (Int(lhs), Divide, Int(rhs)) => Some(Int(lhs / rhs)),

        (Float(lhs), Modulo, Float(rhs)) => Some(Float(lhs % rhs)),
        (Int(lhs), Modulo, Int(rhs)) => Some(Int(lhs % rhs)),

        (Float(lhs), Lt, Float(rhs)) => Some(Boolean(lhs < rhs)),
        (Int(lhs), Lt, Int(rhs)) => Some(Boolean(lhs < rhs)),

        (Float(lhs), Equals, Float(rhs)) => Some(Boolean(lhs == rhs)),
        (Int(lhs), Equals, Int(rhs)) => Some(Boolean(lhs == rhs)),
        (Boolean(lhs), Equals, Boolean(rhs)) => Some(Boolean(lhs == rhs)),
        (Text(lhs), Equals, Text(rhs)) => Some(Boolean(lhs == rhs)),
        (Unit, Equals, Unit) => Some(Boolean(true)),

        (Float(lhs), NotEqual, Float(rhs)) => Some(Boolean(lhs != rhs)),
        (Int(lhs), NotEqual, Int(rhs)) => Some(Boolean(lhs != rhs)),
        (Boolean(lhs), NotEqual, Boolean(rhs)) => Some(Boolean(lhs != rhs)),
        (Text(lhs), NotEqual, Text(rhs)) => Some(Boolean(lhs != rhs)),
        (Unit, NotEqual, Unit) => Some(Boolean(false)),

        (Float(lhs), Gt, Float(rhs)) => Some(Boolean(lhs > rhs)),
        (Int(lhs), Gt, Int(rhs)) => Some(Boolean(lhs > rhs)),

        (Float(lhs), Lte, Float(rhs)) => Some(Boolean(lhs <= rhs)),
        (Int(lhs), Lte, Int(rhs)) => Some(Boolean(lhs <= rhs)),

        (Float(lhs), Gte, Float(rhs)) => Some(Boolean(lhs >= rhs)),
        (Int(lhs), Gte, Int(rhs)) => Some(Boolean(lhs >= rhs)),

        //        (Boolean(lhs), And, Boolean(rhs)) => Some(Boolean(*lhs && *rhs)),
        //        (Boolean(lhs), Or, Boolean(rhs)) => Some(Boolean(*lhs || *rhs)),
        _otherwise => None,
    }
}

impl Value {
    fn apply(self, op: &AluOp, rhs: Self) -> Self {
        try_compute(&self, &op, &rhs).expect("Not applicable")
    }
}

impl From<AluOp> for ast::Operator {
    fn from(value: AluOp) -> Self {
        match value {
            AluOp::Add => Self::Plus,
            AluOp::Subtract => Self::Minus,
            AluOp::Multiply => Self::Times,
            AluOp::Divide => Self::Divides,
            AluOp::Modulo => Self::Modulo,
            AluOp::Lte => Self::LTE,
            AluOp::Gte => Self::GTE,
            AluOp::Lt => Self::LT,
            AluOp::Gt => Self::GT,
            AluOp::And => Self::And,
            AluOp::Or => Self::Or,
            AluOp::Equals => Self::Equals,
            AluOp::NotEqual => Self::NotEqual,
        }
    }
}

impl From<Value> for ast::Constant {
    fn from(value: Value) -> Self {
        match value {
            Value::Int(x) => Self::Int(x),
            Value::Float(x) => Self::Float(x),
            Value::Boolean(x) => Self::Boolean(x),
            Value::Text(x) => Self::Text(x.into_string()),
            Value::Unit => Self::Void,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(x) => write!(f, "#{x}"),
            Value::Float(x) => write!(f, "#{x}"),
            Value::Boolean(x) => write!(f, "{}", if *x { "True" } else { "False" }),
            Value::Text(x) => write!(f, "'{x}'"),
            Value::Unit => write!(f, "()"),
        }
    }
}

impl From<ast::Constant> for Value {
    fn from(value: ast::Constant) -> Self {
        match value {
            ast::Constant::Boolean(x) => Value::Boolean(x),
            ast::Constant::Int(x) => Value::Int(x),
            ast::Constant::Float(x) => Value::Float(x),
            ast::Constant::Text(x) => Value::Text(x.into()),
            ast::Constant::Void => Value::Unit,
        }
    }
}

// Am I going to be working with registers or a stack?
#[derive(Debug, Clone)]
enum Bytecode {
    // Is there a string table too or just one for constants?
    LoadConstant(Value),

    LoadLocal(u8),
    StoreLocal(u8),

    Arithmetic(AluOp),

    InvokeBuiltin(u16),
    Invoke(u16),
    Return,
    Dup,
    Discard,

    Jump(Label),
    ConditionalJump(Label, Label),
}

impl Bytecode {
    fn is_block_terminator(&self) -> bool {
        // Terminate on calls too?
        matches!(
            self,
            Self::Jump(..) | Self::ConditionalJump(..) | Self::Return
        )
    }
}

impl fmt::Display for Bytecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bytecode::LoadConstant(x) => write!(f, "constant\t{x}"),
            Bytecode::LoadLocal(x) => write!(f, "load\t\t{x}"),
            Bytecode::StoreLocal(x) => write!(f, "store\t\t{x}"),
            Bytecode::Arithmetic(AluOp::Add) => write!(f, "add"),
            Bytecode::Arithmetic(AluOp::Subtract) => write!(f, "sub"),
            Bytecode::Arithmetic(AluOp::Multiply) => write!(f, "mul"),
            Bytecode::Arithmetic(AluOp::Divide) => write!(f, "div"),
            Bytecode::Arithmetic(AluOp::Modulo) => write!(f, "mod"),
            Bytecode::Arithmetic(AluOp::Equals) => write!(f, "eq"),
            Bytecode::Arithmetic(AluOp::NotEqual) => write!(f, "neq"),
            Bytecode::Arithmetic(AluOp::Lt) => write!(f, "lt"),
            Bytecode::Arithmetic(AluOp::Gt) => write!(f, "gt"),
            Bytecode::Arithmetic(AluOp::Lte) => write!(f, "lte"),
            Bytecode::Arithmetic(AluOp::Gte) => write!(f, "gte"),
            Bytecode::Arithmetic(AluOp::And) => write!(f, "and"),
            Bytecode::Arithmetic(AluOp::Or) => write!(f, "or"),
            Bytecode::Invoke(x) => write!(f, "invoke\t\t{x}"),
            Bytecode::InvokeBuiltin(x) => write!(f, "invoke_builtin\t{x}"),
            Bytecode::Return => write!(f, "ret"),
            Bytecode::Dup => write!(f, "dup"),
            Bytecode::Discard => write!(f, "discard"),
            Bytecode::Jump(Label(x)) => write!(f, "jump\t\t{x}"),
            Bytecode::ConditionalJump(Label(x), Label(y)) => write!(f, "branch\t\t{x}, {y}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AluOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Lte,
    Gte,
    Lt,
    Gt,
    Equals,
    NotEqual,
    And,
    Or,
}

#[derive(Default)]
struct Generator {
    identifiers: Vec<String>,
    constants: Vec<ast::Constant>,
    functions: Vec<(ast::Name, Label)>,
    current_block_id: usize,
    blocks: Vec<BasicBlock>,
    locals: Vec<ast::Name>,
    builtins: Vec<BuiltinFunction>,
}

impl Generator {
    fn into_executable(self) -> Executable {
        Executable {
            functions: self
                .functions
                .iter()
                .map(|(name, target)| Function {
                    name: name.clone(),
                    target: *target,
                })
                .collect(),
            builtins: self.builtins,
            constants: self.constants.into_iter().map(Into::into).collect(),
            entry_point_block_id: self.current_block_id,
            blocks: self.blocks,
        }
    }

    fn emit_function(&mut self, name: &ast::Name) -> Label {
        let label = Label(self.make_block());
        self.functions.push((name.clone(), label));

        label
    }

    fn lookup_function_index(&self, name: &ast::Select) -> Option<usize> {
        name.as_value()
            .and_then(|name| self.functions.iter().position(|(nm, _)| nm == name))
    }

    fn lookup_builtin_index(&self, name: &ast::Select) -> Option<usize> {
        name.as_value().and_then(|name| {
            self.builtins
                .iter()
                .position(|builtin| &builtin.name == name)
        })
    }

    fn make_block(&mut self) -> u16 {
        self.blocks.push(BasicBlock::default());
        (self.blocks.len() - 1) as u16
    }

    fn intern_constant(&mut self, c: ast::Constant) -> usize {
        if let Some(index) = self.constants.iter().position(|x| *x == c) {
            index
        } else {
            self.constants.push(c);
            self.constants.len() - 1
        }
    }

    fn intern_identifier(&mut self, id: String) -> usize {
        if let Some(index) = self.identifiers.iter().position(|x| x == &id) {
            index
        } else {
            self.identifiers.push(id);
            self.identifiers.len() - 1
        }
    }

    fn emit(&mut self, b: Bytecode) {
        self.blocks[self.current_block_id].instructions.push(b);
    }

    fn unsafe_set_current_target(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            self.current_block_id = block_id
        } else {
            panic!("Block id out of bounds")
        }
    }

    fn current_block(&self) -> &BasicBlock {
        &self.blocks[self.current_block_id]
    }

    fn lookup_block(&self, Label(block_id): &Label) -> &BasicBlock {
        &self.blocks[*block_id as usize]
    }

    fn allocate_local_slot(&mut self, symbol: ast::Name) -> usize {
        self.locals.push(symbol);
        self.locals.len() - 1
    }

    fn resolve_local_slot(&self, symbol: &ast::Name) -> Option<usize> {
        self.locals.iter().position(|x| x == symbol)
    }
}

#[derive(Default)]
pub struct Compile(Generator);

impl Compile {
    fn link(self) -> Executable {
        self.0.into_executable()
    }

    fn register_builtin(&mut self, builtin: BuiltinFunction) {
        self.0.builtins.push(builtin)
    }

    fn program(&mut self, program: ast::Program) {
        for decl in program.declarations {
            self.declaration(decl)
        }

        let entry_point = self.make_label();
        self.select_target(entry_point);

        self.block(program.entry_point);

        if !self.0.current_block().is_terminated() {
            println!("Adding a return to block {}", self.0.current_block_id);
            self.0.emit(Bytecode::LoadConstant(Value::Unit));
            self.0.emit(Bytecode::Return)
        }
    }

    fn declaration(&mut self, declaration: ast::Declaration) {
        match declaration {
            ast::Declaration::Function(declarator) => {
                let target = self.0.emit_function(&declarator.name);
                self.select_target(target);

                for arg in declarator.parameters.into_iter() {
                    // this is probably where a type check needs to happen
                    // Can StoreLocal be made aware of types?
                    // Or just something like: Bytecode::TypeCheck(ast::Type) to verify that
                    // the top of the stack is what I expect.
                    let slot = self.0.allocate_local_slot(arg.name);
                    self.0.emit(Bytecode::StoreLocal(slot as u8))
                }

                self.block(declarator.body);

                println!("Current block: {}", self.0.current_block_id);

                if !self.0.current_block().is_terminated() {
                    println!("Adding a return to block {}", self.0.current_block_id);
                    self.0.emit(Bytecode::LoadConstant(Value::Unit));
                    self.0.emit(Bytecode::Return)
                }
            }
            _otherwise => todo!(),
        }
    }

    fn block(&mut self, block: ast::Block) {
        for stmt in block.statements {
            self.statement(stmt)
        }
    }

    fn make_label(&mut self) -> Label {
        Label(self.0.make_block())
    }

    fn select_target(&mut self, Label(block_id): Label) {
        self.0.unsafe_set_current_target(block_id as usize)
    }

    fn statement(&mut self, statement: ast::Statement) {
        match statement {
            ast::Statement::Let { lhs, rhs } => {
                let index = self.0.allocate_local_slot(ast::Name::simple(&lhs));
                self.expression(rhs);
                self.0.emit(Bytecode::StoreLocal(index as u8))
            }
            ast::Statement::If {
                predicate,
                consequent,
                alternate,
            } => {
                let consequent_target = self.make_label();
                let alternate_taret = self.make_label();

                // This creates an orphan block when either or both if blocks ends in a return.
                let end_target = self.make_label();

                self.expression(predicate);
                self.0.emit(Bytecode::ConditionalJump(
                    consequent_target,
                    alternate_taret,
                ));

                self.select_target(consequent_target);
                self.block(consequent);
                if !self.0.current_block().is_terminated() {
                    self.0.emit(Bytecode::Jump(end_target));
                }

                self.select_target(alternate_taret);
                self.block(alternate);
                if !self.0.current_block().is_terminated() {
                    self.0.emit(Bytecode::Jump(end_target));
                }

                self.select_target(end_target)
                // The when_true and when_false blocks both need this block
                // identifiers as its parent scope, somehow.
            }
            ast::Statement::While { predicate, body } => {
                let eval_predicate = self.make_label();
                let body_target = self.make_label();
                let end_target = self.make_label();

                self.0.emit(Bytecode::Jump(eval_predicate));
                self.select_target(eval_predicate);

                self.expression(predicate);
                self.0
                    .emit(Bytecode::ConditionalJump(body_target, end_target));

                self.select_target(body_target);
                self.block(body);
                self.0.emit(Bytecode::Jump(eval_predicate));

                self.select_target(end_target);
            }
            ast::Statement::Expression(expression) => {
                self.expression(expression);
                self.0.emit(Bytecode::Discard)
            }
            ast::Statement::Return(expression) => {
                self.expression(expression);
                self.0.emit(Bytecode::Return)
            }
        }
    }

    fn expression(&mut self, expression: ast::Expression) {
        let Compile(gen) = self;
        match expression {
            ast::Expression::Literal(constant) => gen.emit(Bytecode::LoadConstant(constant.into())),
            ast::Expression::Variable(name) => {
                // This is getting to be a pain. What value does this add?
                let slot = gen
                    .resolve_local_slot(&name.as_value().expect("type selector used in variable"))
                    .expect(&format!("Unresolved symbol {name:?}"))
                    as u8;
                gen.emit(Bytecode::LoadLocal(slot))
            }
            ast::Expression::ApplyInfix { lhs, symbol, rhs } => {
                self.expression(*lhs);
                self.expression(*rhs);
                self.operator(symbol)
            }
            ast::Expression::Apply { symbol, arguments } => {
                // Where does it validate formals versus actuals?
                for arg in arguments.into_iter().rev() {
                    self.expression(arg)
                }

                if let Some(index) = self.0.lookup_function_index(&symbol) {
                    self.0.emit(Bytecode::Invoke(index as u16))
                } else if let Some(index) = self.0.lookup_builtin_index(&symbol) {
                    self.0.emit(Bytecode::InvokeBuiltin(index as u16))
                } else {
                    panic!("Undefined symbol: {symbol:?}")
                }
            }
        }
    }

    fn operator(&mut self, operator: ast::Operator) {
        let bytecode = match operator {
            ast::Operator::Plus => Bytecode::Arithmetic(AluOp::Add),
            ast::Operator::Minus => Bytecode::Arithmetic(AluOp::Subtract),
            ast::Operator::Times => Bytecode::Arithmetic(AluOp::Multiply),
            ast::Operator::Divides => Bytecode::Arithmetic(AluOp::Divide),
            ast::Operator::Modulo => Bytecode::Arithmetic(AluOp::Modulo),
            ast::Operator::LT => Bytecode::Arithmetic(AluOp::Lt),
            ast::Operator::LTE => Bytecode::Arithmetic(AluOp::Lte),
            ast::Operator::GT => Bytecode::Arithmetic(AluOp::Gt),
            ast::Operator::GTE => Bytecode::Arithmetic(AluOp::Gte),
            ast::Operator::Equals => Bytecode::Arithmetic(AluOp::Equals),
            ast::Operator::NotEqual => Bytecode::Arithmetic(AluOp::NotEqual),
            ast::Operator::And => Bytecode::Arithmetic(AluOp::And),
            ast::Operator::Or => Bytecode::Arithmetic(AluOp::Or),
        };

        self.0.emit(bytecode)
    }
}

#[derive(Clone, Copy, Debug)]
struct Label(u16);

// Do they have to end in jump or return?
#[derive(Clone, Debug, Default)]
struct BasicBlock {
    // Should it really be this way? What is the point?
    instructions: Vec<Bytecode>,
}

impl BasicBlock {
    // What if there's a parse failure here? What does that mean?
    fn instruction_stream(&self) -> &[Bytecode] {
        &self.instructions
    }

    // Ends in Return or Jump.
    fn is_terminated(&self) -> bool {
        self.instructions
            .last()
            .is_some_and(|x| x.is_block_terminator())
    }
}
