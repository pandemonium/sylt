// Stfu for a second, k?
#![allow(dead_code)]
use super::intrinsics::artithmetic;
use crate::ast;
use core::fmt;
use std::cell;

static mut INTERPRETED_BYTECODE_COUNT: usize = 0;

pub struct Interpreter {
    executable: Executable,
    stack: cell::RefCell<Vec<Value>>,
}

impl Interpreter {
    pub fn new(executable: Executable) -> Self {
        Self {
            executable,
            stack: cell::RefCell::new(vec![]),
        }
    }

    pub fn run(self) -> Value {
        let return_value = self.run_automat(self.executable.entry_point());

        let count = unsafe { INTERPRETED_BYTECODE_COUNT };
        println!("Interpreted {count} bytecodes.");

        return_value
    }

    fn run_automat(&self, start: Label) -> Value {
        let mut frame = ActivationFrame::default();
        frame.continue_at(start);

        loop {
            let block = match frame.continuation {
                Continuation::Return(return_value) => break return_value,
                Continuation::Resume(Label(block_id)) => self.executable.resolve_block(block_id),
            };

            for bytecode in block.instruction_stream() {
                unsafe {
                    INTERPRETED_BYTECODE_COUNT += 1;
                }
                self.interpret(&mut frame, bytecode)
            }
        }
    }

    fn push(&self, val: Value) {
        self.stack.borrow_mut().push(val)
    }

    fn pop(&self) -> Option<Value> {
        self.stack.borrow_mut().pop()
    }

    fn duplicate_top_of_stack(&self) {
        let mut stack = self.stack.borrow_mut();
        let index = stack.len() - 1;
        let element = stack[index].clone();
        stack.push(element);
    }

    fn interpret(&self, frame: &mut ActivationFrame, bytecode: &Bytecode) {
        match bytecode {
            Bytecode::LoadConstant(constant) => self.push(constant.clone()),
            Bytecode::LoadLocal(index) => self.push(frame.get_local(*index).clone()),
            Bytecode::StoreLocal(index) => frame.put_local(
                *index,
                self.pop()
                    .expect(&format!("Value on the stack to put in local {index}")),
            ),
            Bytecode::Arithmetic(op) => {
                // make a thing that will pop a specific type.
                // pop the first, require the second to have the same type
                
                let (rhs, lhs) = self
                    .pop()
                    .zip(self.pop())
                    .expect("Two values on the stack to compute {op}");
                self.push(lhs.apply(op, rhs))
            }
            Bytecode::Logic(_) => todo!(),
            Bytecode::Invoke(index) => {
                let target = &self.executable.resolve_function_target(*index).target;
                let return_value = self.run_automat(*target);
                self.push(return_value)
            }
            Bytecode::Return => {
                let return_value = self.pop().expect("no value to return on the stack");
                frame.make_return(return_value)
            }
            Bytecode::Dup => self.duplicate_top_of_stack(),
            Bytecode::Discard => {
                self.pop().expect("Discarding a non-existent top of stack");
            }
            Bytecode::Jump(target) => frame.continue_at(*target),
            Bytecode::ConditionalJump(consequent, alternate) => {
                let top = self
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
    compile.program(program);
    compile.link()
}

#[derive(Debug, Default)]
pub struct Executable {
    constants: Vec<Value>,
    functions: Vec<Function>,
    entry_point_block_id: usize,
    blocks: Vec<BasicBlock>,
}

impl Executable {
    fn resolve_function_target(&self, index: u16) -> &Function {
        &self.functions[index as usize]
    }

    fn entry_point(&self) -> Label {
        Label(self.entry_point_block_id as u16)
    }

    fn resolve_block(&self, id: u16) -> &BasicBlock {
        &self.blocks[id as usize]
    }
}

impl fmt::Display for Executable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Executable {
            constants,
            functions,
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

#[derive(Debug, Default)]
struct ActivationFrame {
    locals: Vec<Option<Value>>,
    continuation: Continuation,
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
    use Value::*;
    use AluOp::*;

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
        // "Call out" into the AST interpreter
        //Re-write this.
//                artithmetic::operator::apply(&self.into(), &op.into(), &rhs.into())
//                    .expect(&format!("Undefined operator sequence"))
//                    .into();

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
    Logic(LogicOp),

    // InvokeBuiltin(u16)   -- e.g.: println
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
            Bytecode::Logic(LogicOp::And) => write!(f, "and"),
            Bytecode::Logic(LogicOp::Or) => write!(f, "or"),
            Bytecode::Invoke(x) => write!(f, "invoke\t\t{x}"),
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
}

#[derive(Debug, Clone, Copy)]
enum LogicOp {
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

    fn program(&mut self, program: ast::Program) {
        for decl in program.declarations {
            self.declaration(decl)
        }

        let entry_point = self.make_label();
        self.select_target(entry_point);

        self.block(program.entry_point)
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
                    // Is it this simple? What is the return value?
                    // Also: how is it determined that a block is terminated?
                    // Just being terminated is not enough, it has to specifically
                    // not have a path that does not end in a return. How is that
                    // even verified?
                    // Push a void first?
                    println!("Adding a return to block {}", self.0.current_block_id);
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
            ast::Operator::And => Bytecode::Logic(LogicOp::And),
            ast::Operator::Or => Bytecode::Logic(LogicOp::Or),
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
