// Stfu for a second, k?
#![allow(dead_code)]
use core::fmt;

use crate::ast;

struct Interpreter {
    stack: Vec<Value>,
}

impl Interpreter {
    fn run(self, exe: Executable) -> Option<Value> {
        let frame = ActivationFrame::default();

        todo!()
    }
}

pub fn compile(program: ast::Program) -> Executable {
    let mut compile = Compile::default();
    compile.program(program);
    compile.link()
}

#[derive(Debug)]
pub struct Executable {
    constants: Vec<Value>,
    functions: Vec<Function>,
    entry_point_block_id: usize,
    blocks: Vec<BasicBlock>,
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
    body: BasicBlock,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Function { name, body } = self;
        writeln!(f, "{}", name.local_name())?;
        for bytecode in body.instruction_stream() {
            writeln!(f, "\t{bytecode}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct ActivationFrame {
    locals: Vec<Value>,
    stack: Vec<Value>,
    return_value: Option<Value>,
}

#[derive(Clone, Debug)]
enum Value {
    Int(i64),
    Float(f64),
    Boolean(bool),
    Text(Box<str>),
    Unit,
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
                    body: self.lookup_block(&target).clone(),
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
            //            ast::Declaration::IntrinsicFunction(_) => todo!(),
            //            ast::Declaration::Operator(_) => todo!(),
            //            ast::Declaration::Static { name, type_, value } => todo!(),
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
                // This is not correct. The first let gets 0, the second 1, etc.
                let index = self.0.intern_identifier(lhs);
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
