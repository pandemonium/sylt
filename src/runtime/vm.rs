// Stfu for a second, k?
#![allow(dead_code)]
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
    entry_point: BasicBlock,
    blocks: Vec<BasicBlock>,
}

#[derive(Debug)]
struct Function {
    prototype: ast::FunctionDeclarator,
    body: BasicBlock,
}

#[derive(Debug, Default)]
struct ActivationFrame {
    locals: Vec<Value>,
    stack: Vec<Value>,
    return_value: Option<Value>,
}

#[derive(Debug)]
enum Value {
    Int(i64),
    Float(f64),
    Boolean(bool),
    Text(Box<str>), // Is this going to be a pain in the ass?
}

// Am I going to be working with registers or a stack?
enum Bytecode {
    // Is there a string table too or just one for constants?
    LoadConstant(u8),

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

enum AluOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}

enum LogicOp {
    And,
    Or,
    Equals,
    NotEqual,
}

#[derive(Default)]
struct Generator {
    identifiers: Vec<String>,
    constants: Vec<ast::Constant>,
    functions: Vec<(ast::Name, Label)>,
    current_block_id: usize,
    blocks: Vec<BasicBlock>,
}

impl Generator {
    fn into_executable(self) -> Executable {
        Executable {
            constants: self.constants.into_iter().map(|x| todo!()).collect(),
            functions: self.functions.into_iter().map(|x| todo!()).collect(),
            entry_point: self.blocks[self.current_block_id].clone(),
            blocks: self.blocks,
        }    
    }

    fn emit_function(&mut self, name: &ast::Name) -> Label {
        let label = Label(self.make_block());
        self.functions.push((name.clone(), label));

        label
    }

    fn lookup_function_index(&self, name: &ast::Select) -> Option<usize> {
        name.as_function()
            .and_then(|name| self.functions.iter().position(|(nm, _)| nm == name))
    }

    fn make_block(&mut self) -> usize {
        self.blocks.push(BasicBlock::default());
        self.blocks.len() - 1
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

    fn emit(&self, b: Bytecode) {
        todo!()
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
        self.block(program.entry_point)
    }

    fn declaration(&mut self, declaration: ast::Declaration) {
        match declaration {
            ast::Declaration::Function(declarator) => {
                let target = self.0.emit_function(&declarator.name);
                self.select_target(target);

                for (index, ..) in declarator.parameters.into_iter().rev().enumerate() {
                    // this is probably where a type check needs to happen
                    // Can StoreLocal be made aware of types?
                    // Or just something like: Bytecode::TypeCheck(ast::Type) to verify that
                    // the top of the stack is what I expect.
                    self.0.emit(Bytecode::StoreLocal(index as u8))
                }

                self.block(declarator.body);

                if !self.0.current_block().is_terminated() {
                    // Is it this simple? What is the return value?
                    // Also: how is it determined that a block is terminated?
                    // Just being terminated is not enough, it has to specifically
                    // not have a path that does not end in a return. How is that
                    // even verified?
                    self.0.emit(Bytecode::Return)
                }
            }
            ast::Declaration::IntrinsicFunction(_) => todo!(),
            ast::Declaration::Operator(_) => todo!(),
            ast::Declaration::Static { name, type_, value } => todo!(),
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
        self.0.unsafe_set_current_target(block_id)
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
                let end_target = self.make_label();

                self.expression(predicate);
                self.0.emit(Bytecode::ConditionalJump(
                    consequent_target,
                    alternate_taret,
                ));

                self.select_target(consequent_target);
                self.block(consequent);
                self.0.emit(Bytecode::Jump(end_target));

                self.select_target(alternate_taret);
                self.block(alternate);
                self.0.emit(Bytecode::Jump(end_target));

                self.select_target(end_target)
                // The when_true and when_false blocks both need this block
                // identifiers as its parent scope, somehow.
            }
            ast::Statement::While { predicate, body } => {
                let while_eval_predicate = self.make_label();
                let while_body = self.make_label();
                let end_target = self.make_label();

                self.0.emit(Bytecode::Jump(while_eval_predicate));
                self.select_target(while_eval_predicate);

                self.expression(predicate);
                self.0
                    .emit(Bytecode::ConditionalJump(while_body, end_target));

                self.select_target(while_body);
                self.block(body);
                self.0.emit(Bytecode::Jump(while_eval_predicate));

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
            ast::Expression::Literal(constant) => {
                let index = gen.intern_constant(constant);
                gen.emit(Bytecode::LoadConstant(index as u8)) // Is this a problem?
            }
            ast::Expression::Variable(_) => todo!(), // Variable => index
            ast::Expression::ApplyInfix { lhs, symbol, rhs } => {
                self.expression(*lhs);
                self.expression(*rhs);
                self.operator(symbol)
            }
            ast::Expression::Apply { symbol, arguments } => {
                // Where does it validate formals versus actuals?
                for arg in arguments {
                    self.expression(arg);
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
        match operator {
            ast::Operator::Plus => self.0.emit(Bytecode::Arithmetic(AluOp::Add)),
            ast::Operator::Minus => todo!(),
            ast::Operator::Times => todo!(),
            ast::Operator::Divides => todo!(),
            ast::Operator::Modulo => todo!(),
            ast::Operator::LT => todo!(),
            ast::Operator::LTE => todo!(),
            ast::Operator::GT => todo!(),
            ast::Operator::GTE => todo!(),
            ast::Operator::Equals => todo!(),
            ast::Operator::NotEqual => todo!(),
            ast::Operator::And => todo!(),
            ast::Operator::Or => todo!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Label(usize);

// Do they have to end in jump or return?
#[derive(Clone, Debug, Default)]
struct BasicBlock {
    // Should it really be this way?
    instructions: Vec<u8>,
}

impl BasicBlock {
    // What if there's a parse failure here? What does that mean?
    fn instruction_stream(&self) -> Vec<Bytecode> {
        todo!()
    }

    // Ends in Return or Jump.
    fn is_terminated(&self) -> bool {
        todo!()
    }
}
