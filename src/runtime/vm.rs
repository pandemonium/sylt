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

fn compile(program: ast::Program) -> Executable {
    todo!()
}

struct Executable {
    constants: Vec<Value>,
    functions: Vec<Function>,
    builtins: Vec<String>,
    entry_point: BasicBlock,
}

struct Function {
    body: BasicBlock,

    // Yeah?
    declaration: ast::FunctionDeclarator,
}

#[derive(Debug, Default)]
struct ActivationFrame {
    locals: Vec<Value>,
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

    // InvokeBuiltin(u16)
    Invoke(u16),
    Return,

    Jump(Label),
    ConditionalJump(Label, Label),
}

struct Generator {
    blocks: Vec<BasicBlock>,
}

impl Generator {
    
    fn emit(&self, b: Bytecode) {
        todo!()
    }
}

#[derive(Debug, Default)]
struct Label(BasicBlock);

impl Label {

}

// Do they have to end in jump or return?
#[derive(Debug, Default)]
struct BasicBlock {
    // Should it really be this way? 
    instructions: Vec<u8>,
}

impl BasicBlock {
    fn instruction_stream(&self) -> Vec<Bytecode> {
        todo!()
    }
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