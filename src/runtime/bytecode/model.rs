use crate::ast::{self, PrimitiveType};
use std::{cell, fmt, rc};

#[derive(Clone, Debug, Default)]
pub enum Value {
    Int(i64),
    Float(f64),
    Boolean(bool),
    Text(Box<str>),
    #[default]
    Unit,
    Array(cell::RefCell<ValueArray>),
    //    Reference(ast::Type, usize),
}

#[derive(Clone, Debug)]
pub enum ValueArray {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Boolean(Vec<bool>),
    Text(Vec<Box<str>>),
}

fn comma_separate<A: fmt::Display>(xs: &[A]) -> String {
    xs.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

impl fmt::Display for ValueArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueArray::Int(xs) => write!(f, "[{}]", comma_separate(xs)),
            ValueArray::Float(xs) => write!(f, "[{}]", comma_separate(xs)),
            ValueArray::Boolean(xs) => write!(f, "[{}]", comma_separate(xs)),
            ValueArray::Text(xs) => write!(f, "[{}]", comma_separate(xs)),
        }
    }
}

impl ValueArray {
    pub fn unsafe_put_element(&mut self, index: usize, new: Value) {
        match (self, new) {
            (ValueArray::Int(array), Value::Int(element)) => array[index] = element,
            (ValueArray::Float(array), Value::Float(element)) => array[index] = element,
            (ValueArray::Boolean(array), Value::Boolean(element)) => array[index] = element,
            (ValueArray::Text(array), Value::Text(element)) => array[index] = element,
            (array, element) => panic!("Must not put {element} in {array}"),
        }
    }

    pub fn get_element(&self, index: usize) -> Option<Value> {
        match self {
            ValueArray::Int(array) => array.get(index).copied().map(Value::Int),
            ValueArray::Float(array) => array.get(index).copied().map(Value::Float),
            ValueArray::Boolean(array) => array.get(index).copied().map(Value::Boolean),
            ValueArray::Text(array) => array.get(index).cloned().map(Value::Text),
        }
    }
}

// This goes in vm.rs. Or perhaps intrinsics.rs?
pub fn try_compute(lhs: &Value, op: &AluOp, rhs: &Value) -> Option<Value> {
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

        (Boolean(lhs), And, Boolean(rhs)) => Some(Boolean(*lhs && *rhs)),
        (Boolean(lhs), Or, Boolean(rhs)) => Some(Boolean(*lhs || *rhs)),
        _otherwise => None,
    }
}

impl Value {
    pub fn apply(self, op: &AluOp, rhs: Self) -> Self {
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

impl From<ast::Operator> for AluOp {
    fn from(value: ast::Operator) -> Self {
        match value {
            ast::Operator::Plus => AluOp::Add,
            ast::Operator::Minus => AluOp::Subtract,
            ast::Operator::Times => AluOp::Multiply,
            ast::Operator::Divides => AluOp::Divide,
            ast::Operator::Modulo => AluOp::Modulo,
            ast::Operator::LT => AluOp::Lt,
            ast::Operator::LTE => AluOp::Lte,
            ast::Operator::GT => AluOp::Gt,
            ast::Operator::GTE => AluOp::Gte,
            ast::Operator::Equals => AluOp::Equals,
            ast::Operator::NotEqual => AluOp::NotEqual,
            ast::Operator::And => AluOp::And,
            ast::Operator::Or => AluOp::Or,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "#{x}"),
            Self::Float(x) => write!(f, "#{x}"),
            Self::Boolean(x) => write!(f, "{}", if *x { "True" } else { "False" }),
            Self::Text(x) => write!(f, "'{x}'"),
            Self::Unit => write!(f, "()"),
            Self::Array(contents) => write!(f, "{}", contents.borrow()),
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
            ast::Constant::Array(array) => {
                // Is this the most efficient way to do this? Does it matter?
                Value::Array(cell::RefCell::new(array.into_inner().into()))
            }
        }
    }
}

impl From<ast::ArrayConstant> for ValueArray {
    fn from(value: ast::ArrayConstant) -> Self {
        match value {
            ast::ArrayConstant::Int(array) => ValueArray::Int(array),
            ast::ArrayConstant::Float(array) => ValueArray::Float(array),
            ast::ArrayConstant::Boolean(array) => ValueArray::Boolean(array),
            ast::ArrayConstant::Text(array) => {
                ValueArray::Text(array.into_iter().map(Into::into).collect())
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Label(pub u16);

#[derive(Debug, Clone)]
pub enum Bytecode {
    LoadConstant(Value), // Push constant onto stack

    LoadLocal(u8),  // Push local onto stack
    StoreLocal(u8), // Pop into local

    Array(ArrayOp),
    Arithmetic(AluOp),

    InvokeBuiltin(u16, u8),
    Invoke(u16),
    Return,
    Dup,
    Discard,

    Jump(Label),
    ConditionalJump(Label, Label),
}

#[derive(Debug, Clone)]
pub enum ArrayOp {
    New(ast::PrimitiveType), // Allocate array; which type?
    Load,                    // Push array element onto stack
    Store,                   // Pop stack into array element
}

impl Bytecode {
    fn is_block_terminator(&self) -> bool {
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
            Bytecode::Array(ArrayOp::New(PrimitiveType::Int)) => write!(f, "int_array"),
            Bytecode::Array(ArrayOp::New(PrimitiveType::Float)) => write!(f, "float_array"),
            Bytecode::Array(ArrayOp::New(PrimitiveType::Boolean)) => write!(f, "boolean_array"),
            Bytecode::Array(ArrayOp::New(PrimitiveType::Text)) => write!(f, "text_array"),
            Bytecode::Array(ArrayOp::New(PrimitiveType::Unit)) => write!(f, "unit_array"),
            Bytecode::Array(ArrayOp::New(PrimitiveType::Array)) => write!(f, "multi_array"), // HMM. Do I support this?
            Bytecode::Array(ArrayOp::Load) => write!(f, "array_load"),
            Bytecode::Array(ArrayOp::Store) => write!(f, "array_store"),
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
            Bytecode::InvokeBuiltin(x, y) => write!(f, "invoke_builtin\t{x}, {y}"),
            Bytecode::Return => write!(f, "ret"),
            Bytecode::Dup => write!(f, "dup"),
            Bytecode::Discard => write!(f, "discard"),
            Bytecode::Jump(Label(x)) => write!(f, "jump\t\t{x}"),
            Bytecode::ConditionalJump(Label(x), Label(y)) => write!(f, "branch\t\t{x}, {y}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AluOp {
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

#[derive(Clone, Debug, Default)]
pub struct BasicBlock {
    pub instructions: Vec<Bytecode>,
}

impl BasicBlock {
    pub fn instruction_stream(&self) -> &[Bytecode] {
        &self.instructions
    }

    pub fn is_terminated(&self) -> bool {
        self.instructions
            .last()
            .is_some_and(|x| x.is_block_terminator())
    }
}

#[derive(Debug)]
pub struct Function {
    pub name: ast::Name,
    pub target: Label,
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
pub struct BuiltinFunction {
    pub name: ast::Name,
    pub prototype: BuiltinFunctionPrototype,
    pub stub: rc::Rc<dyn BuiltinStub>,
}

#[derive(Debug)]
pub enum BuiltinFunctionPrototype {
    Varargs,
    Literally(Vec<ast::Type>),
}

impl fmt::Display for BuiltinFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.prototype {
            BuiltinFunctionPrototype::Varargs => write!(f, "{}(...)", self.name),
            BuiltinFunctionPrototype::Literally(prototype) => {
                let parameter_list = prototype
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                write!(f, "{}({parameter_list})", self.name)
            }
        }
    }
}

pub trait BuiltinStub: fmt::Debug {
    fn call(&self, parameters: &[Value]) -> Option<Value>;
}
