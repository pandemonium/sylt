use crate::runtime::{self, ast::intrinsics};
use core::fmt;
use std::{cell, rc};

#[derive(Clone, Debug)]
pub struct Program {
    pub declarations: Vec<Declaration>,
    pub entry_point: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Declaration {
    Function(FunctionDeclarator),
    IntrinsicFunction(IntrinsicFunctionDeclarator),
    Operator(Operator),
    Static {
        name: Name,
        type_: Name,
        value: Constant,
    },
}

impl Declaration {
    pub fn name(&self) -> Name {
        match self {
            Declaration::Function(def) => def.name().clone(),
            Declaration::IntrinsicFunction(def) => def.name().clone(),
            Declaration::Operator(def) => def.name(),
            Declaration::Static { name, .. } => name.clone(),
        }
    }

    pub fn as_function(&self) -> Option<&FunctionDeclarator> {
        if let Self::Function(def) = self {
            Some(def)
        } else {
            None
        }
    }

    pub fn as_intrinsic(&self) -> Option<&IntrinsicFunctionDeclarator> {
        // Is there a more ergonomic version of this pattern?
        if let Self::IntrinsicFunction(def) = self {
            Some(def)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntrinsicFunctionDeclarator {
    pub name: Name,
    pub parameters: Vec<Parameter>,
    pub return_type: Select,

    // This precludes PartialEq - it has to go somewhere else
    pub dispatch: rc::Rc<dyn intrinsics::IntrinsicProxy>,
}

impl PartialEq for IntrinsicFunctionDeclarator {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.parameters == other.parameters
            && self.return_type == other.return_type
        //            && self.call_proxy == other.call_proxy
    }
}

impl IntrinsicFunctionDeclarator {
    pub fn new<F>(name: &Name, parameters: &[Parameter], return_type: Select, target: F) -> Self
    where
        F: intrinsics::IntrinsicProxy + 'static,
    {
        Self {
            name: name.clone(),
            parameters: parameters.into(),
            return_type,
            dispatch: rc::Rc::new(target),
        }
    }

    pub fn name(&self) -> &Name {
        &self.name
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDeclarator {
    pub name: Name,
    pub parameters: Vec<Parameter>,
    pub return_type: Select,
    pub body: Block,
}

impl FunctionDeclarator {
    pub fn name(&self) -> &Name {
        &self.name
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: Name,
    pub type_: Select,
}

impl Parameter {
    pub fn get_type(&self) -> &Select {
        &self.type_
    }

    pub fn get_name(&self) -> &str {
        self.name.local_name()
    }
}

// Deconstruct these into
// Statement::Let(LetStatement) so that I can do things
// with separate types later?
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Let {
        lhs: String,
        rhs: Expression,
    },
    If {
        predicate: Expression,
        consequent: Block,
        alternate: Block,
    },
    While {
        predicate: Expression,
        body: Block,
    },
    Expression(Expression),
    Return(Expression),
    ArrayUpdate {
        array: Box<Expression>,
        subscript: Box<Expression>,
        rhs: Box<Expression>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Literal(Constant),
    Variable(Select),
    GetArrayElement {
        array: Box<Expression>,
        subscript: Box<Expression>,
    },
    PutArrayElement {
        array: Box<Expression>,
        subscript: Box<Expression>,
        element: Box<Expression>,
    },
    ApplyInfix {
        lhs: Box<Expression>,
        symbol: Operator,
        rhs: Box<Expression>,
    },
    Apply {
        symbol: Select,
        arguments: Vec<Expression>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Module {
    User,
    Intrinsic,
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Name {
    pub module: Module,
    pub name: String,
}

impl Name {
    pub fn qualified_name(&self) -> String {
        match self.module {
            Module::User => self.name.clone(),
            Module::Intrinsic => format!("builtin::{}", self.name),
        }
    }

    pub fn local_name(&self) -> &str {
        &self.name
    }

    pub fn simple(name: &str) -> Self {
        Self {
            module: Module::User,
            name: name.into(),
        }
    }

    pub fn intrinsic(name: &str) -> Self {
        Self {
            module: Module::Intrinsic,
            name: name.into(),
        }
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Name { module, name } = self;
        write!(f, "{module}::{name}",)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Select {
    Type(TypeSelector),
    Value(Name),
}

const NUMBER_STEREOTYPE_NAME: &str = "<<number>>";

#[derive(Clone, Debug, PartialEq)]
pub enum TypeSelector {
    Named(Name),
    Stereotype(Name),
}

impl Select {
    pub fn name(&self) -> &Name {
        match self {
            Self::Type(TypeSelector::Named(name)) => name,
            Self::Type(TypeSelector::Stereotype(name)) => name,
            Self::Value(name) => name,
        }
    }

    pub fn primitive_type(tpe: PrimitiveType) -> Select {
        Select::Type(TypeSelector::Named(tpe.name()))
    }

    pub fn as_value(&self) -> Option<&Name> {
        if let Self::Value(name) = self {
            Some(name)
        } else {
            None
        }
    }

    // This function does not spark joy.
    pub fn is_stereotype_for(&self, scrutinee: &Type) -> bool {
        if let Self::Type(TypeSelector::Stereotype(stereotype)) = self {
            matches!(
                scrutinee,
                Type::Primitive(PrimitiveType::Int | PrimitiveType::Float)
                    if stereotype.name == NUMBER_STEREOTYPE_NAME
            )
        } else {
            false
        }
    }

    pub fn subsumes(&self, scrutinee: &Type) -> bool {
        self.name() == &scrutinee.name() || self.is_stereotype_for(scrutinee)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Named(Name),
    Primitive(PrimitiveType),
}

impl Type {
    pub fn named(name: &Name) -> Self {
        Type::Named(name.clone())
    }

    pub fn name(&self) -> Name {
        match self {
            Self::Named(name) => name.clone(),
            Self::Primitive(tpe) => tpe.name(),
        }
    }

    pub fn subsumes(&self, rhs: &Type) -> bool {
        *self == *rhs
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Named(name) => write!(f, "{name}"),
            Self::Primitive(tpe) => write!(f, "{tpe}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    Boolean,
    Int,
    Float,
    Text,
    Unit,
    Array,
}

impl PrimitiveType {
    pub fn try_from_name(name: &str) -> Option<PrimitiveType> {
        match name {
            "Boolean" => Some(Self::Boolean),
            "Int" => Some(Self::Int),
            "Float" => Some(Self::Float),
            "Text" => Some(Self::Text),
            "Unit" => Some(Self::Unit),
            "Array" => Some(Self::Array),
            _otherwise => None,
        }
    }

    pub fn name(&self) -> Name {
        match self {
            Self::Boolean => Name::intrinsic("Boolean"),
            Self::Int => Name::intrinsic("Int"),
            Self::Float => Name::intrinsic("Float"),
            Self::Text => Name::intrinsic("Text"),
            Self::Unit => Name::intrinsic("Unit"),
            Self::Array => Name::intrinsic("Array"),
        }
    }
}

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Boolean(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Void,
    Array(cell::RefCell<ArrayConstant>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ArrayConstant {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Boolean(Vec<bool>),
    Text(Vec<String>),
}

use runtime::ast::interpreter::Error as RuntimeError;
impl ArrayConstant {
    pub fn new(template: Constant, size: usize) -> Self {
        match template {
            Constant::Boolean(template) => Self::Boolean(vec![template; size]),
            Constant::Int(template) => Self::Int(vec![template; size]),
            Constant::Float(template) => Self::Float(vec![template; size]),
            Constant::Text(template) => Self::Text(vec![template; size]),
            _otherwise => todo!(),
        }
    }

    pub fn length(&self) -> usize {
        match self {
            ArrayConstant::Int(array) => array.len(),
            ArrayConstant::Float(array) => array.len(),
            ArrayConstant::Boolean(array) => array.len(),
            ArrayConstant::Text(array) => array.len(),
        }
    }

    pub fn get_element(&self, index: usize) -> Option<Constant> {
        match self {
            ArrayConstant::Int(array) => array.get(index).copied().map(Constant::Int),
            ArrayConstant::Float(array) => array.get(index).copied().map(Constant::Float),
            ArrayConstant::Boolean(array) => array.get(index).copied().map(Constant::Boolean),
            ArrayConstant::Text(array) => array.get(index).cloned().map(Constant::Text),
        }
    }

    // Bad ref to runtime::ast::interpreter::Error
    // This function is badly placed.
    pub fn put_element(&mut self, index: usize, new_element: Constant) -> Result<(), RuntimeError> {
        match (self, new_element) {
            (ArrayConstant::Int(array), Constant::Int(mut new_element)) => {
                array.get_mut(index).replace(&mut new_element);
                Ok(())
            }
            (ArrayConstant::Float(array), Constant::Float(mut new_element)) => {
                array.get_mut(index).replace(&mut new_element);
                Ok(())
            }
            (ArrayConstant::Boolean(array), Constant::Boolean(mut new_element)) => {
                array.get_mut(index).replace(&mut new_element);
                Ok(())
            }
            (ArrayConstant::Text(array), Constant::Text(mut new_element)) => {
                array.get_mut(index).replace(&mut new_element);
                Ok(())
            }
            (_array, _new_element) => todo!(),
        }
    }
}

impl fmt::Display for ArrayConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayConstant::Int(array) => write!(f, "Int array of {} elements", array.len()),
            ArrayConstant::Float(array) => write!(f, "Int array of {} elements", array.len()),
            ArrayConstant::Boolean(array) => write!(f, "Int array of {} elements", array.len()),
            ArrayConstant::Text(array) => write!(f, "Int array of {} elements", array.len()),
        }
    }
}

impl Constant {
    pub fn as_boolean(&self) -> Option<&bool> {
        if let Self::Boolean(x) = self {
            Some(x)
        } else {
            None
        }
    }

    pub fn get_type(&self) -> Type {
        match self {
            Constant::Boolean(..) => Type::Primitive(PrimitiveType::Boolean),
            Constant::Int(..) => Type::Primitive(PrimitiveType::Int),
            Constant::Float(..) => Type::Primitive(PrimitiveType::Float),
            Constant::Text(..) => Type::Primitive(PrimitiveType::Text),
            Constant::Void => Type::Primitive(PrimitiveType::Unit),
            Constant::Array(..) => Type::Primitive(PrimitiveType::Array),
        }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Boolean(x) => write!(f, "{x}"),
            Constant::Int(x) => write!(f, "{x}"),
            Constant::Float(x) => write!(f, "{x}"),
            Constant::Text(x) => write!(f, "{x}"),
            Constant::Void => write!(f, "()"),
            Constant::Array(array) => write!(f, "Array of {}", array.borrow()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    Plus,
    Minus,
    Times,
    Divides,
    Modulo,

    LT,
    LTE,
    GT,
    GTE,

    Equals,
    NotEqual,

    And,
    Or,
}

impl Operator {
    pub fn name(&self) -> Name {
        match self {
            Operator::Plus => Name::intrinsic("plus"),
            Operator::Minus => Name::intrinsic("minus"),
            Operator::Times => Name::intrinsic("times"),
            Operator::Divides => Name::intrinsic("divides"),
            Operator::Modulo => Name::intrinsic("modulo"),
            Operator::LT => Name::intrinsic("less_than"),
            Operator::LTE => Name::intrinsic("less_than_or_equal"),
            Operator::GT => Name::intrinsic("greater_than"),
            Operator::GTE => Name::intrinsic("greater_than_or_equal"),
            Operator::Equals => Name::intrinsic("equals"),
            Operator::NotEqual => Name::intrinsic("not_equal"),
            Operator::And => Name::intrinsic("and"),
            Operator::Or => Name::intrinsic("or"),
        }
    }

    pub fn local_name(&self) -> String {
        self.name().name.clone()
    }

    pub fn select(&self) -> Select {
        Select::Value(self.name())
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter {
                name: Name::simple("lhs"),
                type_: Select::Type(TypeSelector::Stereotype(Name::intrinsic(
                    NUMBER_STEREOTYPE_NAME,
                ))),
            },
            Parameter {
                name: Name::simple("rhs"),
                type_: Select::Type(TypeSelector::Stereotype(Name::intrinsic(
                    NUMBER_STEREOTYPE_NAME,
                ))),
            },
        ]
    }

    pub fn return_type(&self) -> Select {
        let numeric = Select::Type(TypeSelector::Stereotype(Name::intrinsic(
            NUMBER_STEREOTYPE_NAME,
        )));
        let boolean = Select::Type(TypeSelector::Named(PrimitiveType::Boolean.name()));
        match self {
            Operator::Plus => numeric,
            Operator::Minus => numeric,
            Operator::Times => numeric,
            Operator::Divides => numeric,
            Operator::Modulo => numeric,
            Operator::LT => boolean,
            Operator::LTE => boolean,
            Operator::GT => boolean,
            Operator::GTE => boolean,
            Operator::Equals => boolean,
            Operator::NotEqual => boolean,
            Operator::And => boolean,
            Operator::Or => boolean,
        }
    }

    pub fn resolve(name: &str) -> Option<Operator> {
        vec![
            Self::Plus,
            Self::Minus,
            Self::Times,
            Self::Divides,
            Self::Modulo,
            Self::LT,
            Self::LTE,
            Self::GT,
            Self::GTE,
            Self::Equals,
            Self::NotEqual,
            Self::And,
            Self::Or,
        ]
        .into_iter()
        .find(|op| op.local_name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ast::{
        interpreter,
        intrinsics::{artithmetic::operator, io},
    };

    #[test]
    fn show_hello_world_program() {
        let ast = Program {
            declarations: vec![Declaration::Function(FunctionDeclarator {
                name: Name::simple("print_hello_world"),
                parameters: vec![],
                return_type: Select::Type(TypeSelector::Stereotype(PrimitiveType::Unit.name())),
                body: Block {
                    statements: vec![
                        Statement::Expression(Expression::Apply {
                            symbol: Select::Value(Name::intrinsic("print_line")),
                            arguments: vec![Expression::Literal(Constant::Text(
                                "Hello, world".into(),
                            ))],
                        }),
                        Statement::Return(Expression::Literal(Constant::Int(1))),
                    ],
                },
            })],
            entry_point: Block {
                statements: vec![Statement::Expression(Expression::Apply {
                    symbol: Select::Value(Name::simple("print_hello_world")),
                    arguments: vec![],
                })],
            },
        };

        println!("{ast:#?}");
        let mut builtins = operator::declarations();
        builtins.extend(io::declarations());
        let interpreter = interpreter::Interpreter::new(ast, builtins);
        let result = interpreter.run();

        //        assert_ne!(result.clone(), result);

        println!("--------> {result:#?}");
    }

    #[test]
    fn parse_hello_world_program() {
        let _program = r#"
            fn print_hello_world() -> unit {
                println("hello, world");
            }
            print_hello_world();
            "#;
    }
}
