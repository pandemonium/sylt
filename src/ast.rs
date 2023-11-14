use crate::runtime::intrinsics;
use core::fmt;
use std::rc;

#[derive(Clone, Debug)]
pub struct Program {
    pub declarations: Vec<Declaration>,
    pub entry_point: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Declaration {
    Function(FunctionDef),
    IntrinsicFunction(IntrinsicFunctionPrototype),
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

    pub fn as_function(&self) -> Option<&FunctionDef> {
        if let Self::Function(def) = self {
            Some(def)
        } else {
            None
        }
    }

    pub fn as_intrinsic(&self) -> Option<&IntrinsicFunctionPrototype> {
        // Is there a more ergonomic version of this pattern?
        if let Self::IntrinsicFunction(def) = self {
            Some(def)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntrinsicFunctionPrototype {
    pub name: Name,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,

    // This precludes PartialEq - it has to go somewhere else
    pub dispatch: rc::Rc<dyn intrinsics::IntrinsicProxy>,
}

impl PartialEq for IntrinsicFunctionPrototype {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.parameters == other.parameters
            && self.return_type == other.return_type
        //            && self.call_proxy == other.call_proxy
    }
}

impl IntrinsicFunctionPrototype {
    pub fn new<F>(name: &Name, parameters: &[Parameter], return_type: Type, target: F) -> Self
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
pub struct FunctionDef {
    pub name: Name,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
    pub body: Block,
}

impl FunctionDef {
    pub fn name(&self) -> &Name {
        &self.name
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub type_: Type,
}

impl Parameter {
    pub fn new(name: &str, type_: Type) -> Self {
        Self {
            name: name.into(),
            type_,
        }
    }

    pub fn get_type(&self) -> &Type {
        &self.type_
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Let {
        lhs: String,
        rhs: Expression,
    },
    If {
        predicate: Expression,
        when_true: Block,
        when_false: Block,
    },
    While {
        predicate: Expression,
        body: Block,
    },
    Expression(Expression),
    Return(Expression),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Literal(Constant),
    Variable(Name),
    ApplyInfix {
        // This isn't necessary, really.
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

#[derive(Clone, Debug, PartialEq)]
pub enum Select {
    Type(Name),
    Function(Name),
    Intrinsic(Name),
}

impl Select {
    pub fn name(&self) -> &Name {
        match self {
            Select::Type(name) => name,
            Select::Function(name) => name,
            Select::Intrinsic(name) => name,
        }
    }

    pub fn as_function(&self) -> Option<&Name> {
        if let Self::Function(name) = self {
            Some(name)
        } else {
            None
        }
    }

    pub fn as_intrinsic(&self) -> Option<&Name> {
        if let Self::Intrinsic(name) = self {
            Some(name)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Named(Name),
    Number,
    Unit,
}

impl Type {
    pub fn named(name: &Name) -> Self {
        Type::Named(name.clone())
    }

    pub fn name(&self) -> Name {
        match self {
            Self::Named(name) => name.clone(),
            Self::Number => Name::intrinsic("{number}"),
            Self::Unit => Name::intrinsic("Unit"),
        }
    }

    pub fn subsumes(&self, rhs: &Self) -> bool {
        // Magic numbers a-plenty
        self == rhs
            || self == &Self::Number && rhs == &Self::Named(Name::intrinsic("Int"))
            || self == &Self::Number && rhs == &Self::Named(Name::intrinsic("Float"))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Boolean(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Void,
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
            Constant::Boolean(..) => Type::named(&Name::intrinsic("Boolean")),
            Constant::Int(..) => Type::named(&Name::intrinsic("Int")),
            Constant::Float(..) => Type::named(&Name::intrinsic("Float")),
            Constant::Text(..) => Type::named(&Name::intrinsic("Text")),
            Constant::Void => Type::named(&Name::intrinsic("Unit")),
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
        Select::Intrinsic(self.name())
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("lhs", Type::Number),
            Parameter::new("rhs", Type::Number),
        ]
    }

    pub fn return_type(&self) -> Type {
        Type::Number
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
            // boolean algebra
        ]
        .into_iter()
        .find(|op| op.local_name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{
        interpreter,
        intrinsics::{artithmetic::operator, io},
    };

    #[test]
    fn show_hello_world_program() {
        let ast = Program {
            declarations: vec![Declaration::Function(FunctionDef {
                name: Name::simple("print_hello_world"),
                parameters: vec![],
                return_type: Type::Unit,
                body: Block {
                    statements: vec![
                        Statement::Expression(Expression::Apply {
                            symbol: Select::Intrinsic(Name::intrinsic("print_line")),
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
                    symbol: Select::Function(Name::simple("print_hello_world")),
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
