use core::fmt;
use crate::intrinsics::artithmetic::operator;

#[derive(Debug)]
pub struct Program {
    pub definitions: Vec<Declaration>,
    pub entry_point: Block,
}

#[derive(Clone, Debug)]
pub enum Declaration {
    Function(FunctionDef),
    IntrinsicFunction(IntrinsicFunctionDef),
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
            Declaration::Operator(def) => def.select().name().clone(),
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

    pub fn as_intrinsic(&self) -> Option<&IntrinsicFunctionDef> {
        // Is there a more ergonomic version of this pattern?
        if let Self::IntrinsicFunction(def) = self {
            Some(def)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntrinsicFunctionDef {
    pub name: Name,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
}

impl IntrinsicFunctionDef {
    pub fn new(name: &Name, parameters: &[Parameter], return_type: Type) -> Self {
        Self {
            name: name.clone(),
            parameters: parameters.into(),
            return_type,
        }
    }

    pub fn name(&self) -> &Name {
        &self.name
    }
}

#[derive(Clone, Debug)]
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
    // Aren't these inside out?
    Literal(Constant),
    Variable(Name),
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
pub struct Name {
    pub scope_path: Vec<String>,
    pub name: String,
}

impl Name {
    pub fn qualified_name(&self) -> String {
        let mut scope_path = self.scope_path.join("::");
        scope_path.push_str(&self.name);
        scope_path
    }

    pub fn local_name(&self) -> &str {
        &self.name
    }

    pub fn simple(name: &str) -> Self {
        Self {
            scope_path: vec![],
            name: name.into(),
        }
    }

    pub fn intrinsic(context: &str, module: &str, name: &str) -> Self {
        Self {
            scope_path: vec!["builtins".into(), context.into(), module.into()],
            name: name.into(),
        }
    }

    pub fn std(module: &str, name: &str) -> Self {
        Self {
            scope_path: vec!["std".into(), module.into()],
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
    Simple(Name),
    Number,
    Unit,
}

impl Type {
    pub fn simple(name: &Name) -> Self {
        Type::Simple(name.clone())
    }

    pub fn name(&self) -> Name {
        match self {
            Self::Simple(name) => name.clone(),
            Self::Number => Name::simple("{number}"),
            Self::Unit => Name::simple("unit"),
        }
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
        fn qualified_type_name(name: &str) -> Type {
            Type::simple(&Name::simple(name))
        }

        match self {
            Constant::Boolean(..) => qualified_type_name("boolean"),
            Constant::Int(..) => qualified_type_name("int"),
            Constant::Float(..) => qualified_type_name("float"),
            Constant::Text(..) => qualified_type_name("text"),
            Constant::Void => qualified_type_name("unit"),
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
}

impl Operator {
    pub fn local_name(&self) -> String {
        match self {
            Operator::Plus => "plus".into(),
            Operator::Minus => "minus".into(),
            Operator::Times => "times".into(),
            Operator::Divides => "divides".into(),
            Operator::Modulo => "modulo".into(),
        }
    }

    pub fn select(&self) -> Select {
        Select::Intrinsic(operator::qualified_name(&self.local_name()))
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
            // boolean algebra
            // comparisons
        ]
        .into_iter()
        .find(|op| op.local_name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interpreter, intrinsics::io};

    #[test]
    fn show_hello_world_program() {
        let ast = Program {
            definitions: vec![Declaration::Function(FunctionDef {
                name: Name::simple("print_hello_world"),
                parameters: vec![],
                return_type: Type::Unit,
                body: Block {
                    statements: vec![
                        Statement::Expression(Expression::Apply {
                        symbol: Select::Intrinsic(Name::std("io", "print_line")),
                        arguments: vec![Expression::Literal(Constant::Text("Hello, world".into()))],
                        }),
                        Statement::Return(Expression::Literal(Constant::Int(1)))
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
