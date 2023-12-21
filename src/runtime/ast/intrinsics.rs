use crate::{ast, runtime::ast::interpreter};
use core::fmt;
use std::collections;

pub trait IntrinsicProxy: fmt::Debug {
    fn invoke(&self, arguments: &[ast::Constant]) -> Result<ast::Constant, interpreter::Error>;
}

impl IntrinsicProxy for ast::Operator {
    fn invoke(&self, arguments: &[ast::Constant]) -> Result<ast::Constant, interpreter::Error> {
        if let [lhs, rhs] = arguments {
            artithmetic::operator::apply(lhs, &self, rhs).ok_or_else(|| {
                interpreter::Error::ExpectedArguments(self.name().clone(), self.parameters())
            })
        } else {
            Err(interpreter::Error::ExpectedArguments(
                self.name().clone(),
                self.parameters(),
            ))
        }
    }
}

pub fn initialize() -> collections::HashMap<ast::Name, ast::Declaration> {
    let mut symbols = artithmetic::operator::declarations();
    symbols.extend(io::declarations());

    symbols
}

pub mod artithmetic {
    pub mod operator {
        use crate::{
            ast::{
                self,
                Constant::{self, *},
                Operator::{self, *},
            },
            runtime::ast::interpreter,
        };

        pub fn apply(lhs: &Constant, op: &Operator, rhs: &Constant) -> Option<Constant> {
            // Can this be done golfer than this?
            match (lhs, op, rhs) {
                (Float(lhs), Plus, Float(rhs)) => Some(Float(lhs + rhs)),
                (Int(lhs), Plus, Int(rhs)) => Some(Int(lhs + rhs)),

                (Float(lhs), Minus, Float(rhs)) => Some(Float(lhs - rhs)),
                (Int(lhs), Minus, Int(rhs)) => Some(Int(lhs - rhs)),

                (Float(lhs), Times, Float(rhs)) => Some(Float(lhs * rhs)),
                (Int(lhs), Times, Int(rhs)) => Some(Int(lhs * rhs)),

                (Float(lhs), Divides, Float(rhs)) => Some(Float(lhs / rhs)),
                (Int(lhs), Divides, Int(rhs)) => Some(Int(lhs / rhs)),

                (Float(lhs), Modulo, Float(rhs)) => Some(Float(lhs % rhs)),
                (Int(lhs), Modulo, Int(rhs)) => Some(Int(lhs % rhs)),

                (Float(lhs), LT, Float(rhs)) => Some(Boolean(lhs < rhs)),
                (Int(lhs), LT, Int(rhs)) => Some(Boolean(lhs < rhs)),

                (Float(lhs), Equals, Float(rhs)) => Some(Boolean(lhs == rhs)),
                (Int(lhs), Equals, Int(rhs)) => Some(Boolean(lhs == rhs)),
                (Boolean(lhs), Equals, Boolean(rhs)) => Some(Boolean(lhs == rhs)),
                (Text(lhs), Equals, Text(rhs)) => Some(Boolean(lhs == rhs)),
                (Void, Equals, Void) => Some(Boolean(true)),

                (Float(lhs), NotEqual, Float(rhs)) => Some(Boolean(lhs != rhs)),
                (Int(lhs), NotEqual, Int(rhs)) => Some(Boolean(lhs != rhs)),
                (Boolean(lhs), NotEqual, Boolean(rhs)) => Some(Boolean(lhs != rhs)),
                (Text(lhs), NotEqual, Text(rhs)) => Some(Boolean(lhs != rhs)),
                (Void, NotEqual, Void) => Some(Boolean(false)),

                (Float(lhs), GT, Float(rhs)) => Some(Boolean(lhs > rhs)),
                (Int(lhs), GT, Int(rhs)) => Some(Boolean(lhs > rhs)),

                (Float(lhs), LTE, Float(rhs)) => Some(Boolean(lhs <= rhs)),
                (Int(lhs), LTE, Int(rhs)) => Some(Boolean(lhs <= rhs)),

                (Float(lhs), GTE, Float(rhs)) => Some(Boolean(lhs >= rhs)),
                (Int(lhs), GTE, Int(rhs)) => Some(Boolean(lhs >= rhs)),

                (Boolean(lhs), And, Boolean(rhs)) => Some(Boolean(*lhs && *rhs)),
                (Boolean(lhs), Or, Boolean(rhs)) => Some(Boolean(*lhs || *rhs)),

                _otherwise => None,
            }
        }

        fn make_intrinsic_function(op: &Operator) -> ast::IntrinsicFunctionDeclarator {
            ast::IntrinsicFunctionDeclarator::new(
                &op.name(),
                &op.parameters(),
                op.return_type(),
                op.clone(),
            )
        }

        pub fn declarations() -> interpreter::SymbolTable {
            [
                Plus, Minus, Times, Divides, Modulo, Equals, NotEqual, LT, GT, LTE, GTE, And, Or,
            ]
            .iter()
            .map(make_intrinsic_function)
            .map(|x| (x.name.clone(), ast::Declaration::IntrinsicFunction(x)))
            .collect()
        }
    }
}

mod text {
    use crate::{ast, runtime::ast::interpreter};

    pub fn _apply_by_name(
        _symbol: &str,
        _arguments: &[ast::Constant],
    ) -> Result<ast::Constant, interpreter::Error> {
        todo!()
    }

    fn _interpolate(_components: Vec<ast::Constant>) -> ast::Constant {
        todo!()
    }
}

pub mod io {
    use super::IntrinsicProxy;
    use crate::{
        ast::{self, Name, TypeSelector},
        runtime::ast::interpreter,
    };

    #[derive(Debug)]
    pub struct PrintLineStub;

    impl IntrinsicProxy for PrintLineStub {
        // I shouldn't be passing around ast values, really. There should be an
        //   runtime::Value for this.
        fn invoke(&self, arguments: &[ast::Constant]) -> Result<ast::Constant, interpreter::Error> {
            match arguments {
                [line] => {
                    print_line(line.clone());
                    Ok(ast::Constant::Void)
                }
                _otherwise => Err(interpreter::Error::ExpectedArguments(
                    ast::Name::simple("print_line"),
                    vec![ast::Parameter {
                        name: ast::Name::simple("line"),
                        type_: ast::Select::Type(ast::TypeSelector::Named(Name::intrinsic("Text"))),
                    }],
                )),
            }
        }
    }

    pub fn declarations() -> interpreter::SymbolTable {
        // Perhaps this can be parsed later on
        // "fn print_line(line: <What here>) -> unit" => ast::InstrinsicFunctionDef
        let functions = vec![ast::IntrinsicFunctionDeclarator::new(
            &ast::Name::intrinsic("print_line"),
            &[ast::Parameter {
                name: ast::Name::simple("line"),
                type_: ast::Select::Type(ast::TypeSelector::Named(Name::intrinsic("Text"))),
            }],
            // Quite the mouthful - what am I doing here?!
            ast::Select::Type(TypeSelector::Named(ast::Name::intrinsic(
                &ast::PrimitiveType::Unit.to_string(),
            ))),
            PrintLineStub,
        )];

        functions
            .into_iter()
            .map(|x| (x.name().clone(), ast::Declaration::IntrinsicFunction(x)))
            .collect()
    }

    fn print_line(line: ast::Constant) {
        println!("{line}")
    }
}
