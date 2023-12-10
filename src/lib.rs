use runtime::ast::interpreter;

pub mod ast;
pub mod runtime;
pub mod kombi;
pub mod lexer;
pub mod syntax;


#[derive(Debug)]
pub enum Error {
    CompileTime(syntax::types::Error),
    Runtime(runtime::ast::interpreter::Error),
}

impl From<syntax::types::Error> for Error {
    fn from(value: syntax::types::Error) -> Self {
        Error::CompileTime(value)
    }
}

impl From<interpreter::Error> for Error {
    fn from(value: interpreter::Error) -> Self {
        Error::Runtime(value)
    }
}