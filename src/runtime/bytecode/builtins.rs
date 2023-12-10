use super::{compiler, model};
use crate::ast;
use std::{rc, io};

pub fn provide_standard_library(compiler: &mut compiler::Compile) {
    compiler.register_builtin(model::BuiltinFunction {
        name: ast::Name::simple("print_line"),
        prototype: vec![ast::Type::named(&ast::Name::intrinsic("Text"))],
        stub: rc::Rc::new(PrintLine),
    });
    compiler.register_builtin(model::BuiltinFunction {
        name: ast::Name::simple("read_line"),
        prototype: vec![],
        stub: rc::Rc::new(ReadLine),
    });
}

// How would I make this vararg?
#[derive(Debug)]
pub struct PrintLine;

impl model::BuiltinStub for PrintLine {
    fn call(&self, parameters: &[model::Value]) -> Option<model::Value> {
        for p in parameters {
            let image = match p {
                model::Value::Int(x) => format!("{x}"),
                model::Value::Float(x) => format!("{x}"),
                model::Value::Boolean(x) => format!("{x}"),
                model::Value::Text(x) => format!("{x}"),
                model::Value::Unit => format!("("),
            };
            println!("{image}");
        }
        Some(model::Value::Unit)
    }
}

#[derive(Debug)]
pub struct ReadLine;

impl model::BuiltinStub for ReadLine {
    fn call(&self, parameters: &[model::Value]) -> Option<model::Value> {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        Some(model::Value::Text(buffer.into()))
    }
}
