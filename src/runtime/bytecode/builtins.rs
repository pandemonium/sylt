use super::{
    compiler,
    model::{self, BuiltinFunction, BuiltinStub},
};
use crate::ast;
use std::{io, rc};

pub fn provide_standard_library(artifact: &mut compiler::Compile) {
    artifact.register_builtin(model::BuiltinFunction {
        name: ast::Name::simple("format"),
        prototype: model::BuiltinFunctionPrototype::Varargs,
        stub: rc::Rc::new(Format),
    });
    artifact.register_builtin(make_builtin("print_line", &["Text"], PrintLine));
    artifact.register_builtin(make_builtin("read_line", &[], ReadLine));
}

fn make_builtin<A>(simple_name: &str, type_names: &[&str], stub: A) -> BuiltinFunction
where
    A: BuiltinStub + 'static,
{
    BuiltinFunction {
        name: ast::Name::simple(simple_name),
        prototype: model::BuiltinFunctionPrototype::Literally(
            type_names
                .into_iter()
                .map(|x| ast::Type::named(&ast::Name::intrinsic(x)))
                .collect(),
        ),
        stub: rc::Rc::new(stub),
    }
}

#[derive(Debug)]
pub struct PrintLine;

impl model::BuiltinStub for PrintLine {
    fn call(&self, parameters: &[model::Value]) -> Option<model::Value> {
        if let Some(model::Value::Text(message)) = parameters.get(0) {
            println!("{message}");
            Some(model::Value::Unit)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct ReadLine;

impl model::BuiltinStub for ReadLine {
    fn call(&self, _parameters: &[model::Value]) -> Option<model::Value> {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        Some(model::Value::Text(buffer[..(buffer.len() - 1)].into()))
    }
}

#[derive(Debug)]
pub struct Format;

impl model::BuiltinStub for Format {
    fn call(&self, parameters: &[model::Value]) -> Option<model::Value> {
        let mut buffer = String::new();
        for p in parameters {
            let image = match p {
                model::Value::Int(x) => format!("{x}"),
                model::Value::Float(x) => format!("{x}"),
                model::Value::Boolean(x) => format!("{x}"),
                model::Value::Text(x) => format!("{x}"),
                model::Value::Unit => format!("()"),
            };
            buffer.push_str(&image);
        }
        Some(model::Value::Text(buffer.into()))
    }
}
