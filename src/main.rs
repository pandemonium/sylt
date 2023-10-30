use sylt::{
    ast::*,
    interpreter,
    intrinsics::{artithmetic::operator, io},
};

fn main() {
    let ast = Program {
        definitions: vec![Declaration::Function(FunctionDef {
            name: Name::simple("print_hello_world"),
            parameters: vec![],
            return_type: Type::Unit,
            body: Block {
                statements: vec![Statement::Expression(Expression::Apply {
                    symbol: Select::Intrinsic(Name::std("io", "print_line")),
                    arguments: vec![Expression::Literal(Constant::Text("Hello, world".into()))],
                })],
            },
        })],
        entry_point: Block {
            statements: vec![
                Statement::Expression(Expression::Apply {
                    symbol: Select::Function(Name::simple("print_hello_world")),
                    arguments: vec![],
                }),
                Statement::Return(Expression::Literal(Constant::Int(1))),
            ],
        },
    };

//    println!("{ast:#?}");
    // Declarations are just super complicated and I hate them.
    // Names and types are also... dumb. Mostly because of the Name namespace
    // that I think I need. That I don't.
    let mut builtins = operator::declarations();
    builtins.extend(io::declarations());

    let interpreter = interpreter::Interpreter::new(ast, builtins);
    let result = interpreter.run();

    println!("--------> {result:#?}");
}
