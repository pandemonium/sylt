use sylt::{
    ast::*,
    lexer,
    runtime::{interpreter::Interpreter, intrinsics, vm},
    syntax, Error,
};

fn main() -> Result<(), Error> {
    let source = r#"
    fn fibonacci(n: Int) -> Int {
        if n == 0 {
            return 0;
        } else {
            if n == 1 {
                return 1;
            } else {
                return fibonacci(n - 1) + fibonacci(n - 2);
            }
        }
    }

    {
        return fibonacci(35);
    }
    "#;

//    let source = r#"
//    fn quux(x: Int) -> Int {
//        return x;
//    }
//
//    {
//        return quux(3 - 1);
//    }
//    "#;

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>())?;
    let executable = vm::compile(program);
    let return_value = vm::Interpreter::new(executable).run();
    println!("Returns: {return_value}");

//    println!("{executable}");

//    let interpreter = Interpreter::new(program, intrinsics::initialize());
//    let return_value = interpreter.run()?;
//    println!("{:#?}", return_value);

    Ok(())
}
