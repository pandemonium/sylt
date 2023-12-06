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
        return fibonacci(27);
    }
    "#;

//    let source = r#"
//    fn foo() -> Int {
//        while 1 < 2 {
//            3 + 4;
//        }
//    }
//
//    {
//        foo();
//    }
//    "#;

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>())?;
    let exe = vm::compile(program);
    println!("{exe}");

    //    let interpreter = Interpreter::new(program, intrinsics::initialize());
    //    let return_value = interpreter.run()?;
    //    println!("{:#?}", return_value);

    Ok(())
}
