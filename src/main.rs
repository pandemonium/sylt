use sylt::{
    ast::*,
    lexer,
    runtime::{interpreter::Interpreter, intrinsics},
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
        return fibonacci(30);
    }
    "#;

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>())?;
    let interpreter = Interpreter::new(program, intrinsics::initialize());
    let return_value = interpreter.run()?;
    println!("{:#?}", return_value);

    Ok(())
}
