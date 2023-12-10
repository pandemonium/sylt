use sylt::{
    runtime::{
        ast::interpreter::Interpreter,
        ast::intrinsics,
        bytecode::{self, vm},
    },
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
        return fibonacci(37);
    }
    "#;

    let source = r#"
        fn quux(n: Int) -> Int {
            let sum = 0;
            while n > 0 {
                let sum = sum + n;
                let n = n - 1;
            }
            
            return sum;
        }
    
        {
            return quux(20000000);
        }
        "#;

    let source = r#"
    {
        print_line("What is your name?");
        let name = read_line();
        print_line("Hello: ", name);
    }
    "#;

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>())?;
    let executable = bytecode::compiler::make_executable(program);
    let return_value = vm::Interpreter::new().run(executable);
    println!("Returns: {return_value}");

    // println!("{executable}");

    //    let interpreter = Interpreter::new(program, intrinsics::initialize());
    //    let return_value = interpreter.run()?;
    //    println!("{:#?}", return_value);

    Ok(())
}
