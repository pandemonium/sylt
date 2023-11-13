use sylt::{
    ast::*,
    lexer,
    runtime::{interpreter::{self, Interpreter}, intrinsics},
    syntax,
};

#[derive(Debug)]
enum Error {
    CompileTime(syntax::types::Error),
    Runtime(interpreter::Error),
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

fn main() -> Result<(), Error> {
    let source = r#"
    fn make_money() -> Text {
        if 1 + 2 {
            print_line("Hi, mom");
        } else {
            let bar = 10 + make_hay("Patrik Andersson", 97.0) * 1;
        }
        return "Hello, world"+1;
    }

    fn make_hay(whoami: Text, age: Int) -> Text {
        while 1 + 2 {
            print_line("Hi, mom");
        }
    }
 
{
    print_line("Hello, world");
    while True {
        if 12/34*56 + "Kalle" {
            let foo = 124 + 42.5 / 10 * 0.1;
        } else {
            save(427, "data.txt");
        }
    }
}    
    "#;

    let source = r#"
fn get_message() -> Text {
    return "Hi, mom";
}

fn compute_thing(quux: Int) -> Int {
    return 427 - quux;
}

fn other_stuff(bunk: Text) -> Int {
    return 2;
}

{
    let arith = compute_thing(13) * other_stuff("Hello world");
    let message = get_message();
    if True {
        print_line(message);
    } else {
        print_line("A quick brown fox?");
    }

    return arith;
}
    
    "#;

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>())?;
    let interpreter = Interpreter::new(program, intrinsics::initialize());
    let return_value = interpreter.run()?;
    println!("{:#?}", return_value);

    Ok(())
}
