use sylt::{
    ast::*,
    lexer,
    runtime::{interpreter::Interpreter, intrinsics},
    syntax, Error,
};

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

    if trenger_retina and saab >= 27 {
        print_line("Pain?");
    }

    return arith;
}
    
    "#;

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

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>())?;
    let interpreter = Interpreter::new(program, intrinsics::initialize());
    let return_value = interpreter.run()?;
    println!("{:#?}", return_value);

    Ok(())
}
