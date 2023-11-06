use sylt::{
    ast::*,
    lexer,
    runtime::{
        interpreter,
        intrinsics::{artithmetic::operator, io},
    },
    syntax,
};

fn main() {
    let source = r#"
    fn make_hay(whoami: Text, age: Int) -> Text {
        while 1 + 2 {
            println("Hi, mom");
        }
    }fn make_money() -> Text {
        if 1 + 2 {
            println("Hi, mom");
        } else {
            let bar = 10 + make_hay("Patrik Andersson", 97.0) * 1;
        }
        return "Hello, world"+1;
    }
 
{
    println("Hello, world");
    while True {
        if 12/34*56 + "Kalle" {
            let foo = 124 + 42.5 / 10 * 0.1;
        } else {
            save(427, "data.txt");
        }
    }
}    
    "#;

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>());
    println!("{:?}", program);
}
